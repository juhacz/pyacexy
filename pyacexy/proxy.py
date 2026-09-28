"""Main proxy server implementation"""
import argparse
import asyncio
import logging
import os
import uuid
from typing import Optional, Dict
from urllib.parse import parse_qs, urlparse, urlunparse, urlencode

import aiohttp
from aiohttp import web, ClientSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AceStream:
    """AceStream session information"""

    def __init__(self, playback_url: str, stat_url: str, command_url: str, stream_id: str):
        self.playback_url = playback_url
        self.stat_url = stat_url
        self.command_url = command_url
        self.stream_id = stream_id


class OngoingStream:
    """Represents an ongoing stream with multiple clients.

    Clients are tracked by monotonically increasing integer IDs rather than
    id(response) to avoid collisions when CPython reuses memory addresses
    after garbage collection.

    pending_count tracks clients that are in the process of joining (waiting
    for the stream to start) but haven't been added to the clients dict yet.
    The fetch loop checks both clients and pending_count before deciding to
    stop, preventing premature shutdown when a client is mid-handshake.
    """

    def __init__(self, stream_id: str, acestream: AceStream):
        self.stream_id = stream_id
        self.acestream = acestream
        self.clients: Dict[int, web.StreamResponse] = {}
        self.client_last_write: Dict[int, float] = {}
        self._next_client_id = 0
        self.task: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()
        self.done = asyncio.Event()
        self.started = asyncio.Event()
        self.first_chunk = asyncio.Event()
        self.stopping = False
        self.pending_count = 0
        self.error: Optional[str] = None

    def add_client(self, response: web.StreamResponse) -> int:
        """Add client and return unique client_id. Must be called under lock."""
        self._next_client_id += 1
        cid = self._next_client_id
        self.clients[cid] = response
        self.client_last_write[cid] = asyncio.get_event_loop().time()
        return cid

    def remove_client(self, client_id: int) -> bool:
        """Remove client by ID. Must be called under lock. Returns True if was present."""
        was_present = client_id in self.clients
        self.clients.pop(client_id, None)
        self.client_last_write.pop(client_id, None)
        return was_present


class AcexyProxy:
    """AceStream HTTP Proxy Server"""

    def __init__(
            self,
            acestream_host: str = "localhost",
            acestream_port: int = 6878,
            scheme: str = "http",
            buffer_size: int = 4 * 1024 * 1024,
            m3u8_mode: bool = False,
            empty_timeout: float = 60.0,
            no_response_timeout: float = 1.0,
            stream_timeout: float = 60.0,
            write_timeout: float = 0.5,
    ):
        self.acestream_host = acestream_host
        self.acestream_port = acestream_port
        self.scheme = scheme
        self.buffer_size = buffer_size
        self.m3u8_mode = m3u8_mode
        self.empty_timeout = empty_timeout
        self.no_response_timeout = no_response_timeout
        self.stream_timeout = stream_timeout
        self.write_timeout = write_timeout
        self.endpoint = "/ace/manifest.m3u8" if m3u8_mode else "/ace/getstream"

        self.streams: Dict[str, OngoingStream] = {}
        self.session: Optional[ClientSession] = None
        self.streams_lock = asyncio.Lock()

    async def _fetch_stream_info(self, stream_id: str, infohash: str, extra_params: dict) -> AceStream:
        """Fetch stream information from AceStream middleware"""
        # Generate temporary PID for this request
        temp_pid = str(uuid.uuid4())

        # Build request URL
        url = f"{self.scheme}://{self.acestream_host}:{self.acestream_port}{self.endpoint}"

        # Build query parameters
        params = extra_params.copy()
        params['format'] = 'json'
        params['pid'] = temp_pid

        if stream_id:
            params['id'] = stream_id
        elif infohash:
            params['infohash'] = infohash
        else:
            raise ValueError("Either id or infohash must be provided")

        logger.debug(f"Fetching stream info: {url}?{urlencode(params)}")

        # Set timeout for this request
        timeout = aiohttp.ClientTimeout(total=self.no_response_timeout)

        async with self.session.get(url, params=params, timeout=timeout) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"AceStream middleware returned {response.status}: {error_text}")

            data = await response.json()

            if 'error' in data and data['error']:
                raise Exception(f"AceStream error: {data['error']}")

            if 'response' not in data:
                raise Exception("Invalid response from AceStream middleware")

            resp = data['response']

            return AceStream(
                playback_url=resp['playback_url'],
                stat_url=resp.get('stat_url', ''),
                command_url=resp['command_url'],
                stream_id=stream_id or infohash
            )

    async def _close_stream(self, acestream: AceStream):
        """Close stream on AceStream middleware"""
        try:
            parsed = urlparse(acestream.command_url)
            query = parse_qs(parsed.query)
            query["method"] = ["stop"]
            url = urlunparse(parsed._replace(query=urlencode(query, doseq=True)))
            logger.debug(f"Closing stream: {url}")
            close_timeout = aiohttp.ClientTimeout(total=10.0)
            async with self.session.get(url, timeout=close_timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'error' in data and data['error']:
                        logger.warning(f"Error closing stream: {data['error']}")
                else:
                    logger.warning(f"Failed to close stream, status: {response.status}")
        except Exception as e:
            logger.warning(f"Exception while closing stream: {e}")

    async def _get_or_create_stream(
            self, key: str, stream_id: str, infohash: str, extra_params: dict
    ) -> Optional[OngoingStream]:
        """Get existing stream or create a new one.

        Uses check-fetch-recheck pattern: the expensive _fetch_stream_info
        HTTP call happens OUTSIDE streams_lock so it doesn't block all
        incoming requests. If two handlers race to create the same stream,
        the second one detects the duplicate under lock and closes its
        redundant AceStream session.
        """
        # First check: can we reuse an existing stream?
        async with self.streams_lock:
            if key in self.streams:
                ongoing = self.streams[key]
                if not ongoing.stopping and not ongoing.done.is_set():
                    logger.info(f"Reusing existing stream for {key}")
                    return ongoing
                # Stream is stopping/done, need a new one

        # Fetch stream info OUTSIDE the lock (this may take seconds)
        try:
            acestream = await self._fetch_stream_info(stream_id, infohash, extra_params)
        except Exception as e:
            logger.error(f"Failed to fetch stream info: {e}")
            return None

        # Recheck under lock: another handler may have created the stream
        # while we were doing the HTTP fetch
        redundant_acestream = None
        async with self.streams_lock:
            if key in self.streams:
                existing = self.streams[key]
                if not existing.stopping and not existing.done.is_set():
                    # Another handler beat us — reuse theirs, discard ours
                    redundant_acestream = acestream
                    ongoing = existing
                    logger.info(f"Another handler created stream for {key}, reusing")
                else:
                    ongoing = OngoingStream(key, acestream)
                    self.streams[key] = ongoing
                    logger.info(f"Created new stream for {key}")
            else:
                ongoing = OngoingStream(key, acestream)
                self.streams[key] = ongoing
                logger.info(f"Created new stream for {key}")

        # Close redundant AceStream session outside all locks
        if redundant_acestream is not None:
            await self._close_stream(redundant_acestream)

        return ongoing

    async def _start_acestream_fetch(self, ongoing: OngoingStream):
        """Fetch stream from AceStream and distribute to all clients"""
        logger.info(f"Starting AceStream fetch for {ongoing.stream_id}")

        # Set timeout for reading from AceStream
        timeout = aiohttp.ClientTimeout(sock_read=self.empty_timeout)

        try:
            logger.debug(f"Connecting to AceStream playback URL: {ongoing.acestream.playback_url}")
            async with self.session.get(ongoing.acestream.playback_url, timeout=timeout) as ace_response:
                logger.debug(f"AceStream response status: {ace_response.status}")
                if ace_response.status != 200:
                    logger.error(f"AceStream returned status {ace_response.status}")
                    ongoing.error = f"AceStream returned status {ace_response.status}"
                    ongoing.started.set()
                    return

                # Signal that connection is established BEFORE reading chunks
                ongoing.started.set()
                logger.info(
                    f"AceStream connection established for {ongoing.stream_id}, "
                    f"starting to read chunks"
                )

                # Read chunks and distribute to all clients
                chunk_count = 0
                last_cleanup = asyncio.get_event_loop().time()
                async for chunk in ace_response.content.iter_chunked(8192):
                    if not chunk:
                        break

                    chunk_count += 1
                    if chunk_count % 100 == 0:
                        logger.debug(f"Stream {ongoing.stream_id} sent {chunk_count} chunks")

                    # Signal first chunk on read (not on successful write) so
                    # waiting clients unblock even if there are no active clients yet
                    if chunk_count == 1:
                        ongoing.first_chunk.set()

                    # Get client snapshot and do periodic cleanup under a single lock
                    # acquisition (reduced from 3 separate acquisitions per chunk)
                    current_time = asyncio.get_event_loop().time()
                    do_cleanup = current_time - last_cleanup > 15

                    async with ongoing.lock:
                        # Periodic stale client cleanup
                        if do_cleanup:
                            last_cleanup = current_time
                            stale_ids = [
                                cid for cid, last_write in list(ongoing.client_last_write.items())
                                if current_time - last_write > 30
                            ]
                            for cid in stale_ids:
                                resp = ongoing.clients.pop(cid, None)
                                ongoing.client_last_write.pop(cid, None)
                                if resp is not None:
                                    logger.warning(
                                        f"Client {cid} inactive for "
                                        f"{current_time - ongoing.client_last_write.get(cid, 0):.0f}s, "
                                        f"removing"
                                    )
                                    try:
                                        await resp.write_eof()
                                    except BaseException:
                                        pass
                            if stale_ids:
                                logger.info(
                                    f"Removed {len(stale_ids)} stale client(s) "
                                    f"from stream {ongoing.stream_id}"
                                )

                        # Snapshot current clients for writing outside the lock
                        client_snapshot = dict(ongoing.clients)

                    # Send to all connected clients without holding the lock
                    dead_ids = []
                    for cid, client_response in client_snapshot.items():
                        try:
                            await asyncio.wait_for(
                                client_response.write(chunk),
                                timeout=self.write_timeout
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                f"Timeout writing to client {cid} ({self.write_timeout}s)"
                            )
                            dead_ids.append(cid)
                        except Exception as e:
                            logger.warning(f"Error writing to client {cid}: {e}")
                            dead_ids.append(cid)

                    # Update state under lock
                    no_clients_left = False
                    async with ongoing.lock:
                        # Update write timestamps for successful clients
                        write_time = asyncio.get_event_loop().time()
                        for cid in client_snapshot:
                            if cid not in dead_ids and cid in ongoing.clients:
                                ongoing.client_last_write[cid] = write_time

                        # Remove dead clients
                        for cid in dead_ids:
                            resp = ongoing.clients.pop(cid, None)
                            ongoing.client_last_write.pop(cid, None)
                            if resp is not None:
                                try:
                                    await resp.write_eof()
                                except BaseException:
                                    pass

                        if dead_ids:
                            logger.info(
                                f"Removed {len(dead_ids)} dead client(s) from stream "
                                f"{ongoing.stream_id}, {len(ongoing.clients)} client(s) remaining"
                            )

                        # Stop only when ALL clients are gone AND no pending
                        # clients are in the process of joining
                        if not ongoing.clients and ongoing.pending_count == 0:
                            logger.info(
                                f"No clients left for stream {ongoing.stream_id}, stopping"
                            )
                            ongoing.stopping = True
                            no_clients_left = True

                    if no_clients_left:
                        break

        except asyncio.CancelledError:
            logger.info(f"Stream {ongoing.stream_id} fetch cancelled")
            ongoing.error = ongoing.error or "Stream cancelled"
        except asyncio.TimeoutError:
            logger.info(
                f"Stream {ongoing.stream_id} timed out "
                f"(no data for {self.empty_timeout}s)"
            )
            ongoing.error = ongoing.error or f"Stream timed out (no data for {self.empty_timeout}s)"
        except Exception as e:
            logger.error(f"Error fetching AceStream: {e}")
            ongoing.error = ongoing.error or str(e)
        finally:
            # Ensure events are set so any waiting clients don't hang forever
            ongoing.started.set()
            ongoing.first_chunk.set()

            # Clean up all remaining clients
            async with ongoing.lock:
                for cid, client_response in list(ongoing.clients.items()):
                    try:
                        await client_response.write_eof()
                    except BaseException:
                        pass
                ongoing.clients.clear()
                ongoing.client_last_write.clear()

            # Close the stream on AceStream middleware
            await self._close_stream(ongoing.acestream)

            # Signal stream is done
            ongoing.done.set()

            # Remove stream from active streams
            async with self.streams_lock:
                if self.streams.get(ongoing.stream_id) is ongoing:
                    del self.streams[ongoing.stream_id]
                    logger.info(f"Stream {ongoing.stream_id} cleaned up")

    async def handle_getstream(self, request: web.Request) -> web.StreamResponse:
        """Handle /ace/getstream endpoint"""
        # Get stream ID or infohash from query parameters
        stream_id = request.query.get('id', '')
        infohash = request.query.get('infohash', '')

        if not stream_id and not infohash:
            return web.Response(status=400, text="Missing id or infohash parameter")

        if stream_id and infohash:
            return web.Response(status=400, text="Only one of id or infohash can be specified")

        # Check if PID was provided (not allowed)
        if 'pid' in request.query:
            return web.Response(status=400, text="PID parameter is not allowed")

        # Use stream_id or infohash as the key
        key = stream_id or infohash

        logger.info(
            f"Client {request.remote} requesting stream {key} "
            f"(User-Agent: {request.headers.get('User-Agent', 'unknown')})"
        )

        # Get extra parameters
        extra_params = {k: v for k, v in request.query.items()
                        if k not in ('id', 'infohash', 'pid')}

        # Get or create ongoing stream (fetch happens outside streams_lock)
        ongoing = await self._get_or_create_stream(key, stream_id, infohash, extra_params)
        if ongoing is None:
            return web.Response(status=500, text="Failed to start stream")

        # Register as pending client and optionally start the fetch task
        need_to_start = False
        async with ongoing.lock:
            if ongoing.done.is_set() or ongoing.stopping:
                return web.Response(status=503, text="Stream is ending, please retry")
            ongoing.pending_count += 1
            if ongoing.task is None:
                ongoing.task = asyncio.create_task(
                    self._start_acestream_fetch(ongoing)
                )
                need_to_start = True

        # Serve the stream — handles all cleanup of pending_count and client
        return await self._serve_stream(request, ongoing, key, need_to_start)

    async def _serve_stream(
            self,
            request: web.Request,
            ongoing: OngoingStream,
            key: str,
            need_to_start: bool,
    ) -> web.StreamResponse:
        """Serve stream data to a single client.

        Caller must have already incremented ongoing.pending_count.
        This method always cleans up correctly:
          - If client_id was assigned: removes client from ongoing.clients
          - If client_id was NOT assigned: decrements ongoing.pending_count
        This ensures the fetch loop's stop-condition (no clients + no pending)
        is always accurate.
        """
        client_id = None
        response = None

        try:
            # Wait for stream to be ready BEFORE calling response.prepare().
            # This allows returning clean HTTP error responses (4xx/5xx) on
            # failure — once prepare() is called, HTTP headers are already
            # sent and we can only close the connection.
            if need_to_start:
                try:
                    await asyncio.wait_for(ongoing.started.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    logger.error(f"Timeout waiting for stream {key} to connect")
                    return web.Response(
                        status=503,
                        text="Stream failed to start: connection timeout"
                    )

                # Check if stream reported an error during startup
                if ongoing.error and ongoing.done.is_set():
                    return web.Response(
                        status=502,
                        text=f"Stream error: {ongoing.error}"
                    )

                try:
                    await asyncio.wait_for(
                        ongoing.first_chunk.wait(),
                        timeout=self.stream_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Timeout waiting for stream {key} first data")
                    return web.Response(
                        status=503,
                        text="Stream failed to start: no data received"
                    )

            # Stream is confirmed working — NOW prepare the response
            response = web.StreamResponse()
            response.content_type = (
                'application/x-mpegURL' if self.m3u8_mode else 'video/MP2T'
            )
            if not self.m3u8_mode:
                response.headers['Transfer-Encoding'] = 'chunked'
            await response.prepare(request)

            # Transition from pending to active client
            async with ongoing.lock:
                ongoing.pending_count -= 1
                client_id = ongoing.add_client(response)
                logger.info(
                    f"Stream {key} now has {len(ongoing.clients)} client(s)"
                )

            # Wait for the stream to finish
            # Client disconnect is detected by write errors in the fetch loop
            await ongoing.done.wait()
            logger.debug(f"Stream finished for {key}")
            return response

        except asyncio.CancelledError:
            logger.debug(f"Handler cancelled for stream {key}")
            raise
        except Exception as e:
            logger.error(f"Error serving stream {key}: {e}")
            # If response was already prepared (headers sent), we must return
            # it — can't send a new Response after headers are on the wire
            if response is not None and response.prepared:
                return response
            return web.Response(status=500, text=f"Internal error: {e}")
        finally:
            # Clean up: remove as active client OR decrement pending count
            if client_id is not None:
                async with ongoing.lock:
                    was_present = ongoing.remove_client(client_id)
                    client_count = len(ongoing.clients)
                    if was_present:
                        logger.info(
                            f"Handler cleanup: removed client from stream "
                            f"{key}, {client_count} client(s) remaining"
                        )
                    else:
                        logger.debug(
                            f"Handler cleanup: client already removed from "
                            f"stream {key}, {client_count} client(s) remaining"
                        )
            else:
                # Still pending — decrement count so the fetch loop can
                # detect "no clients + no pending" and stop if needed
                async with ongoing.lock:
                    ongoing.pending_count -= 1

            # Close response if it was prepared
            if response is not None and response.prepared:
                try:
                    await response.write_eof()
                except BaseException:
                    pass

    async def handle_status(self, request: web.Request) -> web.Response:
        """Handle /ace/status endpoint"""
        stream_id = request.query.get('id', '')
        infohash = request.query.get('infohash', '')

        # Global status
        if not stream_id and not infohash:
            async with self.streams_lock:
                status = {
                    'streams': len(self.streams)
                }
            return web.json_response(status)

        # Get stream reference under streams_lock, then release
        key = stream_id or infohash
        async with self.streams_lock:
            ongoing = self.streams.get(key)

        if ongoing is None:
            return web.Response(status=404, text="Stream not found")

        # Access stream data under its own lock — NOT nested inside
        # streams_lock to prevent deadlock with _start_acestream_fetch's
        # finally block (which takes ongoing.lock then streams_lock)
        async with ongoing.lock:
            status = {
                'clients': len(ongoing.clients),
                'pending': ongoing.pending_count,
                'stream_id': key,
                'stat_url': ongoing.acestream.stat_url
            }
        return web.json_response(status)

    async def start_server(self, host: str = "0.0.0.0", port: int = 8080):
        """Start the proxy server"""
        self.session = ClientSession()

        app = web.Application()
        app.router.add_get('/ace/getstream', self.handle_getstream)
        app.router.add_get('/ace/getstream/', self.handle_getstream)
        app.router.add_get('/ace/status', self.handle_status)

        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, host, port)
        await site.start()

        logger.info(f"PyAcexy proxy started on {host}:{port}")
        logger.info(f"Connecting to AceStream at {self.scheme}://{self.acestream_host}:{self.acestream_port}")
        logger.info(f"Endpoint mode: {'M3U8/HLS' if self.m3u8_mode else 'MPEG-TS'}")

        # Keep running
        try:
            await asyncio.Event().wait()
        finally:
            await self.session.close()
            await runner.cleanup()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="PyAcexy - AceStream HTTP Proxy")
    parser.add_argument(
        "--host",
        default=os.getenv("ACEXY_HOST", "localhost"),
        help="AceStream middleware host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("ACEXY_PORT", "6878")),
        help="AceStream middleware port (default: 6878)"
    )
    parser.add_argument(
        "--listen-addr",
        default=os.getenv("ACEXY_LISTEN_ADDR", ":8080"),
        help="Address to listen on (format: [host]:port, default: :8080)"
    )
    parser.add_argument(
        "--scheme",
        default=os.getenv("ACEXY_SCHEME", "http"),
        help="AceStream middleware scheme (http/https, default: http)"
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=int(os.getenv("ACEXY_BUFFER_SIZE", str(4 * 1024 * 1024))),
        help="Buffer size in bytes (default: 4MB)"
    )
    parser.add_argument(
        "--m3u8",
        action="store_true",
        default=os.getenv("ACEXY_M3U8", "").lower() == "true",
        help="Enable M3U8/HLS mode (default: False)"
    )
    parser.add_argument(
        "--empty-timeout",
        type=float,
        default=float(os.getenv("ACEXY_EMPTY_TIMEOUT", "60")),
        help="Timeout in seconds for empty stream data (default: 60s)"
    )
    parser.add_argument(
        "--no-response-timeout",
        type=float,
        default=float(os.getenv("ACEXY_NO_RESPONSE_TIMEOUT", "10")),
        help="Timeout in seconds for AceStream middleware response (default: 10s)"
    )
    parser.add_argument(
        "--m3u8-stream-timeout",
        type=float,
        default=float(os.getenv("ACEXY_M3U8_STREAM_TIMEOUT", "60")),
        help="Timeout in seconds for M3U8 stream (default: 60s)"
    )
    parser.add_argument(
        "--write-timeout",
        type=float,
        default=float(os.getenv("ACEXY_WRITE_TIMEOUT", "5")),
        help="Timeout in seconds for writing to client (default: 5s)"
    )

    args = parser.parse_args()

    # Parse listen address
    listen_parts = args.listen_addr.split(":")
    listen_host = listen_parts[0] if listen_parts[0] else "0.0.0.0"
    listen_port = int(listen_parts[1]) if len(listen_parts) > 1 else 8080

    # Create and start proxy
    proxy = AcexyProxy(
        acestream_host=args.host,
        acestream_port=args.port,
        scheme=args.scheme,
        buffer_size=args.buffer_size,
        m3u8_mode=args.m3u8,
        empty_timeout=args.empty_timeout,
        no_response_timeout=args.no_response_timeout,
        stream_timeout=args.m3u8_stream_timeout,
        write_timeout=args.write_timeout,
    )

    try:
        asyncio.run(proxy.start_server(listen_host, listen_port))
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    main()
