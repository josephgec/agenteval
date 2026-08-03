"""The proxy that runs inside the gateway container.

Stdlib only, single file, no arguments: it is copied into a container and run
there, so it cannot import anything from agenteval. It is a real module rather
than a string constant so the matching logic below can be imported and tested
in this process, which is where the interesting bugs are.

Scope, stated plainly. This forwards, it does not intercept. For HTTPS it sees
the host in the CONNECT line and the byte counts, and nothing else — no path,
no body. Reading inside TLS would mean installing a CA the container trusts,
and a harness that can decrypt the traffic of the code it is evaluating is a
larger security surface than the one it closes. Host and volume is the level
this operates at; the docstring in proxy.py says the same thing to anyone
reading the results.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

PORT = int(os.environ.get("AGENTEVAL_PROXY_PORT", "8080"))
#: Comma-separated. Empty means nothing is allowed out, which is the safe
#: reading of a misconfiguration rather than the convenient one.
ALLOW = [h.strip().lower() for h in os.environ.get("AGENTEVAL_ALLOW", "").split(",") if h.strip()]

#: Bigger than any plausible header block, small enough that a client sending
#: an endless one cannot exhaust the gateway's memory.
MAX_HEAD = 64 * 1024


def allowed(host: str, allow: list[str] | None = None) -> bool:
    """Exact host, or any subdomain of an allowed host.

    `pypi.org` admits `pypi.org` and `files.pythonhosted.pypi.org` but not
    `notpypi.org` — the leading dot is what stops a suffix match from being a
    substring match, which is the classic hole in allowlists of this shape.
    """
    entries = ALLOW if allow is None else allow
    host = host.lower().rstrip(".").partition(":")[0]
    return any(host == entry or host.endswith("." + entry) for entry in entries)


def record(**fields: object) -> None:
    print(json.dumps({"ts": round(time.time(), 3), **fields}), flush=True)


def target_of(request_line: str, headers: dict[str, str]) -> tuple[str, int, str]:
    """Host, port and the URL as the client asked for it.

    Raises on anything that is not a request line. Guessing instead would put
    the guess in the egress log — `this is not http` parses very happily as a
    request for `is not` by the `THIS` method — and an audit trail with
    invented entries in it is worse than one with a gap.
    """
    parts = request_line.split(" ")
    if len(parts) != 3 or not parts[2].startswith("HTTP/"):
        raise ValueError(f"not an HTTP request line: {request_line[:60]!r}")
    method, url = parts[0], parts[1].strip() or "/"
    if method.upper() == "CONNECT":
        host, _, port = url.partition(":")
        return host, int(port or 443), url
    if "://" in url:
        authority = url.split("://", 1)[1].split("/", 1)[0]
    else:
        # Origin-form to a proxy is unusual but legal; the Host header is then
        # the only place the destination appears.
        authority = headers.get("host", "")
    host, _, port = authority.partition(":")
    return host, int(port or 80), url


async def read_head(reader: asyncio.StreamReader) -> tuple[str, dict[str, str], bytes]:
    raw = await reader.readuntil(b"\r\n\r\n")
    if len(raw) > MAX_HEAD:
        raise ValueError("header block too large")
    text = raw.decode("latin-1")
    lines = text.split("\r\n")
    headers = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        if value:
            headers[name.strip().lower()] = value.strip()
    return lines[0], headers, raw


async def pump(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, counter: list[int]
) -> None:
    try:
        while chunk := await reader.read(65536):
            counter[0] += len(chunk)
            writer.write(chunk)
            await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        writer.close()


async def refuse(writer: asyncio.StreamWriter, status: str, message: str) -> None:
    body = message.encode()
    writer.write(
        f"HTTP/1.1 {status}\r\nContent-Length: {len(body)}\r\n"
        f"Content-Type: text/plain\r\nConnection: close\r\n\r\n".encode() + body
    )
    try:
        await writer.drain()
    except ConnectionError:
        pass
    writer.close()


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        request_line, headers, raw = await read_head(reader)
    except Exception:  # noqa: BLE001 - a malformed client is not our problem
        writer.close()
        return

    method = request_line.partition(" ")[0].upper()
    try:
        host, port, url = target_of(request_line, headers)
    except ValueError as exc:
        record(method=method, allowed=False, reason=str(exc))
        await refuse(writer, "400 Bad Request", "could not parse the destination")
        return

    if not host or not allowed(host):
        record(host=host, port=port, method=method, url=url, allowed=False,
               reason="not in the allowlist")
        await refuse(
            writer,
            "403 Forbidden",
            f"agenteval: egress to {host} is not permitted by this task's "
            f"allowlist ({', '.join(ALLOW) or 'empty'}).",
        )
        return

    try:
        remote_reader, remote_writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=20
        )
    except Exception as exc:  # noqa: BLE001
        record(host=host, port=port, method=method, url=url, allowed=True,
               reason=f"upstream failed: {type(exc).__name__}")
        await refuse(writer, "502 Bad Gateway", f"could not reach {host}:{port}")
        return

    up, down = [0], [0]
    if method == "CONNECT":
        writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
        await writer.drain()
    else:
        # The head is replayed as-is and the rest of the connection spliced,
        # rather than parsed. Bodies, chunked encoding and streaming responses
        # then need no handling at all, which is a large amount of protocol
        # this does not have to get right.
        remote_writer.write(raw)
        await remote_writer.drain()
        # The head was already forwarded, so it has to be counted here or the
        # volume reported for a plain GET comes out as zero.
        up[0] = len(raw)

    await asyncio.gather(
        pump(reader, remote_writer, up),
        pump(remote_reader, writer, down),
    )
    record(host=host, port=port, method=method, url=url, allowed=True,
           bytes_up=up[0], bytes_down=down[0])


async def main() -> None:
    server = await asyncio.start_server(handle, "0.0.0.0", PORT)
    record(event="listening", port=PORT, allow=ALLOW)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
