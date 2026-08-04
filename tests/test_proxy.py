"""Monitored egress.

Split three ways. The allowlist and request parsing are pure functions in the
proxy server module and are tested here in-process — they are where the
interesting bugs live. The Docker topology is asserted on the flags, because
the topology *is* the control. The containment tests need the exec image and a
working network, and skip cleanly without either.
"""

import asyncio
import json
import subprocess

import pytest

from agenteval.exec import Environment, EnvironmentSpec, Proxy, ProxyError
from agenteval.exec import environment as env_mod
from agenteval.exec import _proxy_server as server
from agenteval.exec.environment import EnvironmentError_

IMAGE = env_mod.DEFAULT_IMAGE


# --------------------------------------------------------------------------- #
# The allowlist
# --------------------------------------------------------------------------- #


ALLOW = ["pypi.org", "github.com"]


@pytest.mark.parametrize("host", ["pypi.org", "PyPI.org", "files.pypi.org", "pypi.org."])
def test_an_allowed_host_and_its_subdomains(host):
    assert server.allowed(host, ALLOW)


@pytest.mark.parametrize(
    "host",
    [
        "notpypi.org",       # the classic suffix-match hole
        "pypi.org.evil.com",  # allowed name as a prefix of somewhere else
        "evil.com",
        "",
    ],
)
def test_hosts_that_only_look_allowed(host):
    """A leading dot is what makes a suffix match a subdomain match. Without it
    `notpypi.org` ends with `pypi.org` and the allowlist is decorative."""
    assert not server.allowed(host, ALLOW)


def test_an_empty_allowlist_permits_nothing():
    """The safe reading of a misconfiguration rather than the convenient one."""
    assert not server.allowed("pypi.org", [])


def test_a_port_does_not_defeat_the_match():
    assert server.allowed("pypi.org:443", ALLOW)


# --------------------------------------------------------------------------- #
# Reading the request
# --------------------------------------------------------------------------- #


def test_the_destination_of_a_connect():
    assert server.target_of("CONNECT pypi.org:443 HTTP/1.1", {}) == (
        "pypi.org", 443, "pypi.org:443"
    )


def test_connect_defaults_to_443():
    assert server.target_of("CONNECT pypi.org HTTP/1.1", {})[1] == 443


def test_the_destination_of_an_absolute_uri():
    host, port, url = server.target_of("GET http://pypi.org/simple/ HTTP/1.1", {})
    assert (host, port, url) == ("pypi.org", 80, "http://pypi.org/simple/")


def test_a_non_standard_port_is_carried_through():
    assert server.target_of("GET http://pypi.org:8443/x HTTP/1.1", {})[1] == 8443


def test_origin_form_falls_back_to_the_host_header():
    """Unusual against a proxy but legal, and then the header is the only place
    the destination appears."""
    assert server.target_of("GET /simple/ HTTP/1.1", {"host": "pypi.org"})[0] == (
        "pypi.org"
    )


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_naming_hosts_is_how_a_task_asks_for_egress():
    spec = EnvironmentSpec.from_config({"allow_hosts": ["pypi.org"]})
    assert spec.allow_hosts == ["pypi.org"]
    assert spec.network == "none"  # replaced at start by the internal network


def test_an_allowlist_beside_an_open_network_is_refused():
    """Not a stricter policy — a misleading one. Anyone reading the task would
    take the list for the boundary when `bridge` had already opened everything."""
    with pytest.raises(EnvironmentError_, match="already grants unrestricted"):
        EnvironmentSpec.from_config(
            {"allow_hosts": ["pypi.org"], "network": "bridge"}
        )


def test_a_proxy_with_nothing_allowed_is_a_configuration_error():
    with pytest.raises(ProxyError, match="at least one host"):
        Proxy([], image=IMAGE)


def test_the_workspace_is_told_where_the_gateway_is():
    """Both cases: curl and git read the lowercase names, most Python libraries
    read the uppercase ones."""
    variables = Proxy(["pypi.org"], image=IMAGE).environment_variables()
    assert any(v.startswith("HTTPS_PROXY=http://agenteval-egress:") for v in variables)
    assert any(v.startswith("https_proxy=http://agenteval-egress:") for v in variables)


def test_the_gateway_container_is_locked_down_too():
    """It is the one container on the internal network that can reach the
    internet, which makes it the one worth hardening most."""
    proxy = Proxy(["pypi.org"], image=IMAGE)
    calls = []
    proxy._run = lambda *args, **kw: calls.append(args) or _ok()
    try:
        proxy.start()
    except ProxyError:
        pass
    run = next(c for c in calls if c[0] == "run")
    for flag in ("--read-only", "--cap-drop", "--security-opt", "--pids-limit"):
        assert flag in run


def _ok(stdout="x"):
    return subprocess.CompletedProcess([], 0, stdout, "")


def test_the_network_is_created_internal():
    """The whole control. Containers on an internal network have no route off
    the host, so the gateway is not a policy the workspace is asked to respect
    — it is the only wire there is."""
    proxy = Proxy(["pypi.org"], image=IMAGE)
    calls = []
    proxy._run = lambda *args, **kw: calls.append(args) or _ok()
    try:
        proxy.start()
    except ProxyError:
        pass
    assert ("network", "create", "--internal", proxy.network or calls[0][3]) == calls[0]


def test_the_workspace_joins_the_internal_network_not_the_declared_one():
    spec = EnvironmentSpec(allow_hosts=["pypi.org"])
    environment = Environment(spec)
    environment.proxy = Proxy(["pypi.org"], image=IMAGE)
    environment.proxy.network = "agenteval-net-test"
    args = environment._run_args()
    assert args[args.index("--network") + 1] == "agenteval-net-test"
    assert "HTTPS_PROXY=http://agenteval-egress:8080" in args


def test_a_run_without_an_allowlist_starts_no_gateway():
    assert Environment(EnvironmentSpec()).proxy is None


def test_the_gateway_runs_on_the_harnesses_image_not_the_tasks(monkeypatch):
    """The gateway is infrastructure. A benchmark image is frequently gigabytes
    and frequently emulated, and running an HTTP proxy inside an emulated 4 GB
    SWE-bench image works and is absurd."""
    seen = {}

    def build(allow_hosts, image, **kwargs):
        seen["image"] = image
        raise ProxyError("stop here")

    monkeypatch.setattr("agenteval.exec.environment.Proxy", build)
    spec = EnvironmentSpec(image="swebench/huge:latest", allow_hosts=["pypi.org"])
    with pytest.raises(ProxyError):
        Environment(spec).start()
    assert seen["image"] == IMAGE


def test_a_workspace_that_will_not_start_takes_the_gateway_with_it(monkeypatch):
    """The gateway is already up by then. A leaked Docker network is invisible
    until the machine runs out of address space some hours later."""
    stopped = []

    class Gateway:
        network = "agenteval-net-test"

        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            stopped.append(True)

        def snapshot(self):
            return {}

        def environment_variables(self):
            return []

    monkeypatch.setattr("agenteval.exec.environment.Proxy", Gateway)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess([], 1, "", "no such image"),
    )
    environment = Environment(EnvironmentSpec(allow_hosts=["pypi.org"]))
    with pytest.raises(EnvironmentError_, match="could not start"):
        environment.start()
    assert stopped and environment.proxy is None


def test_the_record_says_proxy_rather_than_none():
    """`network: none` in the results for a run that reached the internet would
    be a lie in the one direction that matters."""
    spec = EnvironmentSpec(allow_hosts=["pypi.org"])
    assert Environment(spec).snapshot()["network"] == "proxy"
    assert "proxy" in env_mod.describe(spec)


# --------------------------------------------------------------------------- #
# When the gateway will not come up
# --------------------------------------------------------------------------- #
#
# Every one of these has to leave nothing behind. A half-built gateway means a
# leaked Docker network, and networks leak silently until the machine runs out
# of address space some hours later.


def _failing_at(step: str):
    """A fake docker where one subcommand fails and the rest succeed."""
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        if args[: len(step.split())] == tuple(step.split()):
            return subprocess.CompletedProcess([], 1, "", f"{step} refused")
        return _ok()

    return run, calls


@pytest.mark.parametrize(
    "step, message",
    [
        ("network create", "could not create network"),
        ("run", "could not start the gateway"),
        ("network connect", "could not attach the gateway"),
    ],
)
def test_a_gateway_that_will_not_build_says_which_step(step, message):
    proxy = Proxy(["pypi.org"], image=IMAGE)
    proxy._run, calls = _failing_at(step)
    with pytest.raises(ProxyError, match=message):
        proxy.start()
    assert proxy.container is None
    assert proxy.network is None  # nothing leaked


def test_a_gateway_that_never_listens_is_not_left_running(monkeypatch):
    """Better to fail here than to hand the workspace a proxy address that
    refuses connections, which reads like a network policy problem."""
    proxy = Proxy(["pypi.org"], image=IMAGE)
    probes = []

    def run(*args, **kwargs):
        if args[0] == "exec" and "python" in args:
            probes.append(args)
            return subprocess.CompletedProcess([], 1, "", "")
        return _ok()

    proxy._run = run
    monkeypatch.setattr(proxy, "_install", lambda: None)
    proxy.network, proxy.container = "agenteval-net-test", "deadbeef"
    with pytest.raises(ProxyError, match="did not start listening"):
        proxy._await_listening(timeout=0.5)
    assert probes  # it really did try
    assert proxy.network is None and proxy.container is None  # and cleaned up


def test_reading_the_log_of_a_gateway_that_is_gone_is_empty_not_an_error():
    """`stop()` is called on paths where the container never existed."""
    assert Proxy(["pypi.org"], image=IMAGE).log_text() == ""


def test_stopping_twice_is_harmless():
    proxy = Proxy(["pypi.org"], image=IMAGE)
    proxy._run = lambda *a, **k: _ok()
    proxy.stop()
    proxy.stop()


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #


class FakeProxy(Proxy):
    def __init__(self, entries):
        super().__init__(["pypi.org"], image=IMAGE)
        self._entries = entries

    def log_text(self):
        return "\n".join(json.dumps(e) for e in self._entries)


def test_the_startup_line_is_not_a_request():
    proxy = FakeProxy([
        {"event": "listening", "port": 8080},
        {"host": "pypi.org", "allowed": True, "bytes_down": 10},
    ])
    assert len(proxy.log()) == 1


def test_unparseable_lines_do_not_lose_the_rest():
    proxy = FakeProxy([{"host": "pypi.org", "allowed": True}])
    proxy.log_text = lambda: "not json\n" + json.dumps({"host": "a", "allowed": True})
    assert len(proxy.log()) == 1


def test_the_snapshot_counts_everything_and_stores_a_sample():
    """A pip install that retries is nine log lines saying the same thing, and
    a results file is read far more often than it is written."""
    proxy = FakeProxy([{"host": "pypi.org", "allowed": True}] * 500)
    snapshot = proxy.snapshot(keep=10)
    assert snapshot["total"] == 500
    assert len(snapshot["requests"]) == 10
    assert snapshot["truncated"] == 490


def test_refusals_are_counted_and_named():
    proxy = FakeProxy([
        {"host": "pypi.org", "allowed": True, "bytes_down": 400},
        {"host": "evil.example", "allowed": False},
        {"host": "evil.example", "allowed": False},
    ])
    snapshot = proxy.snapshot()
    assert snapshot["denied"] == 2
    assert snapshot["denied_hosts"] == ["evil.example"]
    assert snapshot["hosts"] == ["evil.example", "pypi.org"]
    assert snapshot["bytes_down"] == 400


# --------------------------------------------------------------------------- #
# The server itself, driven in this process
# --------------------------------------------------------------------------- #
#
# The proxy normally runs inside the gateway container, where nothing here can
# see it. It is a plain asyncio server with no agenteval imports, so it can
# also be started on a loopback port and driven directly — which is how the
# protocol handling gets tested on a machine with no Docker at all.


@pytest.fixture
async def upstream():
    """A pretend internet: one HTTP server on loopback, allowed by name."""

    async def serve(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        body = b"upstream says hello"
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s"
            % (len(body), body)
        )
        await writer.drain()
        writer.close()

    site = await asyncio.start_server(serve, "127.0.0.1", 0)
    yield site.sockets[0].getsockname()[1]
    site.close()


@pytest.fixture
async def proxy_server(monkeypatch):
    """The real handler, with `localhost` as the only allowed host."""
    monkeypatch.setattr(server, "ALLOW", ["localhost"])
    site = await asyncio.start_server(server.handle, "127.0.0.1", 0)
    yield site.sockets[0].getsockname()[1]
    site.close()


async def speak(port: int, request: bytes) -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(request)
    await writer.drain()
    payload = await reader.read(65536)
    writer.close()
    return payload


async def test_an_allowed_request_is_forwarded_and_the_reply_returned(
    proxy_server, upstream
):
    reply = await speak(
        proxy_server,
        b"GET http://localhost:%d/ HTTP/1.1\r\nHost: localhost\r\n\r\n" % upstream,
    )
    assert b"upstream says hello" in reply


async def test_a_refusal_explains_itself(proxy_server):
    reply = await speak(
        proxy_server,
        b"GET http://evil.example/ HTTP/1.1\r\nHost: evil.example\r\n\r\n",
    )
    assert b"403 Forbidden" in reply
    assert b"not permitted by this task's allowlist" in reply


async def test_a_refused_request_never_reaches_the_upstream(proxy_server, capsys):
    """The point of the exercise: refusing after connecting would still have
    told the far end that something here was interested in it."""
    await speak(proxy_server, b"CONNECT evil.example:443 HTTP/1.1\r\n\r\n")
    entry = (await logged(capsys))[-1]
    assert entry["allowed"] is False
    assert entry["reason"] == "not in the allowlist"


async def test_an_unreachable_upstream_is_a_bad_gateway(proxy_server):
    """Distinguishable from a refusal: one is this task's policy, the other is
    the internet having a bad day, and an agent should not confuse them."""
    reply = await speak(
        proxy_server, b"CONNECT localhost:9 HTTP/1.1\r\nHost: localhost\r\n\r\n"
    )
    assert b"502 Bad Gateway" in reply


async def test_a_connect_to_an_allowed_host_opens_a_tunnel(proxy_server, upstream):
    reader, writer = await asyncio.open_connection("127.0.0.1", proxy_server)
    writer.write(b"CONNECT localhost:%d HTTP/1.1\r\n\r\n" % upstream)
    await writer.drain()
    assert b"200 Connection established" in await reader.read(200)
    # And the tunnel then carries whatever the client wants through it.
    writer.write(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
    await writer.drain()
    assert b"upstream says hello" in await reader.read(65536)
    writer.close()


async def logged(capsys) -> list[dict]:
    """The record is written after both directions have finished pumping, which
    is a moment later than the client seeing its reply."""
    for _ in range(100):
        out = capsys.readouterr().out.strip()
        if out:
            return [json.loads(line) for line in out.splitlines()]
        await asyncio.sleep(0.02)
    raise AssertionError("nothing was logged")


async def test_volumes_are_recorded_for_a_forwarded_request(
    proxy_server, upstream, capsys
):
    await speak(
        proxy_server,
        b"GET http://localhost:%d/ HTTP/1.1\r\nHost: localhost\r\n\r\n" % upstream,
    )
    entry = (await logged(capsys))[-1]
    # The head was forwarded before the splice began, so it has to be counted
    # separately or a plain GET reports zero bytes sent.
    assert entry["bytes_up"] > 0 and entry["bytes_down"] > 0


async def test_a_malformed_request_is_rejected_not_guessed_at(proxy_server, capsys):
    """`this is not http` parses very happily as a request for `is not` by the
    `THIS` method. An audit trail with invented entries in it is worse than one
    with a gap, so the parse has to fail rather than improvise."""
    reply = await speak(proxy_server, b"this is not http\r\n\r\n")
    assert b"400 Bad Request" in reply
    entry = (await logged(capsys))[-1]
    assert entry["allowed"] is False and "not an HTTP request line" in entry["reason"]
    assert "host" not in entry  # nothing was invented


async def test_an_endless_header_block_cannot_exhaust_the_gateway(
    proxy_server, monkeypatch
):
    monkeypatch.setattr(server, "MAX_HEAD", 256)
    reply = await speak(
        proxy_server,
        b"GET http://localhost/ HTTP/1.1\r\nX: " + b"a" * 4096 + b"\r\n\r\n",
    )
    assert reply == b""


# --------------------------------------------------------------------------- #
# Containment, for real
# --------------------------------------------------------------------------- #


def _networked() -> bool:
    """Only run the egress tests where egress exists."""
    if not (env_mod.available() and env_mod.image_present(IMAGE)):
        return False
    probe = subprocess.run(
        ["docker", "run", "--rm", IMAGE, "curl", "-sS", "-m", "10",
         "-o", "/dev/null", "https://example.com/"],
        capture_output=True,
    )
    return probe.returncode == 0


needs_network = pytest.mark.skipif(
    not _networked(), reason="needs the exec image and outbound network"
)


@pytest.fixture(scope="module")
def proxied():
    """One gateway for the containment tests; starting it costs seconds."""
    with Environment(EnvironmentSpec(allow_hosts=["example.com"], timeout=30.0)) as e:
        yield e


@needs_network
def test_an_allowed_host_is_reachable(proxied):
    out = proxied.exec("curl -s -o /dev/null -w '%{http_code}' https://example.com/")
    assert out.stdout.strip() == "200"


@needs_network
def test_a_host_that_was_not_named_is_refused(proxied):
    assert not proxied.exec("curl -sS -m 15 https://pypi.org/simple/").ok


@needs_network
def test_plain_http_is_proxied_too_and_says_why(proxied):
    """Over HTTPS the client only sees a failed tunnel, because the refusal
    body is inside the CONNECT. Over plain HTTP the reason reaches the agent."""
    out = proxied.exec("curl -sS -m 15 http://pypi.org/simple/")
    assert "not permitted by this task's allowlist" in out.stdout


@needs_network
def test_ignoring_the_proxy_variables_does_not_get_you_out(proxied):
    """The enforcement has to be the topology. An allowlist that could be
    turned off by unsetting an environment variable would be documentation."""
    out = proxied.exec(
        "env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY "
        "curl -sS -m 10 https://example.com/"
    )
    assert not out.ok


@needs_network
def test_a_raw_socket_has_nowhere_to_go(proxied):
    """Not a client that can be configured — the network itself is internal."""
    out = proxied.exec(
        "python -c \"import socket;socket.create_connection(('1.1.1.1',443),timeout=5)\" 2>&1"
    )
    assert not out.ok
    assert "unreachable" in out.stdout.lower() or "unreachable" in out.stderr.lower()


@needs_network
def test_the_workspace_cannot_even_resolve_names(proxied):
    """A consequence worth having: clients hand the name to the proxy and the
    proxy resolves it, so name resolution is one more thing the sandboxed side
    does not do for itself."""
    out = proxied.exec(
        "python -c \"import socket;print(socket.gethostbyname('example.com'))\" 2>&1"
    )
    assert not out.ok


@needs_network
def test_every_request_is_on_the_record(proxied):
    proxied.exec("curl -s -o /dev/null http://example.com/")
    proxied.exec("curl -s -o /dev/null -m 10 http://blocked.example/")
    snapshot = proxied.proxy.snapshot()
    assert "example.com" in snapshot["hosts"]
    assert "blocked.example" in snapshot["denied_hosts"]
    assert any(r.get("bytes_down", 0) > 0 for r in snapshot["requests"])


@needs_network
def test_the_record_survives_the_gateway_it_came_from():
    """Read before teardown, like everything else that lives in a container."""
    environment = Environment(EnvironmentSpec(allow_hosts=["example.com"]))
    environment.start()
    environment.exec("curl -s -o /dev/null -m 15 https://example.com/")
    network = environment.proxy.network
    environment.stop()
    assert environment.snapshot()["egress"]["hosts"] == ["example.com"]
    listed = subprocess.run(
        ["docker", "network", "ls", "--filter", f"name={network}", "-q"],
        capture_output=True, text=True,
    )
    assert not listed.stdout.strip()  # and the network is cleaned up
