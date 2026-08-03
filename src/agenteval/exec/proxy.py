"""Monitored egress: a gateway container the workspace must go through.

Some benchmarks genuinely need the network — `pip install -r requirements.txt`,
`git clone`, a dataset fetch. Handing the workspace `--network bridge` gives it
that and everything else, unlogged, which is exactly the thing the rest of this
harness is built not to do.

So: a task names the hosts it needs, and gets a network where those are the
only ones reachable.

    environment:
      allow_hosts: [pypi.org, files.pythonhosted.org]

**The enforcement is the topology, not the environment variables.** The
workspace joins a Docker network created `--internal`, which has no route off
the host at all. The only other thing on that network is the gateway, which is
also attached to a second, ordinary network and is therefore the single path
out. `HTTP_PROXY` is set as a convenience so well-behaved clients use it
without being told; a program that ignores it does not thereby escape, it just
fails to connect. An allowlist that could be turned off by unsetting an
environment variable would be documentation, not a control.

A useful consequence: the workspace needs no working DNS. Clients hand the
*name* to the proxy and the proxy resolves it, so name resolution is one more
thing the sandboxed side cannot do for itself.

What this sees is hosts, verdicts and byte volumes. It forwards TLS rather than
intercepting it, so for an HTTPS request the path and body are not visible —
reading those would mean installing a CA the container trusts, and a harness
that can decrypt the traffic of the code it is evaluating is a larger security
surface than the one it closes. The gateway container itself has unrestricted
egress, necessarily; it is the gateway.
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

#: Resolvable from the workspace over the internal network's embedded DNS.
ALIAS = "agenteval-egress"
PORT = 8080

_SOURCE = Path(__file__).with_name("_proxy_server.py")


class ProxyError(Exception):
    pass


class Proxy:
    """The gateway container and the internal network it fronts."""

    def __init__(
        self,
        allow_hosts: list[str],
        image: str,
        docker: str = "docker",
        lifetime: float = 1800.0,
    ) -> None:
        if not allow_hosts:
            raise ProxyError("a proxied environment must allow at least one host")
        self.allow_hosts = [h.strip().lower() for h in allow_hosts if h.strip()]
        self.image = image
        self.docker = docker
        self.lifetime = lifetime
        self.network: str | None = None
        self.container: str | None = None
        self._name = f"agenteval-egress-{uuid.uuid4().hex[:12]}"

    # -- lifecycle ---------------------------------------------------------- #

    def _run(self, *args: str, **kwargs: Any) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.docker, *args], capture_output=True, text=True, **kwargs
        )

    def start(self) -> None:
        network = f"agenteval-net-{uuid.uuid4().hex[:12]}"
        # --internal is the control. Containers on this network have no route
        # off the host, so the gateway is not a policy the workspace is asked
        # to respect — it is the only wire there is.
        created = self._run("network", "create", "--internal", network)
        if created.returncode != 0:
            raise ProxyError(f"could not create network: {created.stderr.strip()[:400]}")
        self.network = network

        started = self._run(
            "run", "--detach", "--name", self._name,
            # The gateway needs real egress; that is its whole job. Everything
            # else about it is still locked down, because it is also the one
            # container on this network that can reach the internet.
            "--memory", "256m", "--cpus", "1.0", "--pids-limit", "128",
            "--security-opt", "no-new-privileges", "--cap-drop", "ALL",
            "--read-only", "--tmpfs", "/tmp:rw,exec,mode=1777,size=32m",
            "--env", "AGENTEVAL_ALLOW=" + ",".join(self.allow_hosts),
            "--env", f"AGENTEVAL_PROXY_PORT={PORT}",
            self.image, "sleep", str(int(self.lifetime)),
        )
        if started.returncode != 0:
            self.stop()
            raise ProxyError(f"could not start the gateway: {started.stderr.strip()[:400]}")
        self.container = started.stdout.strip()

        attached = self._run(
            "network", "connect", "--alias", ALIAS, network, self.container
        )
        if attached.returncode != 0:
            self.stop()
            raise ProxyError(f"could not attach the gateway: {attached.stderr.strip()[:400]}")

        self._install()
        self._await_listening()

    def _install(self) -> None:
        written = subprocess.run(
            [self.docker, "exec", "-i", self.container, "sh", "-c",
             "cat > /tmp/proxy.py"],
            input=_SOURCE.read_text(), capture_output=True, text=True, timeout=30,
        )
        if written.returncode != 0:
            self.stop()
            raise ProxyError(f"could not install the proxy: {written.stderr.strip()[:400]}")
        # Detached, with the log on a file rather than the container's stdout:
        # `docker logs` would also carry the output of anything else that ever
        # runs in here, and the egress record should be exactly one thing.
        self._run("exec", "--detach", self.container, "sh", "-c",
                  "python /tmp/proxy.py >> /tmp/egress.jsonl 2>&1")

    def _await_listening(self, timeout: float = 15.0) -> None:
        """Block until the port answers.

        Without this the workspace can start and make its first request into a
        socket nobody is listening on yet, which surfaces as an intermittent
        connection refused that looks like a network policy problem.
        """
        deadline = time.time() + timeout
        probe = (
            "import socket,sys;"
            f"s=socket.create_connection(('127.0.0.1',{PORT}),timeout=1);s.close()"
        )
        while time.time() < deadline:
            if self._run("exec", self.container, "python", "-c", probe).returncode == 0:
                return
            time.sleep(0.2)
        log = self.log_text()
        self.stop()
        raise ProxyError(
            f"the egress gateway did not start listening within {timeout:g}s"
            + (f": {log[-400:]}" if log else "")
        )

    def stop(self) -> None:
        if self.container:
            self._run("rm", "--force", self.container)
            self.container = None
        if self.network:
            self._run("network", "rm", self.network)
            self.network = None

    # -- what the workspace is given ---------------------------------------- #

    @property
    def endpoint(self) -> str:
        return f"http://{ALIAS}:{PORT}"

    def environment_variables(self) -> list[str]:
        """Both cases, because the ecosystem never agreed on one.

        curl and git read the lowercase names, most Python libraries read the
        uppercase ones, and a client that reads neither is not thereby exempt —
        it is on an internal network and simply has nowhere to go.
        """
        return [
            f"HTTP_PROXY={self.endpoint}", f"http_proxy={self.endpoint}",
            f"HTTPS_PROXY={self.endpoint}", f"https_proxy={self.endpoint}",
            "NO_PROXY=localhost,127.0.0.1", "no_proxy=localhost,127.0.0.1",
        ]

    # -- the record --------------------------------------------------------- #

    def log_text(self) -> str:
        if not self.container:
            return ""
        return self._run("exec", self.container, "cat", "/tmp/egress.jsonl").stdout

    def log(self) -> list[dict[str, Any]]:
        """Every request, allowed or refused.

        Read before teardown, like everything else that lives in a container.
        """
        entries = []
        for line in self.log_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("event") != "listening":
                entries.append(entry)
        return entries

    def snapshot(self, keep: int = 200) -> dict[str, Any]:
        """The egress record, for the results file.

        Counts are over everything; the individual entries are capped. A `pip
        install` that retries is nine log lines saying the same thing, and a
        results file is read far more often than it is written.
        """
        requests = self.log()
        return {
            "allow_hosts": self.allow_hosts,
            "requests": requests[:keep],
            "truncated": max(0, len(requests) - keep),
            "hosts": sorted({r["host"] for r in requests if r.get("host")}),
            "total": len(requests),
            "denied": sum(1 for r in requests if not r.get("allowed")),
            "denied_hosts": sorted(
                {r["host"] for r in requests if r.get("host") and not r.get("allowed")}
            ),
            "bytes_down": sum(r.get("bytes_down", 0) for r in requests),
        }
