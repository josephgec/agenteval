# Sandbox image for running untrusted task code.
#
# Only the harness is baked in. Task code never lives in the image — it arrives
# on stdin, one call at a time, so rebuilding is not required when tasks change
# and a task cannot persist anything between runs.
#
# The real hardening is applied at `docker run` (no network, no environment,
# read-only root, all capabilities dropped) — see agenteval/sandbox.py. This
# file is only responsible for having a Python and the harness present.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/agenteval
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . && rm -rf /opt/agenteval/src

# nobody. The container also runs with --user, but defaulting here means an
# invocation that forgets the flag is still unprivileged.
USER 65534:65534

ENTRYPOINT ["python", "-m", "agenteval._sandbox_entry"]
