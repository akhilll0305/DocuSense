#!/bin/sh
# Container start: seed if asked, then serve.
#
# Seeding runs on every start, not once at build, because the free hosting
# tiers this is meant for give ephemeral storage — a restart wipes the
# database, and an instance that comes back with an empty shelf is a broken
# demo. scripts/seed_demo.py is idempotent and no-ops unless SEED_DEMO is set,
# so this is a cheap check locally and a repair on a public instance.
set -e

if [ "${SEED_DEMO}" = "true" ] || [ "${SEED_DEMO}" = "1" ]; then
    echo "[entrypoint] Seeding the demo shelf..."
    # A failed seed must not stop the server: an instance with an empty shelf
    # is worth more than no instance, and the reason is in the log either way.
    python scripts/seed_demo.py || echo "[entrypoint] Seeding failed; starting anyway."
fi

# Hugging Face Spaces and most PaaS hosts inject the port to listen on.
PORT="${PORT:-8000}"
echo "[entrypoint] Starting DocuSense on port ${PORT}"

# `python -m uvicorn` rather than the `uvicorn` console script: it resolves
# through the interpreter already on PATH instead of a generated shim, which is
# one less thing to differ between the container and a developer's shell.
exec python -m uvicorn docusense.api.app:app --host 0.0.0.0 --port "${PORT}"
