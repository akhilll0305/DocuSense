"""
DocuSense on Modal.

WHY MODAL
---------
The deployment needs about 1GB of memory — measured, not estimated: torch, the
embedding model and the cross-encoder peak at 730MB answering one question, and
627MB with reranking switched off, so the 512MB free tiers cannot run this at
all. Of the hosts that can, Modal is the one that does not ask for a card:
$30/month of compute credits on the Starter plan, and memory is a number you
choose rather than a tier you are stuck with.

Hugging Face Spaces was the original target and is not available on a free
account: `sdk: docker` and `sdk: gradio` both answer HTTP 402, "hosting Gradio
and Docker Spaces on free cpu-basic requires a PRO subscription". Google Cloud
Run works and is documented in DEPLOY.md, but wants a billing account.

WHAT THIS SERVES
----------------
The same FastAPI app as everything else — `docusense.api.app:app` — with
generation pointed at Groq, because a 3B local model needs ~4GB and ~27s per
answer on a shared CPU.

Storage is ephemeral, as it is on every free tier here: the container seeds its
own demo shelf on start, and a restart clears registered accounts and uploads.
`data/demo/` holds two synthetic papers that disagree with each other, so the
comparison path has something real to find. To make it persistent instead,
uncomment the Volume below and drop SEED_DEMO.

DEPLOYING
---------
    pip install modal
    modal setup                       # browser sign-in, no card
    modal secret create docusense \\
        GROQ_API_KEY=gsk_... \\
        JWT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(48))")
    modal deploy deploy/modal_app.py

Written against modal 1.5.5. The API has renamed these parameters before
(`concurrency_limit` -> `max_containers`, `container_idle_timeout` ->
`scaledown_window`, Mounts -> `add_local_dir`), so a failure naming an unknown
keyword means the pin below needs revisiting rather than the design.
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent

# The image is built the same way the Dockerfile builds it, and for the same
# reasons: CPU-only torch, because the default wheel drags ~2GB of CUDA into a
# container with no GPU; libmagic for file-type detection and tesseract as the
# OCR fallback; and both models baked in at build time so the first query is not
# also a download.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libmagic1", "tesseract-ocr")
    .pip_install_from_requirements(
        str(REPO / "requirements.txt"),
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .run_commands(
        "python -c \""
        "from sentence_transformers import SentenceTransformer, CrossEncoder; "
        "SentenceTransformer('all-MiniLM-L6-v2'); "
        "CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')\""
    )
    .env(
        {
            # Hosted generation. Local runs stay on Ollama; a free tier cannot.
            "LLM_PROVIDER": "groq",
            # Refuses to start without a signing key, rather than minting a
            # throwaway one and dropping every session on the next restart.
            "ENVIRONMENT": "prod",
            # Ephemeral storage, so the shelf is reseeded on every start.
            "SEED_DEMO": "true",
            # A public upload endpoint with open sign-up is otherwise a public
            # disk.
            "MAX_DOCUMENTS_PER_USER": "5",
            "MAX_FILE_SIZE_MB": "10",
            # Figure captioning needs a Gemini key this instance does not have.
            "USE_IMAGE_UNDERSTANDING": "false",
            # Measured on QASPER: rewriting costs 18% MRR. See BENCHMARKS.md.
            "QUERY_LLM_BACKEND": "off",
        }
    )
    .add_local_dir(REPO / "docusense", "/root/docusense")
    .add_local_dir(REPO / "scripts", "/root/scripts")
    .add_local_dir(REPO / "data" / "demo", "/root/data/demo")
)

app = modal.App("docusense", image=image)

# Persistence, if you want it: mount this at /root/data and drop SEED_DEMO from
# the env above. Left off by default because a Volume shared by more than one
# container is a corrupted SQLite file and a corrupted Qdrant store — the
# max_containers=1 below is what makes it safe, and that is easy to lose track
# of later.
# volume = modal.Volume.from_name("docusense-data", create_if_missing=True)


@app.function(
    # GROQ_API_KEY and JWT_SECRET_KEY. required_keys means a missing one fails
    # at deploy with a name, rather than at the first question with a 500.
    secrets=[
        modal.Secret.from_name(
            "docusense", required_keys=["GROQ_API_KEY", "JWT_SECRET_KEY"]
        )
    ],
    cpu=2,
    memory=2048,
    # State is per-container: SQLite and the on-disk Qdrant live inside it, so a
    # second container is a second, different shelf, and a visitor's next
    # request landing there would find their upload missing. This is a demo,
    # not a cluster.
    max_containers=1,
    # A cold start pays for the model load. Fifteen minutes of idle keeps one
    # container alive across someone reading an answer and asking a follow-up,
    # without paying to idle overnight.
    scaledown_window=900,
    # Seeding two papers takes ~25s before the app is ready to serve.
    startup_timeout=300,
    # Retrieval and generation take a few seconds; the *first* request also
    # loads the embedding model and the cross-encoder.
    timeout=600,
    # volumes={"/root/data": volume},
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    """
    The whole FastAPI app, unchanged.

    This body runs once per container, before any request is served, which is
    where seeding belongs: an instance that comes up with an empty shelf is a
    broken demo, and the storage here does not survive a restart.
    """
    import subprocess
    import sys

    from loguru import logger

    # Not imported through scripts.seed_demo: a failed seed must not stop the
    # server, and a subprocess cannot take the app down with it. An instance
    # with an empty shelf is worth more than no instance, and the reason is in
    # the logs either way.
    try:
        result = subprocess.run(
            [sys.executable, "scripts/seed_demo.py"],
            cwd="/root",
            capture_output=True,
            text=True,
            timeout=240,
        )
        logger.info(f"[seed] {result.stdout.strip() or result.stderr.strip()}")
    except Exception as e:
        logger.warning(f"[seed] failed, starting anyway: {e}")

    from docusense.api.app import app as fastapi_app

    return fastapi_app
