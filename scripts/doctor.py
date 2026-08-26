"""
Environment diagnostic for DocuSense.

Checks every external dependency the RAG pipeline needs and reports what is
actually reachable — because the pipeline degrades silently when a backend is
down, returning "no results found" instead of an error.

Usage:
    python scripts/doctor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"
_results: list[tuple[str, str]] = []


def report(status: str, check: str, detail: str = "") -> None:
    _results.append((status, check))
    line = f"{status}  {check}"
    if detail:
        line += f"\n        {detail}"
    print(line)


def check_config():
    try:
        from docusense.config.settings import settings
    except Exception as e:
        report(FAIL, "Settings load", str(e))
        return None
    report(OK, "Settings load", f"qdrant mode = {settings.effective_qdrant_mode}")
    return settings


def check_sqlite(settings):
    import sqlite3

    db = settings.sqlite_db_path
    if not db.exists():
        report(WARN, "SQLite database", f"not created yet at {db}")
        return
    try:
        con = sqlite3.connect(db)
        docs = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        con.close()
        status = OK if chunks else WARN
        report(status, "SQLite database", f"{docs} documents, {chunks} chunks")
    except Exception as e:
        report(FAIL, "SQLite database", str(e))


def check_qdrant(settings):
    from qdrant_client import QdrantClient

    mode = settings.effective_qdrant_mode
    try:
        if mode == "server":
            client = QdrantClient(
                url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=10
            )
        elif mode == "disk":
            client = QdrantClient(path=str(settings.qdrant_path))
        else:
            client = QdrantClient(":memory:")

        names = [c.name for c in client.get_collections().collections]
        target = settings.qdrant_collection_name

        if target not in names:
            report(WARN, f"Qdrant ({mode})", f"reachable, but collection '{target}' does not exist")
            return

        count = client.get_collection(target).points_count
        status = OK if count else WARN
        detail = f"collection '{target}' has {count} points"
        if not count:
            detail += " — nothing to retrieve; re-ingest documents"
        report(status, f"Qdrant ({mode})", detail)
    except Exception as e:
        hint = ""
        if mode == "server":
            hint = "\n        Cloud cluster may be paused/expired. Unset QDRANT_URL to fall back to local disk."
        report(FAIL, f"Qdrant ({mode})", f"{type(e).__name__}: {e}{hint}")


def check_ollama(settings):
    try:
        import ollama

        client = ollama.Client(host=settings.ollama_base_url)
        models = [m.get("model", m.get("name", "")) for m in client.list().get("models", [])]
        want = settings.ollama_model
        if any(want in m for m in models):
            report(OK, "Ollama", f"'{want}' available")
        else:
            report(
                WARN,
                "Ollama",
                f"running, but '{want}' not pulled. Run: ollama pull {want}",
            )
    except Exception as e:
        report(
            FAIL,
            "Ollama",
            f"{type(e).__name__}: unreachable at {settings.ollama_base_url}"
            f"\n        Start it with: ollama serve",
        )


def check_gemini(settings):
    if not settings.gemini_api_key:
        report(WARN, "Gemini API", "no key set — query rewriting disabled (system still works)")
        return
    try:
        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(settings.gemini_model)
        model.generate_content("ping")
        report(OK, "Gemini API", f"model '{settings.gemini_model}' responding")
    except Exception as e:
        msg = str(e)
        if "not available" in msg or "404" in msg:
            hint = "Model retired — update GEMINI_MODEL in .env"
        elif "denied" in msg or "403" in msg:
            hint = "Key's project lacks access — issue a new key at aistudio.google.com"
        elif "429" in msg or "quota" in msg.lower():
            hint = "Quota exhausted — retry later or use a different key"
        else:
            hint = "Check GEMINI_API_KEY in .env"
        # Optional dependency: only query rewriting/expansion is lost. Section
        # routing and academic filters are pattern-based and unaffected.
        report(
            WARN,
            "Gemini API (optional)",
            f"{settings.gemini_model}: {msg[:100]}\n        {hint}"
            f"\n        Retrieval still works; only LLM query rewriting is disabled.",
        )


def check_embeddings(settings):
    try:
        from docusense.embeddings.embedding_generator import EmbeddingGenerator

        gen = EmbeddingGenerator()
        dim = len(gen.embed_text("test"))
        if dim != settings.embedding_dimension:
            report(
                FAIL,
                "Embedding model",
                f"produced {dim} dims but settings expect {settings.embedding_dimension}",
            )
        else:
            report(OK, "Embedding model", f"{settings.embedding_model} ({dim} dims)")
    except Exception as e:
        report(FAIL, "Embedding model", f"{type(e).__name__}: {e}")


def main() -> int:
    print("\nDocuSense environment diagnostic\n" + "=" * 60)

    settings = check_config()
    if settings is None:
        return 1

    check_sqlite(settings)
    check_embeddings(settings)
    check_qdrant(settings)
    check_ollama(settings)
    check_gemini(settings)

    failures = sum(1 for s, _ in _results if s == FAIL)
    warnings = sum(1 for s, _ in _results if s == WARN)

    print("=" * 60)
    print(f"{len(_results)} checks — {failures} failed, {warnings} warnings\n")

    if failures:
        print("The RAG pipeline will not answer questions until failures are resolved.\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
