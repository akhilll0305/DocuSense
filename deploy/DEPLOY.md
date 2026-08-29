# Deploying DocuSense

The target is an always-reachable URL that shows a working system on the first
click — a link that can sit on a CV.

Everything the deployment needs is in the repository. What is left is one
browser sign-in and two secrets, which only you can create.

The target is **Modal**: it is the only host with enough memory for this that
does not ask for a credit card. Google Cloud Run is documented
[below](#google-cloud-run) as the alternative for anyone willing to attach one.

---

## Why this shape

**Generation moves off Ollama.** Locally, `llama3.2:3b` runs on your machine:
no per-query cost, no rate limit, and documents never leave the laptop. That is
the right default and it stays the default. It is the wrong choice for a public
instance — the model wants about 4GB of RAM and answers in roughly 27 seconds on
a shared CPU, which is above every free tier and slower than anyone clicking a
link will wait. So `LLM_PROVIDER=groq` points generation at a hosted model on
Groq's free tier, which answers in one to two seconds. Nothing else changes:
retrieval, reranking, citation checking and the UI are identical.

**The host has to give it about 1GB.** Measured, not estimated — the same image
this document deploys, run under a hard container memory limit:

| | RSS |
|---|---|
| Python + torch imported | 195 MB |
| + DocuSense imported, components not yet built | 248 MB |
| + one answered query, reranking **off** | 627 MB |
| + one answered query, reranking on *(the default)* | **730 MB peak** |

Reranking is the single largest measured accuracy gain in the system (+25.8%
MRR, p = 0.0044), so turning it off to fit a smaller box trades away the thing
worth demonstrating — and as the table shows, it does not even work: torch and
the embedding model alone are already over 512 MB.

**512MB free tiers do not fail loudly.** Run under `-m 512m`, the container
does not crash. The seeding process is OOM-killed part-way through, PID 1
survives, `/api/health` returns `200`, and the demo login works — with half a
shelf. Asked how the two demo papers disagree, that instance answered:

> there are no disagreements between the two sources because they are not two
> different papers; rather, they are two sections of the same paper

Fluent, cited, and wrong, because the second paper never finished ingesting.
A host that is too small produces a broken demo that looks healthy, which is
worse than one that refuses to start.

---

## What was ruled out, and how

**Hugging Face Spaces — not available on this account.** Probed against the
account's own token rather than assumed:

```
POST https://huggingface.co/api/repos/create  {"type":"space","sdk":"docker"}
→ HTTP 402  "Static Spaces are free for everyone, but hosting Gradio and
             Docker Spaces on free cpu-basic requires a PRO subscription."
```

`sdk: gradio` returns the identical 402, so the "skip Docker, run uvicorn
inside a Gradio Space" route is closed too. Only `sdk: static` creates — and a
static Space serves files, not a Python process. Hugging Face is viable again
the day the account has PRO ($9/month), and nothing else in this repository
would need to change: `deploy/huggingface/README.md` is the Space card, and the
`Dockerfile` is what a Docker Space builds.

**Render and Koyeb — 512 MB on the free tier**, which the table above rules
out.

**Fly.io, Railway — no free allowance** that fits a 1GB always-on machine.

**Google Cloud Run — works, but wants a billing account.** It takes the image
in this repository unchanged and lets memory be a number rather than a tier.
It is the right answer the moment a card is acceptable; the steps are
[below](#google-cloud-run).

---

## Modal

Modal's Starter plan gives $30/month of compute credits and no monthly fee, and
memory is a parameter rather than a tier — which is what this needs, given the
730MB measured above. `deploy/modal_app.py` builds the same environment the
`Dockerfile` builds, for the same reasons, and serves the same
`docusense.api.app:app`.

Two things to know before starting:

- **It scales to zero, so the first visitor after an idle period waits.**
  Measured on this image locally: 25s from container start to a healthy
  `/api/health` — the demo shelf is seeded in that window — then 20s for the
  first query, which is where the embedding model and the cross-encoder load.
  Every query after that is about a second. `scaledown_window=900` keeps a
  container alive for fifteen minutes of idle, so a visitor reading an answer
  and asking a follow-up does not pay it twice.
- **Storage is ephemeral.** A restart clears registered accounts and uploads,
  and the demo shelf reseeds itself. To make it persistent, uncomment the
  `Volume` in `deploy/modal_app.py` and drop `SEED_DEMO` — but leave
  `max_containers=1` where it is, because a Volume shared by two containers is
  a corrupted SQLite file and a corrupted Qdrant store.

### Deploy

**1. Accounts.** [Groq](https://console.groq.com) for the API key, and
[Modal](https://modal.com) — sign-in is through GitHub or Google.

**2. Install and authenticate.** `modal setup` opens a browser and writes a
token; it is the only interactive step.

```bash
pip install modal
modal setup
```

**3. Store the two secrets.** They are read by name from inside the container,
so they never appear in the source or in a shell history file:

```bash
modal secret create docusense \
    GROQ_API_KEY=gsk_... \
    JWT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(48))")
```

Without `JWT_SECRET_KEY` the app mints a random key per restart, so every
session dies whenever the container recycles. `ENVIRONMENT=prod` is set in the
image, which makes the app refuse to start rather than do that quietly — and
`required_keys` on the Secret turns a missing one into a named failure at
deploy time instead of a 500 on the first question.

**4. Deploy.** The first build takes roughly ten minutes, most of it torch and
the two models being baked in so the first query is not also a download.
Rebuilds reuse the layers.

```bash
modal deploy deploy/modal_app.py
```

It prints the URL, of the form `https://<workspace>--docusense-web.modal.run`.

**5. Check it answers, not just that it started.**

```bash
curl -s https://<workspace>--docusense-web.modal.run/api/health
```

Then open it, sign in as `demo@docusense.app` / `read-the-papers`, and ask
*"How do these two papers disagree about learned signal control?"* — the two
seeded papers reach opposite conclusions from the same benchmarks, so the
answer should cite both, by author and year. If it cites only one, the shelf
seeded half-way: `modal app logs docusense` will show why, and memory is the
usual reason.

### What is unverified here

`deploy/modal_app.py` has not been deployed. Building the app graph is checked
— importing the module constructs the `App`, the `Image` and the function
against modal 1.5.5, which is what catches a renamed parameter, and this API
has renamed several (`concurrency_limit` → `max_containers`,
`container_idle_timeout` → `scaledown_window`, Mounts → `add_local_dir`). What
has *not* been checked is the deploy itself, because that needs an account only
the owner can create. The image contents are the part with prior evidence: the
same package list, the same two models and the same seeded shelf were built and
run under a 1GiB container limit, and answered the demo question citing both
papers in 1.7s.

---

## Google Cloud Run

The alternative, for anyone willing to attach a card. Cloud Run takes the image
in this repository unchanged: it reads `PORT`, runs as a non-root user, and has
a healthcheck. Memory is a number you choose rather than a tier you are stuck
with.

Two things to know before starting:

- **It needs a billing account with a card**, even though the usage here is
  meant to stay inside the always-free allowance. Google publishes the current
  allowance on [the Cloud Run pricing page](https://cloud.google.com/run/pricing)
  — read it there; the numbers move, and they are not reproduced here because a
  stale figure in a repository is worse than no figure.
- **It scales to zero, so the first visitor after an idle period waits.**
  Measured on this image locally: 25s from container start to a healthy
  `/api/health` (the demo shelf is seeded during that window), then 20s for the
  first query, which is where the embedding model and the cross-encoder are
  loaded. Every query after that is about 1s. On Cloud Run add the image pull
  to the front of that. Keeping an instance warm is possible
  (`--min-instances 1`) and is charged for idle time, so it is a decision about
  money rather than about configuration.

### Before you start

1. **Groq** — https://console.groq.com → create an API key.
2. **Google Cloud** — a project with billing enabled, and the
   [gcloud CLI](https://cloud.google.com/sdk/docs/install) installed.
3. A signing key. Without one the app mints a random key per restart, so every
   session dies whenever the container recycles:

   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(48))"
   ```

### Deploy

`deploy/cloudrun/deploy.sh` runs the sequence below end to end; it is written
to be re-runnable, and it prints each command before running it. Read it before
running it — it creates secrets and a public service in whichever project
`gcloud` is pointed at.

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
export GROQ_API_KEY=gsk_...
export JWT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(48))")

./deploy/cloudrun/deploy.sh
```

What it does, if you would rather run the steps by hand:

**1. Enable the APIs.**

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com secretmanager.googleapis.com
```

**2. Store the two secrets** in Secret Manager rather than passing them as
environment variables, which are readable by anyone with view access to the
service:

```bash
printf '%s' "$GROQ_API_KEY"    | gcloud secrets create docusense-groq-key  --data-file=-
printf '%s' "$JWT_SECRET_KEY"  | gcloud secrets create docusense-jwt-key   --data-file=-
```

**3. Deploy from source.** Cloud Build builds the `Dockerfile` and pushes the
image; the build takes roughly ten minutes, most of it torch and the two models
being baked in so the first query is not also a download.

```bash
gcloud run deploy docusense \
    --source . \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --max-instances 1 \
    --timeout 300 \
    --set-env-vars LLM_PROVIDER=groq,ENVIRONMENT=prod,SEED_DEMO=true,MAX_DOCUMENTS_PER_USER=5,MAX_FILE_SIZE_MB=10,USE_IMAGE_UNDERSTANDING=false \
    --set-secrets GROQ_API_KEY=docusense-groq-key:latest,JWT_SECRET_KEY=docusense-jwt-key:latest
```

Why those numbers:

| Flag | Why |
|---|---|
| `--memory 2Gi` | 730 MB is the measured peak for one query, and on Cloud Run the container filesystem is in memory — every uploaded and converted document is charged against this limit as well. 1Gi runs the demo; 2Gi survives visitors uploading their own papers. |
| `--cpu 2` | Reranking is CPU-bound. One vCPU roughly doubles the ~1.8s rerank. |
| `--max-instances 1` | State is per-instance: SQLite and the on-disk Qdrant live in the container. Two instances means two different shelves, and a visitor's second request landing on the other one. This is a demo, not a cluster. |
| `--timeout 300` | Retrieval plus generation is a few seconds, but the *first* request also loads both models. |
| `ENVIRONMENT=prod` | Refuses to start without a signing key, rather than minting a throwaway one. |
| `SEED_DEMO=true` | The filesystem is ephemeral, so the shelf is reseeded on every start. |
| `MAX_DOCUMENTS_PER_USER`, `MAX_FILE_SIZE_MB` | A public upload endpoint with open sign-up is otherwise a public disk. |

**4. Check it.**

```bash
URL=$(gcloud run services describe docusense --region us-central1 --format 'value(status.url)')
curl -s "$URL/api/health"
```

Then open the URL, sign in as `demo@docusense.app` / `read-the-papers`, and ask
*"How do these two papers disagree about learned signal control?"* — the two
seeded papers reach opposite conclusions from the same benchmarks, so the answer
should cite both, by author and year. If it cites only one, the shelf seeded
half-way: check the logs for an OOM kill and raise `--memory`.

---

## Changing the demo shelf

The seeded papers are `data/demo/*.md`, and they are deliberately synthetic:
they are structurally realistic and they contradict each other, and shipping
someone else's PDF in a public image is a licensing question nobody needs.

To use your own, drop files into `data/demo/` and redeploy. The seeder is
idempotent and only ingests what is not already there.

To change the demo credentials, set `DEMO_EMAIL` and `DEMO_PASSWORD` as
environment variables on the service. The account is meant to be shared: it owns
nothing but the demo papers, and per-user isolation keeps it out of every other
account's documents.

---

## Running it somewhere else

Nothing above is specific to Cloud Run. The image reads `PORT`, so any host that
injects one works — the local equivalent of the deployment, which is how the
numbers in this document were measured:

```bash
docker build -t docusense .
docker run -p 8000:8080 -m 1g \
  -e PORT=8080 \
  -e LLM_PROVIDER=groq \
  -e GROQ_API_KEY=... \
  -e JWT_SECRET_KEY=... \
  -e SEED_DEMO=true \
  docusense
```

For a host with persistent disk, mount a volume at `/app/data` and drop
`SEED_DEMO`; the shelf then survives restarts on its own.

To keep generation local instead, leave `LLM_PROVIDER` unset and give the
container a reachable Ollama — `docker compose up` does that, and needs no API
key at all.

---

## If a deployed instance stops answering

The most likely cause is not the deployment. Hosted model ids are retired
without notice — `llama-3.3-70b-versatile` was this project's default until Groq
withdrew it, and the symptom was a valid API key, a `/models` endpoint returning
`200`, and every single answer failing with a bare `404`.

The system now says so in as many words. On a running instance:

```bash
python scripts/doctor.py       # [FAIL] Groq — GROQ_MODEL '...' is not available to this key
```

and an answer that fails for this reason arrives in the UI naming the model, the
cause, and every model the key *can* use. Set `GROQ_MODEL` to one of them and
redeploy.
