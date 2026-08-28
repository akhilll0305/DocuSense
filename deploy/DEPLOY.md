# Deploying DocuSense

The target is a free, always-reachable URL that shows a working system on the
first click — a link that can sit on a CV.

Everything the deployment needs is in the repository. What is left is creating
two accounts and pasting one secret, which only you can do.

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

**Hugging Face Spaces over a general PaaS.** The free CPU tier has enough memory
for torch, the embedding model and the cross-encoder without trimming anything;
it needs no card; and it is where people go looking for ML projects. The cost is
ephemeral storage, which is why the instance seeds its own demo shelf on every
start.

**What ephemeral storage means here.** A restart clears registered accounts and
uploaded documents. For a demo that is acceptable, and the README on the Space
says so rather than letting someone discover it. If you later want persistence,
add a paid persistent disk, or point `QDRANT_URL`/`QDRANT_API_KEY` at Qdrant
Cloud and mount a volume for `data/docusense.db`.

---

## Before you start

Two accounts, both free, both needing an email address:

1. **Groq** — https://console.groq.com → create an API key.
2. **Hugging Face** — https://huggingface.co/join

Generate a signing key. Without one the app mints a random key per restart, so
every session dies whenever the container recycles:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

---

## Deploy

**1. Create the Space.** At https://huggingface.co/new-space:

- Owner: your username
- Space name: `docusense`
- License: MIT
- SDK: **Docker** → **Blank**
- Hardware: **CPU basic (free)**
- Visibility: **Public**

**2. Push the code.**

```bash
git clone https://huggingface.co/spaces/<your-username>/docusense hf-space
cd hf-space

# Everything from this repo except its git history.
git --git-dir=../LLM-COURSE-PROJECT/.git --work-tree=. checkout main -- .

# The Space is configured by YAML at the top of its README, so the Space's
# README replaces the project one. Keep the project README as a second file so
# the source stays readable from the Space's file browser.
mv README.md PROJECT_README.md
cp deploy/huggingface/README.md README.md

git add -A
git commit -m "Deploy DocuSense"
git push
```

**3. Set the secrets.** Space → **Settings** → **Variables and secrets**.

As **secrets** (encrypted, never shown again):

| Name | Value |
|---|---|
| `GROQ_API_KEY` | the key from the Groq console |
| `JWT_SECRET_KEY` | the key you generated above |

As **variables** (visible, and fine to be):

| Name | Value | Why |
|---|---|---|
| `LLM_PROVIDER` | `groq` | Hosted generation instead of Ollama |
| `ENVIRONMENT` | `prod` | Refuses to start without a signing key |
| `SEED_DEMO` | `true` | Reseeds the demo shelf after every restart |
| `MAX_DOCUMENTS_PER_USER` | `5` | A public upload endpoint is a public disk |
| `MAX_FILE_SIZE_MB` | `10` | Same reason |
| `USE_IMAGE_UNDERSTANDING` | `false` | Figure captioning needs a Gemini key |

The build takes roughly ten minutes, most of it torch and the two models being
baked into the image so the first query is not also a download.

**4. Check it.**

```bash
curl -s https://<your-username>-docusense.hf.space/api/health
```

Then open the Space, sign in as `demo@docusense.app` / `read-the-papers`, and
ask *"How do these two papers disagree about learned signal control?"* — the two
seeded papers reach opposite conclusions from the same benchmarks, so the answer
should cite both.

---

## Changing the demo shelf

The seeded papers are `data/demo/*.md`, and they are deliberately synthetic:
they are structurally realistic and they contradict each other, and shipping
someone else's PDF in a public image is a licensing question nobody needs.

To use your own, drop files into `data/demo/` and push. The seeder is
idempotent and only ingests what is not already there.

To change the demo credentials, set `DEMO_EMAIL` and `DEMO_PASSWORD` as Space
variables. The account is meant to be shared: it owns nothing but the demo
papers, and per-user isolation keeps it out of every other account's documents.

---

## Running it somewhere else

Nothing above is specific to Hugging Face beyond the Space README. The image
reads `PORT`, so any host that injects one works:

```bash
docker build -t docusense .
docker run -p 8000:8000 \
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
