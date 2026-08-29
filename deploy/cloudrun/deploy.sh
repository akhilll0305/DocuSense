#!/usr/bin/env bash
# Deploy DocuSense to Google Cloud Run.
#
# Re-runnable: every step either creates something or notices it already
# exists. Each command is printed before it runs, so this doubles as the
# documentation of what was done to the project.
#
# Read deploy/DEPLOY.md before running this. It creates a *public* service and
# two secrets in whichever project gcloud is pointed at.
#
# Usage:
#   export GOOGLE_CLOUD_PROJECT=your-project-id
#   export GROQ_API_KEY=gsk_...
#   export JWT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(48))")
#   ./deploy/cloudrun/deploy.sh

set -euo pipefail

SERVICE="${SERVICE:-docusense}"
REGION="${REGION:-us-central1}"
MEMORY="${MEMORY:-1Gi}"
CPU="${CPU:-2}"

GROQ_SECRET="${GROQ_SECRET:-docusense-groq-key}"
JWT_SECRET="${JWT_SECRET:-docusense-jwt-key}"

# Run from the repository root, and hand gcloud a relative source path: an
# absolute one from a Git Bash shell ("/d/...") is not a path a Windows
# gcloud can open.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

run() {
    printf '\n$ %s\n' "$*"
    "$@"
}

# ------------------------------------------------------------------
# Preconditions — checked up front, because failing half way through
# leaves secrets created and no service.
# ------------------------------------------------------------------

command -v gcloud >/dev/null 2>&1 || {
    echo "gcloud is not installed: https://cloud.google.com/sdk/docs/install" >&2
    exit 1
}

: "${GOOGLE_CLOUD_PROJECT:?set GOOGLE_CLOUD_PROJECT to your project id}"
: "${GROQ_API_KEY:?set GROQ_API_KEY — https://console.groq.com}"
: "${JWT_SECRET_KEY:?set JWT_SECRET_KEY — python -c \"import secrets; print(secrets.token_urlsafe(48))\"}"

if ! gcloud auth print-access-token >/dev/null 2>&1; then
    echo "gcloud is not logged in. Run: gcloud auth login" >&2
    exit 1
fi

echo "Project:  ${GOOGLE_CLOUD_PROJECT}"
echo "Service:  ${SERVICE} (${REGION}, ${CPU} vCPU, ${MEMORY})"
echo "Source:   $(pwd)"

run gcloud config set project "${GOOGLE_CLOUD_PROJECT}"

# ------------------------------------------------------------------
# APIs
# ------------------------------------------------------------------

run gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    secretmanager.googleapis.com

# ------------------------------------------------------------------
# Secrets
#
# In Secret Manager rather than --set-env-vars: environment variables are
# readable by anyone with view access to the service, and the signing key is
# what stands between a visitor and every other account.
# ------------------------------------------------------------------

put_secret() {
    local name="$1" value="$2"
    if gcloud secrets describe "${name}" >/dev/null 2>&1; then
        printf '\n$ gcloud secrets versions add %s (from stdin)\n' "${name}"
        printf '%s' "${value}" | gcloud secrets versions add "${name}" --data-file=-
    else
        printf '\n$ gcloud secrets create %s (from stdin)\n' "${name}"
        printf '%s' "${value}" | gcloud secrets create "${name}" --data-file=-
    fi
}

put_secret "${GROQ_SECRET}" "${GROQ_API_KEY}"
put_secret "${JWT_SECRET}" "${JWT_SECRET_KEY}"

# The runtime service account has to be allowed to read them.
PROJECT_NUMBER="$(gcloud projects describe "${GOOGLE_CLOUD_PROJECT}" --format='value(projectNumber)')"
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

for secret in "${GROQ_SECRET}" "${JWT_SECRET}"; do
    run gcloud secrets add-iam-policy-binding "${secret}" \
        --member="serviceAccount:${RUNTIME_SA}" \
        --role="roles/secretmanager.secretAccessor" \
        --condition=None
done

# ------------------------------------------------------------------
# Deploy
#
# --max-instances 1 is not a cost control: SQLite and the on-disk Qdrant live
# inside the container, so a second instance is a second, different shelf.
# ------------------------------------------------------------------

run gcloud run deploy "${SERVICE}" \
    --source . \
    --region "${REGION}" \
    --allow-unauthenticated \
    --memory "${MEMORY}" \
    --cpu "${CPU}" \
    --max-instances 1 \
    --timeout 300 \
    --set-env-vars "LLM_PROVIDER=groq,ENVIRONMENT=prod,SEED_DEMO=true,MAX_DOCUMENTS_PER_USER=5,MAX_FILE_SIZE_MB=10,USE_IMAGE_UNDERSTANDING=false" \
    --set-secrets "GROQ_API_KEY=${GROQ_SECRET}:latest,JWT_SECRET_KEY=${JWT_SECRET}:latest"

URL="$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format='value(status.url)')"

# ------------------------------------------------------------------
# Verify — deploying is not the same as working.
# ------------------------------------------------------------------

printf '\nDeployed: %s\n' "${URL}"
printf '\n$ curl %s/api/health\n' "${URL}"
curl -fsS "${URL}/api/health" && echo

cat <<EOF

Next, check that it actually answers rather than only starting:

  open ${URL}
  sign in as demo@docusense.app / read-the-papers
  ask "How do these two papers disagree about learned signal control?"

The answer should cite both papers, by author and year. If it cites only one,
the demo shelf seeded half-way — almost always memory. Check the logs for a
kill, and raise --memory:

  gcloud run services logs read ${SERVICE} --region ${REGION} --limit 50
EOF
