# Modal.com Deployment Guide

## 🚀 Deploying DocuSense to Modal.com

Modal.com is perfect for deploying DocuSense because:
- ✅ **Free tier available** (30 free credits/month, ~20 hours compute)
- ✅ **Serverless** - pay only when used
- ✅ **GPU support** - optional acceleration
- ✅ **Simple deployment** - just Python code
- ✅ **Built-in API hosting**

---

## Prerequisites

1. **Create Modal account** (FREE)
   - Visit: https://modal.com/
   - Sign up with GitHub/Google
   - Get 30 free credits/month

2. **Install Modal SDK**
   ```bash
   pip install modal
   ```

3. **Authenticate**
   ```bash
   modal token new
   ```

---

## Deployment Architecture

```
┌─────────────────────────────────────────┐
│         Modal.com Cloud                 │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────────────────────────────┐  │
│  │   DocuSense Container            │  │
│  │                                  │  │
│  │  • Ollama + Llama 3.2           │  │
│  │  • Sentence Transformers        │  │
│  │  • FAISS Vector Store           │  │
│  │  • FastAPI Backend              │  │
│  │  • Gradio Frontend              │  │
│  └──────────────────────────────────┘  │
│                                         │
│  Resources:                             │
│  • CPU: 2-4 cores (FREE tier)          │
│  • RAM: 4-8 GB                         │
│  • GPU: Optional (T4 for speed)        │
│  • Storage: Ephemeral (use volumes)    │
└─────────────────────────────────────────┘
           ↓
    Public HTTPS URL
```

---

## Deployment Files

### 1. `modal_deploy.py` (Main deployment script)

```python
"""
Modal.com deployment for DocuSense.
100% FREE deployment using Modal's free tier.
"""

import modal

# Create Modal app
app = modal.App("docusense")

# Define container image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers",
        "faiss-cpu",
        "fastapi",
        "gradio",
        "ollama",
        "rank-bm25",
        "loguru",
        "pydantic",
        "pydantic-settings",
        "python-dotenv",
    )
    .run_commands(
        # Install Ollama
        "curl -fsSL https://ollama.ai/install.sh | sh",
        # Pull model (cached in image)
        "ollama pull llama3.2:3b",
    )
)

# Create persistent volume for documents and vector stores
volume = modal.Volume.from_name("docusense-data", create_if_missing=True)

# Create secret for environment variables
secrets = modal.Secret.from_dict({
    "ENVIRONMENT": "prod",
    "DEFAULT_LLM_PROVIDER": "ollama",
    "EMBEDDING_PROVIDER": "sentence-transformers",
})


@app.function(
    image=image,
    secrets=[secrets],
    volumes={"/data": volume},
    cpu=2,  # FREE tier
    memory=4096,  # 4 GB
    timeout=600,  # 10 minutes
)
@modal.web_endpoint(method="POST")
def query_endpoint(request: dict):
    """
    FastAPI endpoint for document queries.
    """
    from docusense.api.main import process_query
    
    query = request.get("query", "")
    result = process_query(query)
    
    return result


@app.function(
    image=image,
    secrets=[secrets],
    volumes={"/data": volume},
    cpu=2,
    memory=4096,
)
@modal.web_endpoint()
def gradio_ui():
    """
    Gradio UI endpoint.
    """
    import gradio as gr
    from docusense.ui.app import create_interface
    
    demo = create_interface()
    return demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


@app.function(
    image=image,
    secrets=[secrets],
    volumes={"/data": volume},
    cpu=2,
    memory=4096,
)
def index_documents(document_paths: list):
    """
    Index documents into vector store.
    """
    from docusense.retrieval.indexer import DocumentIndexer
    
    indexer = DocumentIndexer()
    indexer.index_documents(document_paths)
    
    # Commit volume changes
    volume.commit()
    
    return {"status": "success", "indexed": len(document_paths)}


# Local development
@app.local_entrypoint()
def main():
    """
    Deploy DocuSense to Modal.
    """
    print("🚀 Deploying DocuSense to Modal.com...")
    print(f"📝 Query API: {query_endpoint.web_url}")
    print(f"🎨 Gradio UI: {gradio_ui.web_url}")
    print("\n✅ Deployment complete!")
```

### 2. Deployment Commands

```bash
# Deploy to Modal
modal deploy modal_deploy.py

# Run locally (for testing)
modal run modal_deploy.py

# View logs
modal app logs docusense

# Check volume contents
modal volume ls docusense-data
```

---

## Cost Optimization (Stay FREE)

### Free Tier Limits
- **30 credits/month** (FREE)
- **~20 hours** of CPU compute
- **Storage**: 10 GB free

### Tips to Stay Free
1. **Use CPU-only**: GPU costs more
2. **Optimize timeout**: Set to minimum needed
3. **Cache models**: Build them into the image
4. **Use volumes wisely**: 10 GB free limit
5. **Sleep when idle**: Modal auto-sleeps (FREE)

### Cost-Saving Configuration
```python
@app.function(
    cpu=2,              # Minimum for decent performance
    memory=4096,        # 4 GB sufficient
    timeout=300,        # 5 min max
    container_idle_timeout=120,  # Sleep after 2 min
)
```

---

## Deployment Checklist

### Before Deployment
- [ ] Test locally with `modal run`
- [ ] Verify Ollama works in container
- [ ] Test with sample documents
- [ ] Check embeddings generate correctly
- [ ] Verify FAISS indexing works

### Deployment Steps
1. **Authenticate**: `modal token new`
2. **Create secrets**: Set environment variables
3. **Create volume**: For persistent data
4. **Deploy**: `modal deploy modal_deploy.py`
5. **Test endpoints**: Use provided URLs
6. **Monitor**: Check logs and usage

### Post-Deployment
- [ ] Test query endpoint
- [ ] Test Gradio UI
- [ ] Upload test documents
- [ ] Monitor credit usage
- [ ] Set up monitoring/alerts (optional)

---

## Alternative: Modal GPU (Optional)

If you have credits or want faster inference:

```python
@app.function(
    gpu="T4",  # Cheapest GPU (~$0.60/hr)
    # OR
    gpu="A10G",  # Better performance (~$1.10/hr)
)
```

**Note**: GPUs cost credits, but CPU-only is FREE and sufficient!

---

## Troubleshooting

### Issue: Ollama not starting
```python
# Add explicit Ollama startup
.run_commands(
    "curl -fsSL https://ollama.ai/install.sh | sh",
    "ollama serve &",  # Start server
    "sleep 5",         # Wait for startup
    "ollama pull llama3.2:3b",
)
```

### Issue: Out of memory
- Reduce `memory` to 2048 MB
- Use smaller model (phi3:mini)
- Reduce batch sizes

### Issue: Timeout
- Increase `timeout` value
- Optimize chunking (smaller chunks)
- Reduce top_k results

---

## Monitoring & Logs

```bash
# View real-time logs
modal app logs docusense --follow

# Check usage/credits
modal profile current

# List deployments
modal app list
```

---

## Local Development vs Modal

| Feature | Local | Modal |
|---------|-------|-------|
| **Cost** | FREE | FREE tier |
| **Setup** | Install Ollama | Zero setup |
| **Speed** | Fast | Network latency |
| **Privacy** | 100% local | Cloud hosted |
| **Scaling** | Single machine | Auto-scales |
| **Access** | localhost | Public URL |

---

## Next Steps

1. **Finish Phase 1-7** of project
2. **Test everything locally**
3. **Create `modal_deploy.py`**
4. **Deploy to Modal**
5. **Share your public URL!** 🎉

---

## Resources

- **Modal Docs**: https://modal.com/docs
- **Modal Examples**: https://modal.com/docs/examples
- **Pricing**: https://modal.com/pricing
- **Community**: https://modal.com/slack

---

**Remember**: With Modal's free tier and our free models, you can run DocuSense at **ZERO COST**! 🆓✨
