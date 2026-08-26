# 🆓 FREE Models & Services Guide

DocuSense is designed to be **100% FREE** - no API keys, no credits, no costs! This guide explains all the free components and alternatives.

---

## 🎯 Core Philosophy

**Everything should be FREE and open-source:**
- No monthly subscriptions
- No API costs
- No credit cards required
- Run completely offline if desired
- Privacy-first (your documents stay local)

---

## 🤖 Large Language Models (LLMs)

### Recommended: Ollama (100% FREE)

**Why Ollama:**
- Completely free, no limits
- Runs locally on your machine
- Privacy-preserving
- Fast inference
- Simple to use

**Installation:**
```powershell
# Download from https://ollama.ai/
# Or install via winget:
winget install Ollama.Ollama

# Verify installation:
ollama --version
```

**Recommended Models:**

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| **llama3.2:3b** | ~2 GB | ⚡⚡⚡ | ⭐⭐⭐ | **Default** - Best balance |
| llama3.1:8b | ~4.7 GB | ⚡⚡ | ⭐⭐⭐⭐ | Better quality, slower |
| mistral:7b | ~4.1 GB | ⚡⚡ | ⭐⭐⭐⭐ | Good reasoning |
| phi3:mini | ~2.3 GB | ⚡⚡⚡⚡ | ⭐⭐⭐ | Fastest, Microsoft |
| gemma2:2b | ~1.6 GB | ⚡⚡⚡⚡ | ⭐⭐⭐ | Very small, Google |

**Pull a model:**
```bash
# Download default model (one-time, ~2 GB)
ollama pull llama3.2:3b

# Or try others:
ollama pull mistral:7b
ollama pull phi3:mini

# List installed models
ollama list

# Test a model
ollama run llama3.2:3b "Explain RAG in one sentence"
```

**Configure in DocuSense:**
```env
DEFAULT_LLM_PROVIDER=ollama
DEFAULT_CHAT_MODEL=llama3.2:3b
DEFAULT_SMART_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434
```

### Alternative: HuggingFace (FREE with limits)

**Pros:**
- No local installation needed
- Access to thousands of models
- Free tier available

**Cons:**
- Rate limits on free tier
- Requires internet
- API key needed (but FREE)

**Setup:**
1. Create account: https://huggingface.co/
2. Get API key: https://huggingface.co/settings/tokens
3. Configure:
```env
DEFAULT_LLM_PROVIDER=huggingface
HUGGINGFACE_API_KEY=hf_your_key_here
```

**Recommended FREE models:**
- `mistralai/Mistral-7B-Instruct-v0.2`
- `meta-llama/Llama-2-7b-chat-hf`
- `google/gemma-2b-it`

---

## 🔢 Embeddings

### Recommended: Sentence Transformers (100% FREE)

**Why Sentence Transformers:**
- Completely free, unlimited
- Runs locally
- No API calls
- High quality
- Battle-tested

**Installation:**
```bash
pip install sentence-transformers
```

**Recommended Models:**

| Model | Dim | Speed | Quality | Size | Use Case |
|-------|-----|-------|---------|------|----------|
| **all-MiniLM-L6-v2** | 384 | ⚡⚡⚡⚡ | ⭐⭐⭐ | 80 MB | **Default** - Fast |
| all-mpnet-base-v2 | 768 | ⚡⚡⚡ | ⭐⭐⭐⭐ | 420 MB | Better quality |
| all-MiniLM-L12-v2 | 384 | ⚡⚡⚡ | ⭐⭐⭐⭐ | 120 MB | Balanced |
| paraphrase-MiniLM-L3-v2 | 384 | ⚡⚡⚡⚡⚡ | ⭐⭐ | 60 MB | Fastest |

**Configure:**
```env
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
EMBEDDING_DEVICE=cpu  # or 'cuda' for GPU
```

**Performance Tips:**
```python
# GPU acceleration (if available)
EMBEDDING_DEVICE=cuda  # ~10x faster

# Batch processing
EMBEDDING_BATCH_SIZE=32  # Adjust based on RAM
```

---

## 🔍 Vector Search

### Recommended: FAISS (100% FREE)

**Why FAISS:**
- Facebook's production-grade library
- Completely free
- Extremely fast
- Runs locally
- No limitations

**Installation:**
```bash
# CPU version (FREE)
pip install faiss-cpu

# GPU version (if you have NVIDIA GPU)
pip install faiss-gpu
```

**Configure:**
```env
VECTOR_STORE_TYPE=faiss
```

### Alternative: ChromaDB (100% FREE)

**Pros:**
- Built-in metadata filtering
- Automatic persistence
- Easy to use

**Cons:**
- Slightly slower than FAISS
- More memory usage

**Setup:**
```bash
pip install chromadb
```

```env
VECTOR_STORE_TYPE=chroma
```

---

## 🎯 Re-Ranking

### Recommended: Cross-Encoder (100% FREE)

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Completely free
- High quality
- Runs locally
- Trained on MS MARCO dataset

**Configure:**
```env
USE_RERANKING=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

## 🌐 Deployment

### Recommended: Modal.com (FREE tier)

**Free Tier:**
- 30 credits/month (FREE)
- ~20 hours CPU compute
- 10 GB storage
- No credit card for trial

**Alternatives (also FREE tiers):**

| Platform | Free Tier | Pros | Cons |
|----------|-----------|------|------|
| **Modal.com** | 30 credits/mo | Serverless, GPU support | Limited hours |
| HuggingFace Spaces | Unlimited CPU | Simple, integrated | Public only |
| Railway | $5 credit | Easy deploy | Limited free tier |
| Render | 750 hrs/mo | Simple | Slower |
| Fly.io | 3 VMs free | Fast, global | Complex setup |

See [MODAL_DEPLOYMENT.md](MODAL_DEPLOYMENT.md) for details.

---

## 💰 Cost Comparison

### DocuSense (FREE) vs Commercial

**Monthly costs for 1000 queries:**

| Component | DocuSense (FREE) | OpenAI | Anthropic |
|-----------|------------------|--------|-----------|
| **LLM** | $0 (Ollama) | ~$15 | ~$24 |
| **Embeddings** | $0 (Local) | ~$0.01 | N/A |
| **Vector DB** | $0 (FAISS) | ~$0 | ~$0 |
| **Hosting** | $0 (Modal free) | ~$5 | ~$5 |
| **Total** | **$0** | **~$20** | **~$29** |

**Annual savings: ~$240-$348** by using DocuSense! 💰

---

## ⚡ Performance Comparison

### Latency (1000-token response)

| Provider | Latency | Cost |
|----------|---------|------|
| **Ollama (local)** | ~2-5s | FREE |
| GPT-3.5-turbo | ~1-2s | $0.002 |
| GPT-4 | ~5-8s | $0.06 |

**Takeaway:** Local models are competitive in speed and completely FREE!

---

## 🎓 Model Quality Comparison

For RAG tasks (document Q&A):

| Model | Accuracy | Speed | Cost |
|-------|----------|-------|------|
| GPT-4 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 💰💰💰 |
| GPT-3.5 Turbo | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 💰 |
| **Llama 3.1 8B** | ⭐⭐⭐⭐ | ⭐⭐⭐ | **FREE** |
| **Llama 3.2 3B** | ⭐⭐⭐ | ⭐⭐⭐⭐ | **FREE** |
| Mistral 7B | ⭐⭐⭐⭐ | ⭐⭐⭐ | **FREE** |

**Note:** For focused RAG tasks with good retrieval, smaller models (3-8B) perform surprisingly well!

---

## 🔧 Hardware Requirements

### Minimum (FREE models work!)
- **CPU**: Any modern CPU (2+ cores)
- **RAM**: 8 GB
- **Storage**: 10 GB
- **GPU**: None required

### Recommended
- **CPU**: 4+ cores
- **RAM**: 16 GB
- **Storage**: 20 GB
- **GPU**: Optional (speeds up embeddings)

### With GPU (optional speedup)
- **NVIDIA GPU**: 4+ GB VRAM
- **Benefits**: 5-10x faster embeddings, 2-3x faster inference

---

## 📊 Quick Reference Table

| Component | FREE Options | Default | Why |
|-----------|-------------|---------|-----|
| **LLM** | Ollama, HF Inference | Ollama (llama3.2:3b) | Fast, local, unlimited |
| **Embeddings** | SentenceTransformers | all-MiniLM-L6-v2 | Small, fast, quality |
| **Vector DB** | FAISS, Chroma | FAISS | Fastest in-memory |
| **Re-ranker** | Cross-encoder | ms-marco-MiniLM-L-6-v2 | Accurate, fast |
| **Backend** | FastAPI | FastAPI | Modern, fast |
| **Frontend** | Gradio, Streamlit | Gradio | Quick UI |
| **Deploy** | Modal, HF Spaces | Modal.com | Serverless, free tier |

---

## 🚀 Getting Started with FREE Stack

1. **Install Ollama:**
   ```bash
   winget install Ollama.Ollama
   ollama pull llama3.2:3b
   ```

2. **Configure DocuSense:**
   ```bash
   # Already configured by default!
   # Just verify .env has:
   DEFAULT_LLM_PROVIDER=ollama
   EMBEDDING_PROVIDER=sentence-transformers
   ```

3. **Install dependencies:**
   ```bash
   pip install sentence-transformers faiss-cpu ollama
   ```

4. **Start building!** 🎉

---

## 💡 Pro Tips

### Speed Optimization
1. **Use smaller models**: phi3:mini instead of llama3.1:8b
2. **Reduce chunk size**: 300 tokens instead of 500
3. **Lower top_k**: Retrieve 3 instead of 5
4. **Batch embeddings**: Process multiple chunks together

### Memory Optimization
1. **Use quantized models**: 4-bit quantization in Ollama
2. **Reduce batch sizes**: Lower EMBEDDING_BATCH_SIZE
3. **Clear cache**: Periodically restart Ollama

### Quality Optimization
1. **Use larger models**: llama3.1:8b for complex questions
2. **Increase context**: More chunks in context
3. **Better embeddings**: all-mpnet-base-v2 (768 dim)
4. **Enable re-ranking**: Improves precision

---

## ❓ FAQ

**Q: Do I need any API keys?**  
A: NO! Everything works locally for free.

**Q: Can I use commercial APIs for comparison?**  
A: Yes! Just set OPENAI_API_KEY in .env. But it's optional.

**Q: Will this cost money on Modal.com?**  
A: Not on the free tier! 30 credits/month = ~20 hours FREE.

**Q: Is local Ollama fast enough?**  
A: Yes! Llama 3.2 3B runs at ~30 tokens/sec on modern CPUs.

**Q: Can I run this completely offline?**  
A: Yes! Once models are downloaded, no internet needed.

**Q: What if I want better quality?**  
A: Use llama3.1:8b or mistral:7b. Still FREE!

---

## 📚 Resources

- **Ollama**: https://ollama.ai/
- **Sentence Transformers**: https://sbert.net/
- **FAISS**: https://github.com/facebookresearch/faiss
- **Modal.com**: https://modal.com/
- **HuggingFace**: https://huggingface.co/

---

**Remember: Quality AI doesn't have to cost money!** 🆓✨

Our FREE stack is production-ready and competitive with commercial alternatives for most use cases.
