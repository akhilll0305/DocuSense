# Quick Start Guide

## ✅ Phase 0 Complete!

Your DocuSense project is now set up and ready for development.

---

## What Was Just Created

### 📁 Project Structure
- `docusense/` - Main package with modules for LLMs, retrieval, evaluation, etc.
- `data/` - Data directories for raw docs, processed chunks, vector stores
- `tests/` - Test suite
- `logs/` - Application logs
- `.env` - Your environment configuration

### ⚙️ Configuration
- All settings managed via `.env` file
- Pydantic validation for type safety
- Easy to switch between dev/test/prod environments

### 🧪 Testing
- 2/2 tests passing ✅
- Code coverage: 69%
- Run with: `pytest tests/ -v`

---

## Next Steps

### 1. Add Your API Keys

Edit `.env` and add your API keys:

```env
# Required for LLM features
OPENAI_API_KEY=sk-...your-key-here...

# Optional
ANTHROPIC_API_KEY=your-key-here
COHERE_API_KEY=your-key-here
```

### 2. Ready for Phase 1!

**Phase 1: Knowledge Ingestion & Chunking**

We'll build:
- Document loaders (PDF, DOCX, TXT)
- Text preprocessing pipeline
- Chunking strategies
- Chunk storage

Run this when ready:
```powershell
# Just say: "let's start Phase 1"
```

---

## Useful Commands

### Activate Virtual Environment
```powershell
.\venv\Scripts\activate
```

### Run Tests
```powershell
pytest tests/ -v
```

### Test Configuration Loading
```powershell
python -c "from docusense.config import settings; print(f'Project: {settings.project_name}')"
```

### Format Code (Before Committing)
```powershell
black docusense/
```

### Check Code Quality
```powershell
ruff check docusense/
```

---

## Project Resources

- **[PROJECT_PLAN.md](PROJECT_PLAN.md)** - Complete roadmap
- **[PHASE_0_SUMMARY.md](PHASE_0_SUMMARY.md)** - What we just built
- **[README.md](README.md)** - Project overview

---

## Environment Info

- **Python**: 3.13.0
- **Virtual Env**: `venv/`
- **Project Root**: `d:\My Stuff\LLM COURSE PROJECT`

---

## Questions?

Just ask! I'm here to help you build this step by step.

**Ready to continue?** → Say "start Phase 1" when ready!

---

Happy Learning! 🚀
