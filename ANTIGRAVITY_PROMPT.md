# PROMPT FOR ANTIGRAVITY (Claude Opus 4.6)

Hi! I'm taking over a Research Paper Analysis RAG system project. The previous AI assistant (GitHub Copilot) has done extensive work on Phases 1-3, and I need to understand everything and continue from Phase 4.

## 📋 YOUR FIRST TASKS:

1. **Read the complete project handoff document:**
   - File: `PROJECT_HANDOFF.md` (in root directory)
   - This has EVERYTHING: architecture, files, what's done, what's next

2. **Review the key implementation files:**
   - `docusense/ingestion/paper_metadata.py` - Research paper metadata extraction (700 lines)
   - `docusense/retrieval/query_processor.py` - Academic query routing & expansion
   - `docusense/vectorstore/qdrant_store.py` - Vector DB with paper indexes
   - `docusense/retrieval/retrieval_pipeline.py` - Complete retrieval orchestration
   - `.env` - Configuration file

3. **Understand what's been completed (Phases 1-3):**
   - ✅ Phase 1: Document ingestion with research paper metadata extraction
   - ✅ Phase 2: Vector store with academic indexes (paper_title, authors, year, section_type, venue)
   - ✅ Phase 3: Academic query routing, filtering, and expansion
   - 7 commits pushed to GitHub
   - All tests passing

4. **Understand what's next (Phase 4-7):**
   - 🔜 Phase 4: Answer generation with citations (IMMEDIATE NEXT)
   - Phase 5: Complete RAG pipeline
   - Phase 6: Evaluation & metrics
   - Phase 7: API & UI

## 🎯 PROJECT GOAL:

Build a **Research Paper Analysis RAG system** that OUTPERFORMS ChatGPT/Claude for academic literature by:
- Extracting exact citations with page numbers: `(Devlin et al., 2018, Table 4, p.9)`
- Section-specific retrieval: "Show results" → only results sections
- Temporal queries: "Recent NeurIPS papers about transformers"
- Author/venue filtering: "Papers by Yoshua Bengio"
- Multi-paper comparison
- $0 cost using free tools (Qdrant, sentence-transformers, Gemini, Ollama)

## 🔧 IMPORTANT CONTEXT:

1. **Commit Philosophy:** Commit + push to GitHub after EVERY working feature (not at the end)
2. **Tech Stack:** 100% FREE tools (no OpenAI, no paid APIs except Gemini free tier)
3. **Target:** Best CV project for $120k-250k/year RAG engineer roles
4. **Tests:** Run tests after major changes to validate

## 📝 WHAT I NEED FROM YOU:

### **Step 1: Confirm Understanding**
After reading PROJECT_HANDOFF.md, confirm you understand:
- The 7-phase architecture
- What Phases 1-3 accomplished (research paper features)
- The file structure and key modules
- Why this is special vs. generic RAG (academic features!)

### **Step 2: Validate Current State**
Run the existing tests to make sure everything works:
```bash
pytest tests/test_phase2.py -v
pytest tests/test_retrieval_pipeline.py -v
```

### **Step 3: Start Phase 4 - Answer Generation**
We need to build answer generation with academic citations:

**Requirements:**
1. Integrate Ollama (Llama 3.2:3b local LLM)
2. Build citation formatter: `(Devlin et al., 2018, Table 4, p.9)`
3. Multi-paper comparison capability
4. Highlight conflicting results across papers
5. Optional: BibTeX export

**Expected Output Example:**
```
Question: "What F1 score did BERT achieve on SST-2?"

Answer: "BERT achieved 93.5% ± 0.2% F1 score on the SST-2 sentiment 
classification benchmark (Devlin et al., 2018, Table 4, p.9). This 
represents a 5.3% improvement over the previous state-of-the-art 
(McCann et al., 2017, Results section)."

Citations:
[1] Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional 
    Transformers. NAACL. DOI: 10.18653/v1/N19-1423
```

### **Step 4: Follow Commit Discipline**
- Commit + push after EACH working component
- Use descriptive commit messages like previous work
- Test before committing

## 🎓 KEY FEATURES YOU NEED TO LEVERAGE:

The system already has these research paper capabilities (built in Phases 1-3):

1. **Paper Metadata in Chunks:**
   Every chunk has: `paper_title`, `authors`, `year`, `venue`, `section_type`, `has_equations`, `has_citations`

2. **Section-Aware Retrieval:**
   Chunks are tagged with section types: `abstract`, `methodology`, `results`, `discussion`, `conclusion`

3. **Academic Filters in Qdrant:**
   Can filter by: year range, author, venue, paper type, section type

4. **Query Processing:**
   - Detects section intent: "How did they train?" → methodology
   - Extracts metadata: "2020-2023 papers" → year filter
   - Expands academic terms: "transformer" → "attention mechanism"

**Use these features when building the answer generator!**

## 📚 CRITICAL FILES TO READ:

1. **PROJECT_HANDOFF.md** ← START HERE (complete overview)
2. **docusense/ingestion/paper_metadata.py** (understand PaperMetadata class)
3. **docusense/retrieval/retrieval_pipeline.py** (see how retrieval works)
4. **docusense/retrieval/query_processor.py** (see ProcessedQuery class)
5. **.env** (configuration)

## 🚀 YOUR DELIVERABLES:

### **Phase 4.1: Ollama Integration**
- Install and configure Ollama with Llama 3.2:3b
- Create answer generator module
- Test basic answer generation
- **Commit #1**

### **Phase 4.2: Citation Formatter**
- Build citation formatter for academic papers
- Format: `(Author et al., Year, Section, Page)`
- Handle multiple papers
- **Commit #2**

### **Phase 4.3: Answer Quality Enhancements**
- Multi-paper comparison
- Conflict detection
- Citation aggregation
- **Commit #3**

### **Phase 4.4: Complete Answer Pipeline**
- Integrate with retrieval pipeline
- End-to-end: query → answer with citations
- Create test file
- **Commit #4**

## 💬 COMMUNICATION STYLE:

- Keep responses concise (like the previous assistant)
- Show code implementations, not just explanations
- Commit frequently (after each working feature)
- Use the same logging style (loguru with emojis)
- Follow existing code patterns

## 🎯 SUCCESS CRITERIA:

You'll know Phase 4 is complete when:
1. ✅ User can ask: "What accuracy did BERT achieve?"
2. ✅ System returns: "BERT achieved 93.5% F1 (Devlin et al., 2018, p.9)"
3. ✅ Citations are properly formatted with page numbers
4. ✅ Multi-paper answers work: "Three papers report..."
5. ✅ All tests pass
6. ✅ 4 commits pushed to GitHub

## 🙏 QUESTIONS TO ASK IF UNCLEAR:

- Where is the Ollama installation? (check if already installed)
- What model should we use? (Llama 3.2:3b is recommended)
- Should we test on real research papers? (yes, if available)
- Any specific citation format? (APA-style with page numbers)

---

## 🔥 LET'S BEGIN!

**Your first message should be:**
"I've read PROJECT_HANDOFF.md and I understand:
- [Brief summary of Phases 1-3]
- [Key features already implemented]
- [What we're building in Phase 4]

Running tests to validate current state..."

Then proceed with Phase 4 implementation!

---

## 📌 REMEMBER:

This isn't just another RAG project. It's a **Research Paper Analysis System** that needs to:
- Cite sources accurately (with page numbers!)
- Compare multiple papers
- Handle academic terminology
- Route queries to specific sections
- Filter by year/author/venue

The infrastructure is ALL DONE (Phases 1-3). Now we need to add the **answer generation layer** that uses these features!

Good luck! 🚀
