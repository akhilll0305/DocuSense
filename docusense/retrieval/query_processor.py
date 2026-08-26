"""
Query Processor - Query rewriting, expansion, and enhancement.

This is Step 1 of Phase 3: Query Processing & Retrieval.

PURPOSE:
--------
Transform user queries for better retrieval:
1. Query rewriting (clarify ambiguous queries)
2. Query expansion (add synonyms, related terms)
3. Multi-query generation (multiple search angles)
4. Intent classification (factual, procedural, comparative)

WHY QUERY PROCESSING?
----------------------
Raw user queries are often:
- Too short: "RAG" instead of "Retrieval Augmented Generation"
- Ambiguous: "What is it?" (needs context)
- Missing synonyms: "ML" doesn't match "machine learning"
- Single-angle: Better to search from multiple perspectives

QUERY REWRITING STRATEGIES:
----------------------------
1. **Term Expansion**: "ML" → "machine learning algorithms"
2. **Synonym Addition**: "quick" → "fast, rapid, speedy"
3. **Context Injection**: Add conversation history
4. **Multi-Query**: One question → Multiple search queries
5. **Question Decomposition**: Complex → Multiple simple queries

EXAMPLES:
---------
Input: "How does RAG work?"
Outputs:
- "How does Retrieval Augmented Generation work?"
- "Explain the RAG architecture and components"
- "What are the steps in a RAG system?"

Input: "ML performance metrics"
Outputs:
- "Machine learning model performance metrics and evaluation"
- "How to measure ML model accuracy, precision, recall"
- "Common metrics for evaluating machine learning models"

FREE PROVIDER:
--------------
Uses Gemini 2.0 Flash (FREE tier):
- 15 requests/minute
- 1 million tokens/day
- Fast response time
- Good query understanding
"""

from typing import List, Dict, Optional, Literal, Any
from dataclasses import dataclass, field
import re
from loguru import logger

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed. Query rewriting disabled.")

from docusense.config.settings import settings


@dataclass
class QueryIntent:
    """Classification of user query intent."""
    intent_type: Literal["factual", "procedural", "comparative", "analytical", "conversational"]
    confidence: float  # 0-1
    description: str
    suggested_strategy: str  # Which retrieval strategy to use


@dataclass
class ProcessedQuery:
    """Result of query processing."""
    original_query: str
    rewritten_query: str
    expanded_queries: List[str] = field(default_factory=list)
    intent: Optional[QueryIntent] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_all_queries(self) -> List[str]:
        """Get all query variations for multi-query search."""
        queries = [self.rewritten_query]
        queries.extend(self.expanded_queries)
        return list(set(queries))  # Remove duplicates


class QueryProcessor:
    """
    Process and enhance user queries for better retrieval.
    
    Features:
    - Query rewriting (clarify and expand)
    - Multi-query generation (search from multiple angles)
    - Intent classification
    - Term expansion
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        enable_rewriting: bool = True,
        enable_expansion: bool = True,
        enable_intent_classification: bool = True
    ):
        """
        Initialize QueryProcessor.

        Args:
            api_key: Gemini API key (uses settings if None)
            model: Gemini model to use (uses settings if None)
            enable_rewriting: Enable query rewriting
            enable_expansion: Enable multi-query expansion
            enable_intent_classification: Enable intent detection
        """
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model or settings.gemini_model
        # Tripped when Gemini is unreachable or unauthorized, so a dead API
        # does not produce three identical warnings on every single query.
        self._gemini_disabled_reason: Optional[str] = None
        self.enable_rewriting = enable_rewriting and settings.enable_query_rewriting
        self.enable_expansion = enable_expansion
        self.enable_intent_classification = enable_intent_classification and settings.enable_intent_classification
        
        # Initialize Gemini if available and API key provided
        self.gemini_model = None
        if GEMINI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.gemini_model = genai.GenerativeModel(self.model_name)
                logger.info(f"QueryProcessor initialized with Gemini {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
                logger.info("Query processing will use basic expansion only")
        else:
            logger.info("QueryProcessor initialized (basic mode - no Gemini)")
    
    # Errors that will not resolve by retrying the next query.
    _PERMANENT_GEMINI_ERRORS = (
        "no longer available", "not found", "404",
        "denied access", "permission", "403",
        "api key", "unauthorized", "401",
    )

    def _handle_gemini_error(self, stage: str, error: Exception) -> None:
        """
        Log a Gemini failure and trip the circuit breaker for permanent errors.

        Transient failures (timeouts, rate limits) stay retryable; a retired
        model or a revoked key would otherwise log the same warning on every
        query for the life of the process.
        """
        msg = str(error)
        if any(tok in msg.lower() for tok in self._PERMANENT_GEMINI_ERRORS):
            self._gemini_disabled_reason = msg
            logger.warning(
                f"{stage} failed: {msg}\n"
                f"Disabling Gemini query enhancement for this session. "
                f"Section routing and academic filters are pattern-based and still active. "
                f"Update GEMINI_MODEL/GEMINI_API_KEY in .env to re-enable."
            )
        else:
            logger.warning(f"{stage} failed: {msg}")

    def process(
        self,
        query: str,
        context: Optional[str] = None,
        num_expansions: int = 2
    ) -> ProcessedQuery:
        """
        Process a query through rewriting, expansion, and intent classification.
        
        Args:
            query: User's original query
            context: Optional conversation context
            num_expansions: Number of query variations to generate
            
        Returns:
            ProcessedQuery with all enhancements
        """
        logger.info(f"Processing query: '{query}'")
        
        # Start with original query
        rewritten = query
        expanded = []
        intent = None
        
        gemini_available = self.gemini_model and not self._gemini_disabled_reason

        # 1. Query Rewriting (if Gemini available)
        if self.enable_rewriting and gemini_available:
            try:
                rewritten = self._rewrite_query(query, context)
                logger.info(f"Rewritten: '{rewritten}'")
            except Exception as e:
                self._handle_gemini_error("Query rewriting", e)
                rewritten = query

        gemini_available = self.gemini_model and not self._gemini_disabled_reason

        # 2. Query Expansion (if enabled)
        if self.enable_expansion and gemini_available:
            try:
                expanded = self._expand_query(rewritten, num_expansions)
                logger.info(f"Generated {len(expanded)} expanded queries")
            except Exception as e:
                self._handle_gemini_error("Query expansion", e)
                expanded = []

        gemini_available = self.gemini_model and not self._gemini_disabled_reason

        # 3. Intent Classification (if enabled)
        if self.enable_intent_classification and gemini_available:
            try:
                intent = self._classify_intent(query)
                logger.info(f"Intent: {intent.intent_type} ({intent.confidence:.2f})")
            except Exception as e:
                self._handle_gemini_error("Intent classification", e)

        # 4. Fallback: Basic expansion when Gemini is unavailable.
        #    Section routing and academic filters below are pattern-based and
        #    keep working regardless, so retrieval quality degrades gracefully.
        if not expanded:
            expanded = self._basic_expansion(query)
        
        # 5. ACADEMIC ENHANCEMENTS (NEW!)
        # Detect section-specific intent
        section_intent = self.detect_section_intent(query)
        
        # Extract academic filters (year, author, venue)
        academic_filters = self.extract_academic_filters(query)
        
        # Add academic term expansions
        academic_expansions = self.expand_with_academic_terms(rewritten)
        if academic_expansions:
            expanded.extend(academic_expansions)
            # Remove duplicates
            expanded = list(set(expanded))
        
        result = ProcessedQuery(
            original_query=query,
            rewritten_query=rewritten,
            expanded_queries=expanded,
            intent=intent,
            metadata={
                "context_provided": context is not None,
                "gemini_available": self.gemini_model is not None,
                "section_intent": section_intent,  # NEW
                "academic_filters": academic_filters,  # NEW
                "is_academic_query": bool(section_intent or academic_filters)  # NEW
            }
        )
        
        logger.success(f"✅ Query processed: {len(result.get_all_queries())} total variations")
        return result
    
    def _rewrite_query(self, query: str, context: Optional[str] = None) -> str:
        """Rewrite query for clarity and completeness."""
        prompt = f"""You are a query rewriting assistant for a document search system.

Task: Rewrite the user's query to be clearer, more specific, and better suited for semantic search.

Guidelines:
- Expand abbreviations (ML → machine learning)
- Add context if missing
- Make implicit questions explicit
- Keep it concise (one sentence)
- Maintain original intent

{"Context: " + context if context else ""}
User Query: {query}

Rewritten Query:"""
        
        response = self.gemini_model.generate_content(prompt)
        rewritten = response.text.strip()
        
        # Remove quotes if present
        rewritten = rewritten.strip('"\'')
        
        return rewritten if rewritten else query
    
    def _expand_query(self, query: str, num_variations: int = 2) -> List[str]:
        """Generate multiple query variations for multi-query search."""
        prompt = f"""You are a query expansion assistant for a document search system.

Task: Generate {num_variations} alternative phrasings of the query that capture different search angles.

Guidelines:
- Each variation should approach the question differently
- Use synonyms and related terms
- Maintain original meaning
- Make each variation distinct
- Keep each variation concise (one sentence)

Original Query: {query}

Generate {num_variations} variations (one per line, no numbering):"""
        
        response = self.gemini_model.generate_content(prompt)
        text = response.text.strip()
        
        # Parse variations (split by newlines, remove numbering/bullets)
        variations = []
        for line in text.split('\n'):
            line = line.strip()
            # Remove numbering like "1.", "2)", "-", "*", etc.
            line = re.sub(r'^[\d\-\*\)\.]+\s*', '', line)
            line = line.strip('"\'')
            if line and line != query:
                variations.append(line)
        
        return variations[:num_variations]
    
    def _classify_intent(self, query: str) -> QueryIntent:
        """Classify the intent of the query."""
        prompt = f"""Classify the intent of this search query.

Query: {query}

Choose ONE intent type:
- factual: Asking for facts, definitions, or information ("What is RAG?")
- procedural: Asking how to do something ("How do I implement X?")
- comparative: Comparing options ("What's better, X or Y?")
- analytical: Asking for analysis or reasoning ("Why does X happen?")
- conversational: General discussion or unclear intent

Respond in this exact format:
Intent: [intent_type]
Confidence: [0.0-1.0]
Strategy: [recommended retrieval strategy]"""
        
        response = self.gemini_model.generate_content(prompt)
        text = response.text.strip()
        
        # Parse response
        intent_type = "conversational"  # default
        confidence = 0.5
        strategy = "hybrid"
        
        for line in text.split('\n'):
            if line.startswith('Intent:'):
                intent_type = line.split(':', 1)[1].strip().lower()
            elif line.startswith('Confidence:'):
                try:
                    confidence = float(line.split(':', 1)[1].strip())
                except (ValueError, IndexError):
                    confidence = 0.5
            elif line.startswith('Strategy:'):
                strategy = line.split(':', 1)[1].strip()
        
        return QueryIntent(
            intent_type=intent_type,
            confidence=confidence,
            description=f"Query classified as {intent_type}",
            suggested_strategy=strategy
        )
    
    def detect_section_intent(self, query: str) -> Optional[str]:
        """
        Detect if query targets a specific research paper section.
        
        ACADEMIC QUERY ROUTING:
        -----------------------
        Analyzes query to determine which section(s) of research papers
        are most relevant. This enables section-specific retrieval!
        
        Examples:
        - "How did they train the model?" → methodology
        - "What accuracy did BERT achieve?" → results
        - "Why is this approach better?" → discussion
        - "What is Transformer architecture?" → introduction/abstract
        
        Returns:
            Section type string or None if no specific section detected
        """
        query_lower = query.lower()
        
        # Methodology indicators
        methodology_patterns = [
            r'\bhow (did|do) (they|we|you|the authors)?\s*(implement|train|build|design|develop)',
            r'\b(methodology|method|approach|technique|algorithm)\b',
            r'\bwhat (algorithm|model|architecture|framework)\b',
            r'\b(implementation|experimental setup|training process)\b'
        ]
        
        # Results indicators
        results_patterns = [
            r'\b(accuracy|precision|recall|f1|performance|score|metric)\b',
            r'\b(achieve|report|obtain|show|demonstrate|outperform)\b.*\b(result)',
            r'\bwhat (accuracy|performance|results?)\b',
            r'\b(benchmark|evaluation|comparison)\b'
        ]
        
        # Discussion/Analysis indicators
        discussion_patterns = [
            r'\bwhy (does|is|did)\b',
            r'\b(advantage|disadvantage|limitation|benefit|drawback)\b',
            r'\b(analysis|discussion|interpretation|insight)\b',
            r'\b(compare|comparison|versus|vs\.?)\b'
        ]
        
        # Abstract/Introduction indicators
        intro_patterns = [
            r'\bwhat is\b',
            r'\bdefin(e|ition)\b',
            r'\bintroduc(e|tion)\b',
            r'\boverview\b',
            r'\b(summary|abstract|background)\b'
        ]
        
        # Conclusion indicators
        conclusion_patterns = [
            r'\b(conclusion|future work|summary|takeaway)\b',
            r'\bwhat (next|are the implications)\b'
        ]
        
        # Check patterns in priority order
        for pattern in methodology_patterns:
            if re.search(pattern, query_lower):
                logger.info("📊 Section routing: METHODOLOGY detected")
                return "methodology"
        
        for pattern in results_patterns:
            if re.search(pattern, query_lower):
                logger.info("📊 Section routing: RESULTS detected")
                return "results"
        
        for pattern in discussion_patterns:
            if re.search(pattern, query_lower):
                logger.info("📊 Section routing: DISCUSSION detected")
                return "discussion"
        
        for pattern in intro_patterns:
            if re.search(pattern, query_lower):
                logger.info("📊 Section routing: ABSTRACT/INTRODUCTION detected")
                return "abstract"
        
        for pattern in conclusion_patterns:
            if re.search(pattern, query_lower):
                logger.info("📊 Section routing: CONCLUSION detected")
                return "conclusion"
        
        logger.info("📊 Section routing: NO specific section detected")
        return None
    
    def extract_academic_filters(self, query: str) -> Dict[str, Any]:
        """
        Extract research paper filters from query.
        
        ACADEMIC METADATA EXTRACTION:
        -----------------------------
        Detects year ranges, authors, venues in natural language queries.
        
        Examples:
        - "papers from 2020-2023" → {"year": {"$gte": 2020, "$lte": 2023}}
        - "by Yoshua Bengio" → {"authors": "Yoshua Bengio"}
        - "NeurIPS papers about transformers" → {"venue": "NeurIPS"}
        - "recent BERT papers" → {"year": {"$gte": current_year - 2}}
        
        Returns:
            Dictionary of Qdrant filters
        """
        filters = {}
        query_lower = query.lower()
        
        # Extract year or year range
        # Pattern: "2020", "2020-2023", "from 2020 to 2023", "in 2021"
        year_match = re.search(r'\b(20\d{2})\s*(?:-|to)\s*(20\d{2})\b', query)
        if year_match:
            start_year = int(year_match.group(1))
            end_year = int(year_match.group(2))
            filters["year"] = {"$gte": start_year, "$lte": end_year}
            logger.info(f"🗓️ Year filter: {start_year}-{end_year}")
        else:
            # Single year
            single_year = re.search(r'\b(20\d{2})\b', query)
            if single_year:
                year = int(single_year.group(1))
                filters["year"] = year
                logger.info(f"🗓️ Year filter: {year}")
        
        # Extract "recent" (last 2-3 years)
        if re.search(r'\b(recent|latest|new)\b', query_lower):
            from datetime import datetime
            current_year = datetime.now().year
            filters["year"] = {"$gte": current_year - 2}
            logger.info(f"🗓️ Recent filter: >= {current_year - 2}")
        
        # Extract author names
        # Pattern: "by [Author Name]", "[Author Name] paper"
        author_match = re.search(r'\bby\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', query)
        if author_match:
            author = author_match.group(1)
            filters["authors"] = author
            logger.info(f"👤 Author filter: {author}")
        
        # Extract venue/conference
        venues = [
            "NeurIPS", "ICML", "ICLR", "CVPR", "ICCV", "ECCV", "ACL", "EMNLP",
            "NAACL", "AAAI", "IJCAI", "KDD", "WWW", "SIGIR", "ICSE", "FSE",
            "IEEE", "ACM", "Nature", "Science", "arXiv"
        ]
        for venue in venues:
            if re.search(r'\b' + re.escape(venue) + r'\b', query, re.IGNORECASE):
                filters["venue"] = venue
                logger.info(f"📍 Venue filter: {venue}")
                break
        
        # Extract paper type
        if re.search(r'\barxiv\b', query_lower):
            filters["paper_type"] = "arxiv"
            logger.info("📄 Paper type: arxiv")
        elif re.search(r'\b(conference|workshop) paper', query_lower):
            filters["paper_type"] = "conference"
            logger.info("📄 Paper type: conference")
        elif re.search(r'\bjournal (paper|article)', query_lower):
            filters["paper_type"] = "journal"
            logger.info("📄 Paper type: journal")
        
        return filters
    
    def expand_with_academic_terms(self, query: str) -> List[str]:
        """
        Expand query with academic synonyms and related terms.
        
        ACADEMIC QUERY EXPANSION:
        -------------------------
        Adds research paper terminology to improve retrieval.
        
        Examples:
        - "transformer" → ["transformer", "attention mechanism", "self-attention"]
        - "accuracy" → ["accuracy", "F1 score", "precision", "recall"]
        - "training" → ["training", "fine-tuning", "optimization"]
        
        Returns:
            List of expanded query variations
        """
        # Academic term mappings
        academic_expansions = {
            # Models/Architectures
            "transformer": ["transformer", "attention mechanism", "self-attention", "multi-head attention"],
            "bert": ["BERT", "bidirectional encoder", "masked language model"],
            "gpt": ["GPT", "generative pre-trained transformer", "autoregressive model"],
            "cnn": ["CNN", "convolutional neural network", "convolution"],
            "rnn": ["RNN", "recurrent neural network", "LSTM", "GRU"],
            
            # Metrics
            "accuracy": ["accuracy", "F1 score", "precision", "recall", "performance metric"],
            "loss": ["loss function", "objective function", "cost function"],
            "perplexity": ["perplexity", "language model evaluation", "PPL"],
            
            # Training
            "training": ["training", "fine-tuning", "optimization", "learning"],
            "learning rate": ["learning rate", "LR", "step size", "optimizer"],
            "batch size": ["batch size", "mini-batch", "training batch"],
            
            # Tasks
            "classification": ["classification", "categorization", "prediction"],
            "generation": ["generation", "text generation", "synthesis"],
            "translation": ["translation", "machine translation", "MT"],
            
            # General
            "model": ["model", "network", "architecture", "system"],
            "dataset": ["dataset", "corpus", "benchmark"],
            "baseline": ["baseline", "comparison method", "reference model"]
        }
        
        query_lower = query.lower()
        expanded_queries = [query]  # Start with original
        
        # Find matching terms and add expansions
        for term, expansions in academic_expansions.items():
            if term in query_lower:
                # Replace term with each expansion
                for expansion in expansions[:2]:  # Limit to 2 expansions per term
                    expanded = query_lower.replace(term, expansion)
                    if expanded != query_lower:
                        expanded_queries.append(expanded)
        
        # Remove duplicates and limit
        expanded_queries = list(set(expanded_queries))[:4]
        
        if len(expanded_queries) > 1:
            logger.info(f"🔬 Academic expansion: {len(expanded_queries)} variations")
        
        return expanded_queries
    
    def _basic_expansion(self, query: str) -> List[str]:
        """Basic query expansion without Gemini (fallback)."""
        # Simple rule-based expansion
        expansions = []
        
        # Add question variations
        if not query.endswith('?'):
            expansions.append(f"What is {query}?")
        
        # Add "explain" variation
        if not query.lower().startswith(('what', 'how', 'why', 'explain')):
            expansions.append(f"Explain {query}")
        
        return expansions[:2]


# Convenience function
def process_query(
    query: str,
    context: Optional[str] = None,
    num_expansions: int = 2
) -> ProcessedQuery:
    """
    Process a query with default settings.
    
    Args:
        query: User query
        context: Optional context
        num_expansions: Number of variations
        
    Returns:
        ProcessedQuery object
    """
    processor = QueryProcessor()
    return processor.process(query, context, num_expansions)
