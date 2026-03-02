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
        model: str = "gemini-2.0-flash",
        enable_rewriting: bool = True,
        enable_expansion: bool = True,
        enable_intent_classification: bool = True
    ):
        """
        Initialize QueryProcessor.
        
        Args:
            api_key: Gemini API key (uses settings if None)
            model: Gemini model to use
            enable_rewriting: Enable query rewriting
            enable_expansion: Enable multi-query expansion
            enable_intent_classification: Enable intent detection
        """
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model
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
        
        # 1. Query Rewriting (if Gemini available)
        if self.enable_rewriting and self.gemini_model:
            try:
                rewritten = self._rewrite_query(query, context)
                logger.info(f"Rewritten: '{rewritten}'")
            except Exception as e:
                logger.warning(f"Query rewriting failed: {e}, using original")
                rewritten = query
        
        # 2. Query Expansion (if enabled)
        if self.enable_expansion and self.gemini_model:
            try:
                expanded = self._expand_query(rewritten, num_expansions)
                logger.info(f"Generated {len(expanded)} expanded queries")
            except Exception as e:
                logger.warning(f"Query expansion failed: {e}")
                expanded = []
        
        # 3. Intent Classification (if enabled)
        if self.enable_intent_classification and self.gemini_model:
            try:
                intent = self._classify_intent(query)
                logger.info(f"Intent: {intent.intent_type} ({intent.confidence:.2f})")
            except Exception as e:
                logger.warning(f"Intent classification failed: {e}")
        
        # 4. Fallback: Basic expansion if Gemini not available
        if not expanded and not self.gemini_model:
            expanded = self._basic_expansion(query)
        
        result = ProcessedQuery(
            original_query=query,
            rewritten_query=rewritten,
            expanded_queries=expanded,
            intent=intent,
            metadata={
                "context_provided": context is not None,
                "gemini_available": self.gemini_model is not None
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
                except:
                    confidence = 0.5
            elif line.startswith('Strategy:'):
                strategy = line.split(':', 1)[1].strip()
        
        return QueryIntent(
            intent_type=intent_type,
            confidence=confidence,
            description=f"Query classified as {intent_type}",
            suggested_strategy=strategy
        )
    
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
