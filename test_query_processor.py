"""
Test QueryProcessor - Phase 3 Query Processing

Tests:
1. Basic query processing (without Gemini)
2. Query rewriting with Gemini
3. Query expansion (multi-query generation)
4. Intent classification
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from docusense.retrieval import QueryProcessor, process_query


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_query_processor():
    """Test query processor functionality."""
    
    print_section("TEST 1: Basic Query Processing (No Gemini)")
    
    # Test without Gemini (basic expansion)
    processor = QueryProcessor(api_key=None)
    
    query = "What is RAG?"
    result = processor.process(query, num_expansions=2)
    
    print(f"📝 Original Query: {result.original_query}")
    print(f"✏️  Rewritten Query: {result.rewritten_query}")
    print(f"📚 Expanded Queries ({len(result.expanded_queries)}):")
    for i, exp in enumerate(result.expanded_queries, 1):
        print(f"   {i}. {exp}")
    
    print(f"\n🔍 All Query Variations:")
    for i, q in enumerate(result.get_all_queries(), 1):
        print(f"   {i}. {q}")
    
    print(f"\n✅ KEY INSIGHT:")
    print(f"   Even without Gemini, basic expansion creates search variations!")
    
    print_section("TEST 2: Query Processing with Gemini (if available)")
    
    # Try with Gemini (will use settings.gemini_api_key)
    processor_gemini = QueryProcessor()
    
    if processor_gemini.gemini_model:
        print("✅ Gemini API available!\n")
        
        test_queries = [
            "ML algorithms",
            "How does RAG work?",
            "Compare vector search and keyword search"
        ]
        
        for query in test_queries:
            print(f"\n{'─' * 60}")
            print(f"📝 Query: {query}")
            print(f"{'─' * 60}")
            
            result = processor_gemini.process(query, num_expansions=2)
            
            print(f"✏️  Rewritten: {result.rewritten_query}")
            print(f"📚 Expansions:")
            for i, exp in enumerate(result.expanded_queries, 1):
                print(f"   {i}. {exp}")
            
            if result.intent:
                print(f"🎯 Intent: {result.intent.intent_type} (confidence: {result.intent.confidence:.2f})")
                print(f"💡 Strategy: {result.intent.suggested_strategy}")
        
        print(f"\n✅ KEY INSIGHT:")
        print(f"   Gemini enhances queries for better retrieval!")
        print(f"   - Expands abbreviations (ML → machine learning)")
        print(f"   - Creates diverse search angles")
        print(f"   - Classifies intent for strategy selection")
    
    else:
        print("ℹ️  Gemini not available (no API key or import failed)")
        print("   Query processor works in basic mode with rule-based expansion")
    
    print_section("TEST 3: Convenience Function")
    
    result = process_query("What are the benefits of semantic search?")
    print(f"📝 Original: {result.original_query}")
    print(f"✏️  Rewritten: {result.rewritten_query}")
    print(f"📚 Total variations: {len(result.get_all_queries())}")
    
    print(f"\n✅ KEY INSIGHT:")
    print(f"   process_query() is a quick convenience wrapper!")
    
    print_section("SUMMARY: Query Processor Tests")
    
    print("✅ Query Processor Module Complete!\n")
    print("📦 What we built:")
    print("   1. ✅ QueryProcessor class")
    print("   2. ✅ Query rewriting (Gemini-powered)")
    print("   3. ✅ Query expansion (multi-query search)")
    print("   4. ✅ Intent classification")
    print("   5. ✅ Basic fallback (works without Gemini)\n")
    
    print("🔥 Next Steps:")
    print("   - Hybrid Search (Vector + BM25)")
    print("   - Reranker (Cross-encoder)")
    print("   - Context Builder")
    print("   - Retrieval Pipeline\n")


if __name__ == "__main__":
    test_query_processor()
