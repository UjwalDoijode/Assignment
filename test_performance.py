"""Quick performance test to compare different optimization modes."""

import os
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.graph import build_research_graph


def test_query(mode_name: str, query: str = "What models does Meridian offer?"):
    """Test a query and measure execution time."""
    print(f"\n{'='*80}")
    print(f"Testing: {mode_name}")
    print(f"Query: {query}")
    print('='*80)
    
    graph = build_research_graph()
    
    start = time.time()
    
    try:
        # Run the graph
        result = graph.invoke({
            "original_question": query,
            "sub_questions": [],
            "sub_results": [],
            "final_brief": "",
            "is_complete": False,
            "needs_web": False,
            "error": None,
            "verification_passed": False,
            "verification_notes": "",
            "citations": []
        })
        
        elapsed = time.time() - start
        
        print(f"\n✓ Success in {elapsed:.2f}s")
        print(f"  Sub-questions: {len(result.get('sub_questions', []))}")
        print(f"  Evidence chunks: {sum(len(r.get('evidence', [])) for r in result.get('sub_results', []))}")
        print(f"  Brief length: {len(result.get('final_brief', ''))} chars")
        
        return elapsed
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n✗ Failed in {elapsed:.2f}s")
        print(f"  Error: {str(e)[:100]}")
        return elapsed


def main():
    """Run performance comparison tests."""
    
    print("""
╔════════════════════════════════════════════════════════════════╗
║         Multi-Agent Research System - Performance Test         ║
╚════════════════════════════════════════════════════════════════╝

This script tests different optimization modes to help you choose
the right performance/quality trade-off for your use case.

Make sure you've set your API keys in .env:
  - OPENROUTER_API_KEY
  - TAVILY_API_KEY
""")
    
    input("Press Enter to start tests...")
    
    results = {}
    
    # Test 1: Ultra-Fast Mode
    os.environ["DIRECT_RETRIEVAL"] = "true"
    os.environ["SKIP_VERIFICATION"] = "true"
    os.environ["MAX_RAG_ITERATIONS"] = "1"
    results["Ultra-Fast"] = test_query("Ultra-Fast Mode (DIRECT_RETRIEVAL=true, SKIP_VERIFICATION=true)")
    
    # Test 2: Balanced Mode
    os.environ["DIRECT_RETRIEVAL"] = "false"
    os.environ["SKIP_VERIFICATION"] = "true"
    os.environ["MAX_RAG_ITERATIONS"] = "1"
    results["Balanced"] = test_query("Balanced Mode (Agentic retrieval, skip verification)")
    
    # Test 3: Full Quality Mode
    os.environ["DIRECT_RETRIEVAL"] = "false"
    os.environ["SKIP_VERIFICATION"] = "false"
    os.environ["MAX_RAG_ITERATIONS"] = "3"
    results["Full Quality"] = test_query("Full Quality Mode (All features enabled)")
    
    # Summary
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print('='*80)
    print(f"\n{'Mode':<20} {'Time (s)':<12} {'Speedup':<10}")
    print('-'*42)
    
    baseline = results.get("Full Quality", 1.0)
    
    for mode, elapsed in results.items():
        speedup = f"{baseline / elapsed:.1f}x" if elapsed > 0 else "N/A"
        print(f"{mode:<20} {elapsed:>6.2f}s      {speedup}")
    
    print("\nRecommendation:")
    print("  • Ultra-Fast: Best for demos, testing, interactive chat")
    print("  • Balanced:   Best for production (good speed + accuracy)")
    print("  • Full Quality: Best for critical research reports")
    
    print("\nTo use a mode permanently, add these to your .env:")
    print("\n  # Ultra-Fast Mode")
    print("  DIRECT_RETRIEVAL=true")
    print("  SKIP_VERIFICATION=true")
    print("  MAX_RAG_ITERATIONS=1")
    
    print("\nSee docs/PERFORMANCE.md for detailed optimization guide.")


if __name__ == "__main__":
    main()
