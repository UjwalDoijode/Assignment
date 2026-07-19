"""Main entry point for the research system."""
import sys
from pathlib import Path
from rich.console import Console

from src.config import settings
from src.models import ResearchState
from src.graph import build_research_graph
from src.tracer import init_tracer
from src.registry import Registry


console = Console()

# LangSmith is configured automatically via settings.__init__


def main():
    """Main research entry point."""
    # Get question from command line
    if len(sys.argv) < 2:
        console.print("[red]Error: No question provided[/red]")
        console.print("\nUsage: python main.py \"your research question here\"")
        sys.exit(1)
    
    question = " ".join(sys.argv[1:])
    
    # Check registry exists
    if not settings.registry_path.exists():
        console.print("[red]Error: Registry not found. Run ingest.py first.[/red]")
        console.print(f"\nExpected: {settings.registry_path}")
        sys.exit(1)
    
    # Verify registry has content
    registry = Registry()
    if not registry.get_all_collections() and not registry.get_all_databases():
        console.print("[red]Error: Registry is empty. Run ingest.py to index your corpus.[/red]")
        sys.exit(1)
    
    # Initialize tracer
    tracer = init_tracer(question)
    
    try:
        # Build graph
        graph = build_research_graph()
        
        # Initialize state
        initial_state = ResearchState(original_question=question)
        
        # Execute graph
        final_state = None
        
        for step_output in graph.stream(initial_state.model_dump()):
            # Extract node name and state dict
            node_name = list(step_output.keys())[0]
            state_dict = step_output[node_name]
            
            # Convert dict back to ResearchState for processing
            state = ResearchState(**state_dict)
            
            # Log to tracer
            if node_name == "orchestrator":
                tracer.print_orchestrator_result(state.sub_questions)
            
            elif node_name in ["rag_agent", "sql_agent", "web_agent"]:
                tracer.print_agent_start(node_name)
                tracer.print_agent_result(node_name, state.sub_results)
            
            elif node_name == "synthesis":
                tracer.print_synthesis_start()
            
            final_state = state
        
        # Print final results
        if final_state:
            if final_state.error:
                tracer.print_error(final_state.error)
            elif final_state.final_brief:
                tracer.print_final_brief(
                    final_state.final_brief,
                    final_state.verification_passed,
                    final_state.verification_notes
                )
            else:
                tracer.print_error("No research brief generated")
        
        # Print footer
        tracer.print_footer()
        
        # Save trace
        tracer.save_trace()
        
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Research cancelled by user[/yellow]")
        tracer.save_trace()
        sys.exit(1)
    
    except Exception as e:
        console.print(f"\n\n[red]Unexpected error: {str(e)}[/red]")
        import traceback
        console.print(traceback.format_exc())
        tracer.save_trace()
        sys.exit(1)


if __name__ == "__main__":
    main()
