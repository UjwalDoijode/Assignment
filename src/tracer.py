"""Tracer for console and JSON logging of execution."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

from src.config import settings


class ExecutionTracer:
    """Handles Rich console output and JSON trace logging."""
    
    def __init__(self, question: str):
        self.console = Console()
        self.question = question
        self.start_time = datetime.now()
        
        # Trace data for JSON export
        self.trace_data = {
            "question": question,
            "start_time": self.start_time.isoformat(),
            "events": []
        }
        
        # Generate trace file path
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.trace_file = settings.trace_output_dir / f"trace_{timestamp}.json"
    
    def log_event(self, event_type: str, data: Any):
        """Log an event to both console and JSON trace."""
        timestamp = datetime.now().isoformat()
        
        event = {
            "timestamp": timestamp,
            "type": event_type,
            "data": data
        }
        
        self.trace_data["events"].append(event)
    
    def print_header(self):
        """Print initial header."""
        self.console.print(
            Panel(
                f"[bold cyan]Multi-Agent Deep Research System[/bold cyan]\n\n"
                f"Question: [white]{self.question}[/white]",
                border_style="cyan"
            )
        )
        self.console.print()
    
    def print_orchestrator_result(self, sub_questions: list):
        """Print orchestrator decomposition."""
        self.console.print("[bold green]📋 Orchestrator:[/bold green] Decomposed into sub-questions")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID", style="dim")
        table.add_column("Question")
        table.add_column("Intent", style="cyan")
        table.add_column("Target")
        
        for sq in sub_questions:
            target = sq.target_collection or sq.intent
            table.add_row(sq.id, sq.question, sq.intent, target)
        
        self.console.print(table)
        self.console.print()
        
        self.log_event("orchestrator", {
            "sub_questions": [sq.model_dump() for sq in sub_questions]
        })
    
    def print_agent_start(self, agent_name: str):
        """Print agent start."""
        self.console.print(f"[bold yellow]🤖 {agent_name}:[/bold yellow] Starting...")
        self.log_event(f"{agent_name}_start", {})
    
    def print_agent_result(self, agent_name: str, results: list):
        """Print agent results."""
        for result in results:
            if result.agent_used == agent_name:
                status = "✅" if result.sufficient else "⚠️"
                self.console.print(
                    f"  {status} {result.sub_question_id}: "
                    f"{len(result.evidence)} evidence chunks"
                )
        
        self.console.print()
        
        self.log_event(f"{agent_name}_result", {
            "results": [r.model_dump() for r in results if r.agent_used == agent_name]
        })
    
    def print_synthesis_start(self):
        """Print synthesis start."""
        self.console.print("[bold blue]📝 Synthesis:[/bold blue] Composing research brief...")
        self.log_event("synthesis_start", {})
    
    def print_final_brief(self, brief: str, verification_passed: bool, verification_notes: str):
        """Print final brief."""
        self.console.print()
        self.console.print(
            Panel(
                Markdown(brief),
                title="[bold green]Research Brief[/bold green]",
                border_style="green"
            )
        )
        
        # Verification status
        if verification_passed:
            self.console.print("\n[bold green]✅ Verification: PASSED[/bold green]")
        else:
            self.console.print("\n[bold yellow]⚠️ Verification: ISSUES FOUND[/bold yellow]")
        
        if verification_notes:
            self.console.print(f"[dim]{verification_notes}[/dim]")
        
        self.log_event("synthesis_complete", {
            "brief": brief,
            "verification_passed": verification_passed,
            "verification_notes": verification_notes
        })
    
    def print_error(self, error: str):
        """Print error."""
        self.console.print(f"\n[bold red]❌ Error:[/bold red] {error}")
        self.log_event("error", {"error": error})
    
    def print_footer(self):
        """Print footer with timing and trace file info."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        self.console.print()
        self.console.print(
            Panel(
                f"[bold]Execution complete[/bold]\n\n"
                f"Duration: {duration:.2f}s\n"
                f"Trace file: [cyan]{self.trace_file}[/cyan]",
                border_style="dim"
            )
        )
        
        self.trace_data["end_time"] = end_time.isoformat()
        self.trace_data["duration_seconds"] = duration
    
    def save_trace(self):
        """Save JSON trace to file."""
        with open(self.trace_file, "w", encoding="utf-8") as f:
            json.dump(self.trace_data, f, indent=2, ensure_ascii=False)


# Global tracer instance
_tracer = None


def init_tracer(question: str) -> ExecutionTracer:
    """Initialize global tracer."""
    global _tracer
    _tracer = ExecutionTracer(question)
    _tracer.print_header()
    return _tracer


def get_tracer() -> ExecutionTracer:
    """Get global tracer instance."""
    return _tracer
