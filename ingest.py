"""Ingestion entry point: auto-discover corpus, parse, index, generate registry."""
import sys
import sqlite3
import json
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from langchain_openai import ChatOpenAI

from src.config import settings
from src.ingestion.pdf_parser import parse_pdf, get_first_n_pages_text
from src.ingestion.indexer import DocumentIndexer
from src.registry import Registry


console = Console()


def generate_collection_metadata(pdf_path: Path, indexer: DocumentIndexer) -> dict:
    """
    Generate collection metadata using LLM.
    
    Uses PROMPT 7 from the spec.
    """
    # Get first 3 pages of text
    text_samples = get_first_n_pages_text(pdf_path, n=3)
    
    # Truncate if too long
    if len(text_samples) > 2000:
        text_samples = text_samples[:2000] + "..."
    
    prompt = f"""SYSTEM:
You are indexing a document into a research knowledge base.
Read the provided text samples and generate metadata that will help
an AI agent decide whether to search this collection for a given question.

Rules:
1. display_name: short human-readable name (3-5 words)
2. description: 1-2 sentences describing what topics and data this document covers
3. keywords: 15-20 specific terms a user would use when asking about this document's content.
   Include entity names, column headers, topic areas, and key terminology.
4. Return ONLY JSON

OUTPUT FORMAT:
{{
  "display_name": "Product Technical Specifications",
  "description": "Contains specifications, pricing, battery life, and performance data for all product models.",
  "keywords": ["battery", "payload", "price", "speed", "model", "spec", "dimensions", ...]
}}

USER:
Document samples:
{text_samples}

Generate collection metadata."""
    
    llm = ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        model=settings.llm_model,
        temperature=0.0
    )
    
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # Extract JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        return json.loads(content)
    
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to generate metadata via LLM: {e}[/yellow]")
        # Fallback to simple metadata
        return {
            "display_name": pdf_path.stem.replace("_", " ").title(),
            "description": f"Content from {pdf_path.name}",
            "keywords": [pdf_path.stem.lower()]
        }


def ingest_pdfs(indexer: DocumentIndexer, registry: Registry):
    """Discover and index all PDFs."""
    pdf_dir = settings.corpus_pdf_path
    
    if not pdf_dir.exists():
        console.print(f"[yellow]PDF directory not found: {pdf_dir}[/yellow]")
        return 0
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        console.print("[yellow]No PDF files found in corpus/pdfs/[/yellow]")
        return 0
    
    console.print(f"\n[bold]Indexing {len(pdf_files)} PDF files...[/bold]\n")
    
    total_chunks = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        for pdf_path in pdf_files:
            task = progress.add_task(f"Processing {pdf_path.name}...", total=None)
            
            try:
                # Generate collection key (slug from filename)
                collection_key = pdf_path.stem.lower().replace(" ", "_")
                
                # Parse PDF
                chunks = list(parse_pdf(pdf_path))
                
                if not chunks:
                    progress.update(task, description=f"[yellow]{pdf_path.name}: No content[/yellow]")
                    continue
                
                # Index document
                chunk_count, centroid = indexer.index_document(
                    collection_key=collection_key,
                    chunks=iter(chunks),  # Convert back to iterator
                    source_filename=pdf_path.name
                )
                
                # Generate metadata
                metadata = generate_collection_metadata(pdf_path, indexer)
                
                # Add to registry
                registry.add_collection(
                    collection_key=collection_key,
                    display_name=metadata.get("display_name", pdf_path.stem),
                    description=metadata.get("description", ""),
                    keywords=metadata.get("keywords", []),
                    source_files=[pdf_path.name],
                    chunk_count=chunk_count,
                    centroid_embedding=centroid
                )
                
                total_chunks += chunk_count
                
                progress.update(
                    task,
                    description=f"[green]✓ {pdf_path.name}: {chunk_count} chunks[/green]"
                )
                
            except Exception as e:
                progress.update(
                    task,
                    description=f"[red]✗ {pdf_path.name}: {str(e)}[/red]"
                )
    
    return total_chunks


def ingest_databases(registry: Registry):
    """Discover and introspect SQLite databases."""
    db_dir = settings.sqlite_path
    
    if not db_dir.exists():
        console.print(f"[yellow]Database directory not found: {db_dir}[/yellow]")
        return 0
    
    db_files = list(db_dir.glob("*.sqlite")) + list(db_dir.glob("*.db"))
    
    if not db_files:
        console.print("[yellow]No SQLite databases found in corpus/databases/[/yellow]")
        return 0
    
    console.print(f"\n[bold]Introspecting {len(db_files)} databases...[/bold]\n")
    
    for db_path in db_files:
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Build schema
            schema_parts = []
            sample_parts = []
            
            for table in tables:
                # Get schema
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                
                col_defs = [f"{col[1]} {col[2]}" for col in columns]
                schema_parts.append(f"CREATE TABLE {table} ({', '.join(col_defs)})")
                
                # Get sample rows
                cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                rows = cursor.fetchall()
                col_names = [col[1] for col in columns]
                
                if rows:
                    sample_parts.append(f"Table {table}:")
                    for row in rows:
                        row_dict = dict(zip(col_names, row))
                        sample_parts.append(f"  {row_dict}")
            
            conn.close()
            
            schema = "\n".join(schema_parts)
            sample_rows = "\n".join(sample_parts)
            
            # Generate description
            description = f"Contains tables: {', '.join(tables)}"
            
            # Add to registry
            registry.add_sql_database(
                db_path=str(db_path),
                schema=schema,
                sample_rows=sample_rows,
                description=description
            )
            
            console.print(f"[green]✓ {db_path.name}: {len(tables)} tables[/green]")
            
        except Exception as e:
            console.print(f"[red]✗ {db_path.name}: {str(e)}[/red]")
    
    return len(db_files)


def main():
    """Main ingestion entry point."""
    console.print("[bold cyan]Multi-Agent Research System - Corpus Ingestion[/bold cyan]\n")
    
    # Initialize components
    indexer = DocumentIndexer()
    registry = Registry()
    
    # Ingest PDFs
    total_chunks = ingest_pdfs(indexer, registry)
    
    # Ingest databases
    total_dbs = ingest_databases(registry)
    
    # Save registry
    registry.save()
    
    # Summary
    console.print(f"\n[bold green]✓ Ingestion complete![/bold green]\n")
    console.print(f"Collections: {len(registry.get_all_collections())}")
    console.print(f"Total chunks: {total_chunks}")
    console.print(f"Databases: {total_dbs}")
    console.print(f"Registry: {settings.registry_path}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Ingestion cancelled[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        sys.exit(1)
