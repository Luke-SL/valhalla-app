#!/usr/bin/env python3
import sys
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

from rag_engine import PDFDocumentLoader, VectorStoreManager, NormasRAG

console = Console()

def build_index(force: bool = False):
    console.print("[bold blue]Iniciando carregamento dos PDFs...[/bold blue]")
    loader = PDFDocumentLoader()
    chunks = loader.extract_chunks()
    console.print(f"[green]Extraídos {len(chunks)} chunks dos documentos PDF em /home/lucas/workspace/valhalla/documentos.[/green]")

    vector_mgr = VectorStoreManager()
    vector_mgr.build_index(chunks, force=force)
    console.print("[bold green] Indexação concluída com sucesso![/bold green]")

def run_query(rag: NormasRAG, question: str, top_k: int = 5, category: str = "Todas"):
    console.print(f"\n[bold yellow]Pergunta:[/bold yellow] {question}")
    if category != "Todas":
        console.print(f"[dim]Filtrando por categoria: {category}[/dim]")
    
    with console.status("[bold green]Buscando documentos e gerando resposta com Llama 3.2...[/bold green]"):
        res = rag.query(question, top_k=top_k, category_filter=category if category != "Todas" else None)

    console.print("\n" + "="*80)
    console.print(Panel(Markdown(res["answer"]), title="[bold cyan]Resposta do Llama 3.2[/bold cyan]", border_style="cyan"))

    if res["sources"]:
        table = Table(title=" Fontes Citadas / Trechos Utilizados", border_style="dim")
        table.add_column("#", style="cyan", width=4)
        table.add_column("Documento", style="green")
        table.add_column("Pág.", style="magenta", width=6)
        table.add_column("Categoria", style="yellow")
        table.add_column("Similaridade", style="blue", width=12)
        table.add_column("Trecho do Texto", style="white")

        for idx, src in enumerate(res["sources"], 1):
            meta = src["metadata"]
            snippet = src["text"][:120].replace("\n", " ") + "..."
            table.add_row(
                str(idx),
                meta["file_name"],
                str(meta["page_number"]),
                meta["category"],
                f"{src['similarity']*100:.1f}%",
                snippet
            )

        console.print(table)
    console.print("="*80 + "\n")

def interactive_mode(rag: NormasRAG):
    console.print(Panel.fit(
        "[bold cyan]Sistema RAG - Consulta às Normas e Resoluções (Llama 3.2)[/bold cyan]\n"
        "Digite a sua pergunta ou '[bold red]sair[/bold red]' para encerrar.",
        border_style="green"
    ))

    while True:
        try:
            query = console.input("[bold green]Perguntar > [/bold green]").strip()
            if not query:
                continue
            if query.lower() in ["sair", "exit", "quit", "q"]:
                console.print("[yellow]Encerrando o sistema. Até logo![/yellow]")
                break
            if query.lower() == "reindexar":
                build_index(force=True)
                continue
            
            run_query(rag, query)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Encerrado.[/yellow]")
            break

def main():
    parser = argparse.ArgumentParser(description="RAG com Llama 3.2 para consultar normas institucionais (PDFs)")
    parser.add_argument("--index", action="store_true", help="Reconstruir o índice vetorial de PDFs")
    parser.add_argument("--force", action="store_true", help="Forçar a reconstrução do índice do zero")
    parser.add_argument("--ask", type=str, help="Pergunta direta via CLI")
    parser.add_argument("--top-k", type=int, default=5, help="Número de trechos a recuperar (padrão: 5)")
    parser.add_argument("--category", type=str, default="Todas", help="Filtrar por subpasta de documento")

    args = parser.parse_args()

    if args.index or args.force:
        build_index(force=args.force)
        if not args.ask:
            return

    rag = NormasRAG()

    if not rag.check_ollama_status():
        console.print("[bold red]AVISO: Não foi possível conectar ao servidor Ollama (http://localhost:11434).[/bold red]")
        console.print("[dim]Certifique-se de que o Ollama está rodando ('ollama serve').[/dim]\n")

    if args.ask:
        run_query(rag, args.ask, top_k=args.top_k, category=args.category)
    else:
        interactive_mode(rag)

if __name__ == "__main__":
    main()
