"""Web UI command launcher."""

from __future__ import annotations

import typer
from rich.console import Console

from vaquila.exceptions import VaquilaError
from vaquila.webui import create_web_app

console = Console()


def cmd_ui(host: str, port: int) -> None:
    """Start the local vAquila Web UI server."""
    try:
        try:
            import uvicorn
        except ImportError as exc:
            raise VaquilaError(
                "Missing `uvicorn` dependency. Reinstall vAquila with web dependencies enabled."
            ) from exc

        console.print(f"[bold green]✅ vAquila Web UI[/bold green] available at [cyan]http://{host}:{port}[/cyan]")
        uvicorn.run(create_web_app(), host=host, port=port, log_level="info")
    except VaquilaError as exc:
        console.print(f"[bold red]❌ {exc}[/bold red]")
        raise typer.Exit(code=1)
