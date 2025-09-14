from rich.console import Console
from rich.theme import Theme

_theme = Theme({
    "info": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "bold red",
})

console = Console(theme=_theme)

def info(msg: str):
    console.print(msg, style="info")

def success(msg: str):
    console.print(msg, style="success")

def warn(msg: str):
    console.print(msg, style="warning")

def error(msg: str):
    console.print(msg, style="error")

