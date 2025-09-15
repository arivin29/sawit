from rich.console import Console
from rich.theme import Theme

_theme = Theme({
    "info": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "bold red",
})

# Disable Rich markup to prevent errors when messages contain
# square brackets that look like tags (e.g., "valid[/test]").
console = Console(theme=_theme, markup=False)

def info(msg: str):
    console.print(msg, style="info")

def success(msg: str):
    console.print(msg, style="success")

def warn(msg: str):
    console.print(msg, style="warning")

def error(msg: str):
    console.print(msg, style="error")
