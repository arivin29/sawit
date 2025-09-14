from pathlib import Path
import os


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_root() -> Path:
    # Prefer current working directory as project root
    return Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve()


def path_from_root(*parts: str) -> Path:
    return project_root().joinpath(*parts)

