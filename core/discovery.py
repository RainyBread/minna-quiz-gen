"""Auto-discovery for optional local resources.

No dictionary version or vendor path is hard-coded.  We scan the configured
UniDic search root (default: ``data/``) for MeCab dictionary directories.
"""

from __future__ import annotations

from pathlib import Path

from config import Settings

_SKIP_DIR_NAMES = {
    "cache",
    "db",
    "logs",
    "mdicts",
    "教材pdf",
    "dlut-nihongo-quiz",
    "__pycache__",
    ".git",
    ".venv",
}


def _is_unidic_dir(path: Path) -> bool:
    return (
        (path / "sys.dic").is_file()
        and (path / "dicrc").is_file()
        and (path / "matrix.bin").is_file()
    )


def discover_unidic(settings: Settings) -> Path | None:
    """Return the best local UniDic directory, or None."""
    override = Path(settings.unidic_root)
    if _is_unidic_dir(override):
        return override
    if not override.exists():
        return None

    candidates: list[Path] = []
    try:
        for child in override.rglob("sys.dic"):
            parent = child.parent
            if any(part.lower() in _SKIP_DIR_NAMES for part in parent.parts):
                continue
            if len(parent.relative_to(override).parts) > 5:
                continue
            if _is_unidic_dir(parent):
                candidates.append(parent)
    except OSError:
        return None
    if not candidates:
        return None
    # Prefer the most recently modified dictionary; path is a deterministic tie-break.
    candidates.sort(key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)
    return candidates[0]


def discover_unidic_candidates(settings: Settings) -> list[Path]:
    override = Path(settings.unidic_root)
    candidates: list[Path] = []
    if _is_unidic_dir(override):
        candidates.append(override)
    if not override.exists():
        return candidates
    try:
        for child in override.rglob("sys.dic"):
            parent = child.parent
            if any(part.lower() in _SKIP_DIR_NAMES for part in parent.parts):
                continue
            if len(parent.relative_to(override).parts) > 5:
                continue
            if _is_unidic_dir(parent) and parent not in candidates:
                candidates.append(parent)
    except OSError:
        pass
    return sorted(candidates, key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)


def create_unidic(settings: Settings, *, verbose: bool = False):
    """Create a UniDicRetriever if a dictionary and fugashi are available."""
    path = discover_unidic(settings)
    if path is None:
        return None
    try:
        from core.retriever import UniDicRetriever

        retriever = UniDicRetriever(
            path, cache_dir=settings.cache_dir, verbose=verbose
        )
        # Force a load so missing fugashi/broken dictionary fails here.
        retriever.get_reading("学校")
        return retriever
    except Exception:
        return None
