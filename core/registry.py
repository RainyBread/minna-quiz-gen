"""Small plugin registries.

Concrete implementations live outside ``core/`` under ``plugins/``.  Core code
asks a registry for an interface; adding a plugin is one file plus one import
line in ``plugins/__init__.py``.
"""

from __future__ import annotations

from typing import Any, Callable, Generic, Iterator, TypeVar

T = TypeVar("T")

_PLUGINS_LOADED = False
_PLUGIN_ERROR = ""


class RegistryError(RuntimeError):
    pass


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(
        self,
        name: str,
        value: T | None = None,
        *,
        replace: bool = False,
    ) -> Any:
        key = str(name or "").strip()
        if not key:
            raise RegistryError(f"{self.kind}: empty plugin name")

        if value is None:
            def decorator(item: T) -> T:
                self.register(key, item, replace=replace)
                return item

            return decorator

        if key in self._items and not replace:
            raise RegistryError(f"{self.kind}: duplicate plugin {key!r}")
        self._items[key] = value
        return value

    def get(self, name: str) -> T:
        ensure_plugins_loaded()
        key = str(name or "").strip()
        try:
            return self._items[key]
        except KeyError as exc:
            detail = f"（插件加载错误：{_PLUGIN_ERROR}）" if _PLUGIN_ERROR else ""
            raise RegistryError(
                f"{self.kind}: 未注册 {key!r}{detail}"
            ) from exc

    def maybe_get(self, name: str) -> T | None:
        ensure_plugins_loaded()
        return self._items.get(str(name or "").strip())

    def names(self) -> list[str]:
        ensure_plugins_loaded()
        return list(self._items)

    def all(self) -> dict[str, T]:
        ensure_plugins_loaded()
        return dict(self._items)

    def __contains__(self, name: str) -> bool:
        return self.maybe_get(name) is not None

    def clear(self) -> None:
        self._items.clear()


question_types: Registry[Any] = Registry("question_type")
dict_sources: Registry[Any] = Registry("dict_source")
source_strategies: Registry[Any] = Registry("source_strategy")
exporters: Registry[Any] = Registry("exporter")
llm_providers: Registry[Any] = Registry("llm_provider")

_REGISTRIES = (
    question_types,
    dict_sources,
    source_strategies,
    exporters,
    llm_providers,
)


def ensure_plugins_loaded() -> None:
    """Import the plugin package once.  Safe to call from any core entry."""
    global _PLUGINS_LOADED, _PLUGIN_ERROR
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    try:
        import plugins  # noqa: F401
    except Exception as exc:  # pragma: no cover - visible through get()
        _PLUGIN_ERROR = f"{type(exc).__name__}: {exc}"


def plugin_status() -> dict[str, Any]:
    ensure_plugins_loaded()
    return {
        "loaded": _PLUGINS_LOADED,
        "error": _PLUGIN_ERROR,
        "question_types": question_types.names(),
        "dict_sources": dict_sources.names(),
        "source_strategies": source_strategies.names(),
        "exporters": exporters.names(),
        "llm_providers": llm_providers.names(),
    }


def clear_registries() -> None:
    """Test helper: clear registrations but keep the loaded flag."""
    for registry in _REGISTRIES:
        registry.clear()
