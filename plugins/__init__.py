"""Plugin registration point.

Adding a plugin is one file plus one import line here; ``core/`` stays
untouched.  Import order matters only for duplicate-name detection.
"""

from plugins.dict_sources import builtin as _dict_sources_builtin  # noqa: F401
from plugins.exporters import builtin as _exporters_builtin  # noqa: F401
from plugins.exporters import demo as _exporters_demo  # noqa: F401
from plugins.llm_providers import builtin as _llm_providers_builtin  # noqa: F401
from plugins.question_types import builtin as _question_types_builtin  # noqa: F401
from plugins.question_types import demo as _question_types_demo  # noqa: F401
from plugins.source_strategies import builtin as _source_strategies_builtin  # noqa: F401
