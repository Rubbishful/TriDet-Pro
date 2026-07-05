"""E2E: Backward-compatible shim — re-exports from E2E.module.

After the refactor, backend modules live under E2E.module/.
This __init__.py preserves the legacy ``from E2E import ...`` interface.
New code should prefer ``from E2E.module import ...`` directly.
"""

from E2E.module import *  # noqa: F401,F403
