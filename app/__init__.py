"""Independent application backend; importing this package starts no services."""

# The application version reported by /api/v1/system/status. The desktop compares it
# with its own copy: a mismatch means an older backend still owns the port. Keep this
# in sync with pyproject.toml, and keep it lightweight (the desktop imports this file).
APP_VERSION = "1.2.9"

# Bump when a running API must not be reused by newer desktop/frontend code.
# Version 3 requires the dynamically discovered /api/v1/adapters catalog.
# Keep this lightweight: the desktop must not import API services to check it.
INSTALLATION_WORKFLOW_VERSION = 3
