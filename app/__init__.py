"""Independent application backend; importing this package starts no services."""

# Bump when a running API must not be reused by newer desktop/frontend code.
# Version 3 requires the dynamically discovered /api/v1/adapters catalog.
# Keep this lightweight: the desktop must not import API services to check it.
INSTALLATION_WORKFLOW_VERSION = 3
