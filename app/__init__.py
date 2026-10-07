"""Independent application backend; importing this package starts no services."""

# Bump when a running API must not be reused by a newer desktop installation flow.
# Keep this lightweight: the desktop must not import API services to check it.
INSTALLATION_WORKFLOW_VERSION = 2
