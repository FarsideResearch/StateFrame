"""Pinned, self-contained repository-intelligence primitives.

Source provenance is explicit because these pinned primitives must remain
self-contained at runtime.
"""

import os

SOURCE_PATH = os.environ.get("STATEFRAME_VENDOR_SOURCE_PATH", "")
MC_COMMIT = "7bcb7efef5dd579a04cea5f005ae5d0a63365a83"
VENDOR_SCHEMA = "mc-repo-intel-vendor/1"

__all__ = ["MC_COMMIT", "SOURCE_PATH", "VENDOR_SCHEMA"]
