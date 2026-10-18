"""Single entry point for the sealed deployment."""
import os
import traceback
try:
    from dsv4_runtime.bootstrap import initialize
    initialize()
except BaseException:
    traceback.print_exc()
    os._exit(70)
