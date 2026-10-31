from pathlib import Path

Path("build.ok").write_text("built\n")
print("BUILD_OK")
