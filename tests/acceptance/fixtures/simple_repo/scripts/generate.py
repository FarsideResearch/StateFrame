from pathlib import Path

Path("generated.py").write_text('GENERATED = "ready"\n')
print("GENERATED_OK")
