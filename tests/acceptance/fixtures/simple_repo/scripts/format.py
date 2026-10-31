from pathlib import Path

path = Path("app.py")
path.write_text(path.read_text().replace('VALUE = "before"', 'VALUE = "after"'))
print("FORMATTED")
