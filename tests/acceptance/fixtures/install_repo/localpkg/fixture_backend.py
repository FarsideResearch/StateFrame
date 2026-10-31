"""Offline fixture wheel builder; no ambient setuptools dependency."""
import base64
import csv
import hashlib
import io
from pathlib import Path
import zipfile

DIST = 'localpkg-0.0.1.dist-info'


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    name = 'localpkg-0.0.1-py3-none-any.whl'
    files = {
        'localpkg/__init__.py': (Path(__file__).parent / 'localpkg/__init__.py').read_bytes(),
        f'{DIST}/METADATA': b'Metadata-Version: 2.1\nName: localpkg\nVersion: 0.0.1\n',
        f'{DIST}/WHEEL': b'Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n',
    }
    record = io.StringIO()
    writer = csv.writer(record, lineterminator='\n')
    for path, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
        writer.writerow((path, 'sha256=' + digest, len(data)))
    writer.writerow((f'{DIST}/RECORD', '', ''))
    files[f'{DIST}/RECORD'] = record.getvalue().encode()
    with zipfile.ZipFile(Path(wheel_directory) / name, 'w') as wheel:
        for path, data in files.items():
            wheel.writestr(path, data)
    return name
