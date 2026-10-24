import os
import sys
import tempfile
from pathlib import Path

os.environ["AGRO_DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
