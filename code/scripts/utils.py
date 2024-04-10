import sys
from pathlib import Path


def fix_path():
    # Nasty hack to add src directory
    sys.path.insert(1, str(Path(__file__).parents[1] / "src"))
    # I am sorry


fix_path()  # noqa
