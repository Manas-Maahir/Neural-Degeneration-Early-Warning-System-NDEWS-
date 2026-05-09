import sys
from pathlib import Path
print("Path debug starting...")
root = Path(__file__).resolve().parent.parent
print(f"Computed root: {root}")
sys.path.insert(0, str(root))
try:
    from src.dataset import get_cifar_loaders
    print("Import successful!")
except Exception as e:
    print(f"Error: {e}")
