"""Repository paths shared by runtime, training and knowledge modules."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "agent.json"
DATA_ROOT = PROJECT_ROOT / "data"
KNOWLEDGE_ROOT = DATA_ROOT / "knowledge"

