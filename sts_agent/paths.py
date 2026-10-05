"""Repository paths shared by runtime, training and data tools."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "agent.json"
DATA_ROOT = PROJECT_ROOT / "data"
