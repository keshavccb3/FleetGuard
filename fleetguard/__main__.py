import argparse
import json
from pathlib import Path
from .pipeline import run


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train and evaluate FleetGuard on NASA C-MAPSS.")
    parser.add_argument("--data-dir", type=Path, default=root/"data"/"raw")
    parser.add_argument("--config", type=Path, default=root/"configs"/"default.json")
    parser.add_argument("--results-dir", type=Path, default=root/"results")
    args = parser.parse_args()
    run(args.data_dir, args.results_dir, json.loads(args.config.read_text(encoding="utf-8")))


if __name__ == "__main__": main()
