"""Evaluate an existing baseline prediction JSON against manual annotations."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_localization.artifacts import write_json
from drone_localization.evaluation import evaluate_predictions, load_ground_truth


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local historical ST-IoU, frame origin 0; not an official evaluator")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--video-id", action="append", help="Explicit evaluation subset; default is all GT records")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        gt = load_ground_truth(args.annotations)
        with args.predictions.open(encoding="utf-8") as handle:
            predictions = json.load(handle)
        result = evaluate_predictions(predictions, gt, args.video_id if args.video_id is not None else list(gt))
        if args.output:
            if args.output.exists():
                raise FileExistsError(f"Refusing to overwrite: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
