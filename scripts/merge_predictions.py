"""Merge separate GPU runs only after verifying provenance and coverage."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_localization.config import load_config
from drone_localization.merge import merge_runs
from drone_localization.preflight import preflight


def main(argv=None):
    parser = argparse.ArgumentParser(description="Merge disjoint inference runs and recompute mean ST-IoU")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/production_0_740013.yaml")
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-dir", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--yolo-weights", type=Path)
    parser.add_argument("--siamese-weights", type=Path)
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, {k: getattr(args, k) for k in ("samples_dir", "annotations", "yolo_weights", "siamese_weights", "output_dir")}, no_eval=args.no_eval)
        selection, gt = preflight(config)
        summary = merge_runs(args.runs, config.paths.output_dir, selection["video_ids"], gt,
                             selection["checkpoint_hashes"], selection["annotation_sha256"])
        print(json.dumps(summary, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
