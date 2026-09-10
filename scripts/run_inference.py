"""Run from any working directory; model imports happen after preflight."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_localization.config import load_config
from drone_localization.preflight import preflight


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline production pipeline; paths are configurable, no silent weight downloads.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/production_0_740013.yaml")
    for name in ("samples-dir", "annotations", "yolo-weights", "siamese-weights", "output-dir"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--device", help="auto, cpu, cuda:0 or cuda:1")
    parser.add_argument("--video-id", action="append", help="Optional video subset; repeat for multiple IDs")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--no-eval", action="store_true", help="Skip GT and report evaluation=null")
    parser.add_argument("--dry-run", action="store_true", help="Check paths, checkpoint hashes and GT without importing torch")
    args = parser.parse_args(argv)
    try:
        overrides = {name: getattr(args, name) for name in ("samples_dir", "annotations", "yolo_weights", "siamese_weights", "output_dir")}
        config = load_config(args.config, overrides, args.device, args.no_eval)
        selection, gt_map = preflight(config, args.shard_index, args.num_shards, args.video_id)
        if args.dry_run:
            print(json.dumps({"config": config.to_dict(), "selection": selection,
                              "evaluation_enabled": gt_map is not None,
                              "gpu_or_video_decode_tested": False}, indent=2))
            return 0
        from drone_localization.pipelines.inference import run
        run(config, selection, gt_map)
        return 0
    except (ValueError, TypeError, KeyError, OSError, ImportError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
