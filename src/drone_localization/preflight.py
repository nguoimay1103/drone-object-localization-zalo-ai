"""Check paths, checkpoint identity and evaluation coverage before loading a GPU."""
from .artifacts import sha256_file
from .evaluation import load_ground_truth


def preflight(config, shard_index=0, num_shards=1, video_ids=None):
    if type(num_shards) is not int or num_shards < 1 or type(shard_index) is not int or not 0 <= shard_index < num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    paths = config.paths
    if not paths.samples_dir.is_dir():
        raise FileNotFoundError(f"samples_dir not found: {paths.samples_dir}; edit YAML or use --samples-dir")
    all_ids = sorted(p.name for p in paths.samples_dir.iterdir() if p.is_dir())
    if not all_ids:
        raise ValueError("samples_dir contains no video folders")
    requested = all_ids if video_ids is None else sorted(video_ids)
    if len(set(requested)) != len(requested) or set(requested) - set(all_ids):
        raise ValueError("Requested video IDs must be unique and present in samples_dir")
    selected = requested[shard_index::num_shards]
    if not selected:
        raise ValueError("This shard has no videos")
    for video_id in selected:
        video = paths.samples_dir / video_id / "drone_video.mp4"
        if not video.is_file():
            raise FileNotFoundError(f"Video missing: {video}")
    hashes = {}
    for name, path, expected in (("yolo", paths.yolo_weights, config.checkpoints.yolo_sha256),
                                 ("siamese", paths.siamese_weights, config.checkpoints.siamese_sha256)):
        if not path.is_file():
            raise FileNotFoundError(f"{name} weights missing: {path}; no automatic download/fallback")
        actual = sha256_file(path)
        if expected is not None and actual != expected:
            raise ValueError(f"{name} checkpoint SHA-256 differs from the configured baseline")
        hashes[name] = actual
    gt_map = None
    gt_hash = None
    if paths.annotations is not None:
        gt_map = load_ground_truth(paths.annotations)
        if set(selected) - set(gt_map):
            raise ValueError(f"Missing GT records: {sorted(set(selected)-set(gt_map))}")
        gt_hash = sha256_file(paths.annotations)
    return {"all_video_ids": all_ids, "requested_video_ids": requested, "video_ids": selected,
            "shard_index": shard_index, "num_shards": num_shards,
            "checkpoint_hashes": hashes, "annotation_sha256": gt_hash}, gt_map
