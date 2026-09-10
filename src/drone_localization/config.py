"""Validated YAML configuration; no model imports or filesystem mutations."""
from dataclasses import asdict, dataclass, field, replace
import math
import os
from pathlib import Path
import re


@dataclass(frozen=True)
class Paths:
    samples_dir: Path
    yolo_weights: Path
    siamese_weights: Path
    output_dir: Path
    annotations: Path | None = None


@dataclass(frozen=True)
class Inference:
    confidence_threshold: float = 0.05
    matching_threshold: float = 0.45
    yolo_only_threshold: float = 0.2
    weight_yolo: float = 0.4
    weight_siamese: float = 0.3
    weight_color: float = 0.3
    use_tta: bool = True
    use_multiscale_ref: bool = True
    ref_scales: tuple[int, ...] = (224, 112, 56)
    temporal_max_gap: int = 5
    temporal_min_seg: int = 3
    temporal_max_center_speed: float | None = None
    temporal_max_log_scale_speed: float | None = None
    device: str = "auto"
    imgsz: int | None = None
    batch: int | None = None

    def __post_init__(self):
        for name in ("confidence_threshold", "matching_threshold", "yolo_only_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        for name in ("weight_yolo", "weight_siamese", "weight_color"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if self.weight_yolo <= 0:
            raise ValueError("weight_yolo must be positive to support missing-reference fallback")
        for name in ("use_tta", "use_multiscale_ref"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a YAML boolean")
        if not isinstance(self.ref_scales, (list, tuple)) or not self.ref_scales or any(type(s) is not int or s <= 0 for s in self.ref_scales):
            raise ValueError("ref_scales must contain positive integers")
        object.__setattr__(self, "ref_scales", tuple(self.ref_scales))
        for name, minimum in (("temporal_max_gap", 0), ("temporal_min_seg", 1)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("temporal_max_center_speed", "temporal_max_log_scale_speed"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be null or a finite positive number")
        for name in ("imgsz", "batch"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be null or a positive integer")
        if not isinstance(self.device, str) or not re.fullmatch(r"auto|cpu|cuda(?::\d+)?", self.device):
            raise ValueError("device must be auto, cpu, cuda or cuda:N")


@dataclass(frozen=True)
class Checkpoints:
    yolo_sha256: str | None = None
    siamese_sha256: str | None = None

    def __post_init__(self):
        for value in (self.yolo_sha256, self.siamese_sha256):
            if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)):
                raise ValueError("Checkpoint hashes must be null or lowercase SHA-256")


@dataclass(frozen=True)
class Objective:
    priority: str = "offline_quality"
    target_fps: float = 25.0
    fps_is_hard_constraint: bool = False

    def __post_init__(self):
        if self.priority != "offline_quality":
            raise ValueError("This runner supports the offline_quality objective")
        if isinstance(self.target_fps, bool) or not isinstance(self.target_fps, (int, float)) or not math.isfinite(self.target_fps) or self.target_fps <= 0:
            raise ValueError("target_fps must be finite and positive")
        if type(self.fps_is_hard_constraint) is not bool:
            raise ValueError("fps_is_hard_constraint must be a boolean")


@dataclass(frozen=True)
class Config:
    paths: Paths
    inference: Inference = field(default_factory=Inference)
    checkpoints: Checkpoints = field(default_factory=Checkpoints)
    objective: Objective = field(default_factory=Objective)

    def to_dict(self):
        result = asdict(self)
        result["paths"] = {k: str(v) if v is not None else None for k, v in result["paths"].items()}
        return result


def load_config(path, path_overrides=None, device=None, no_eval=False):
    import yaml

    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "paths" not in data:
        raise ValueError("Config must be a mapping containing paths")
    if set(data) - {"paths", "inference", "checkpoints", "objective"}:
        raise ValueError(f"Unknown config sections: {set(data) - {'paths', 'inference', 'checkpoints', 'objective'}}")
    raw_paths = dict(data["paths"])
    overrides = {k: v for k, v in (path_overrides or {}).items() if v is not None}
    raw_paths.update(overrides)
    if no_eval:
        raw_paths["annotations"] = None
    resolved = {}
    for key, value in raw_paths.items():
        if value is None and key == "annotations":
            resolved[key] = None
            continue
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"paths.{key} must be a nonempty path")
        p = Path(os.path.expandvars(str(value))).expanduser()
        base = Path.cwd() if key in overrides else config_path.parent
        resolved[key] = (base / p).resolve() if not p.is_absolute() else p.resolve()
    inference = Inference(**data.get("inference", {}))
    if device is not None:
        inference = replace(inference, device=device)
    return Config(Paths(**resolved), inference, Checkpoints(**data.get("checkpoints", {})), Objective(**data.get("objective", {})))
