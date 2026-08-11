import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


RGB_FEATURE_CANDIDATES = (
    "z_rgb_feature_input",
    "z_rgb_feature",
    "rgb_feature",
)
DEPTH_FEATURE_CANDIDATES = (
    "z_depth_feature_pred",
    "pred_z_depth_feature",
    "z_depth_feature_pred_model7_1",
    "z_depth_feature_gt",
    "z_depth_feature",
    "depth_feature",
)
ACTION_CANDIDATES = (
    "action_vector",
    "raw_actions",
    "action",
    "actions",
)
IMAGE_KEY_CANDIDATES = (
    "image",
    "image_path",
    "image_paths",
    "rgb_path",
    "rgb_paths",
    "file_name",
    "file_names",
    "id",
    "ids",
)
ID_KEY_CANDIDATES = IMAGE_KEY_CANDIDATES


def _sequence_len(value: object) -> int:
    if torch.is_tensor(value):
        return int(value.shape[0])
    if isinstance(value, (list, tuple)):
        return len(value)
    raise TypeError(f"Expected a tensor/list/tuple, got {type(value).__name__}.")


def load_manifest(manifest_path: Optional[Path]) -> Dict[str, object]:
    if manifest_path is None:
        return {}
    return json.loads(Path(manifest_path).read_text())


def _load_manifest_parts(manifest_path: Path, data_dir: Path) -> List[Path]:
    manifest = load_manifest(manifest_path)
    parts = []
    for part in manifest.get("parts", []):
        source_path = Path(part["path"])
        local_path = data_dir / source_path.name
        parts.append(local_path if local_path.exists() else source_path)
    return parts


def discover_part_files(data_dir: Path, manifest_path: Optional[Path] = None) -> List[Path]:
    if manifest_path is not None:
        return _load_manifest_parts(manifest_path, data_dir)
    return sorted(list(data_dir.glob("*_part*.pt")) + list(data_dir.glob("*_part*.pth")))


def _first_present_key(shard: Dict[str, object], candidates: Sequence[str]) -> Optional[str]:
    for key in candidates:
        if key in shard:
            return key
    return None


def _manifest_key(manifest: Dict[str, object], candidates: Sequence[str]) -> Optional[str]:
    for key in candidates:
        value = manifest.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def resolve_feature_keys(
    shard: Dict[str, object],
    manifest: Optional[Dict[str, object]] = None,
    rgb_feature_key: str = "auto",
    depth_feature_key: str = "auto",
    action_key: str = "auto",
    image_key: str = "auto",
) -> Tuple[str, str, str, Optional[str]]:
    manifest = manifest or {}

    if rgb_feature_key == "auto":
        rgb_feature_key = (
            _manifest_key(manifest, ("rgb_feature_key_input", "rgb_feature_key"))
            or _first_present_key(shard, RGB_FEATURE_CANDIDATES)
        )
    if depth_feature_key == "auto":
        depth_feature_key = (
            _manifest_key(manifest, ("feature_key", "feature_key_pred", "depth_feature_key"))
            or _first_present_key(shard, DEPTH_FEATURE_CANDIDATES)
        )
    if action_key == "auto":
        action_key = _manifest_key(manifest, ("action_key",)) or _first_present_key(shard, ACTION_CANDIDATES)
    if image_key == "auto":
        image_key = _manifest_key(manifest, ("image_key", "image_path_key")) or _first_present_key(shard, IMAGE_KEY_CANDIDATES)

    missing = []
    if not rgb_feature_key or rgb_feature_key not in shard:
        missing.append(("rgb_feature_key", rgb_feature_key, RGB_FEATURE_CANDIDATES))
    if not depth_feature_key or depth_feature_key not in shard:
        missing.append(("depth_feature_key", depth_feature_key, DEPTH_FEATURE_CANDIDATES))
    if not action_key or action_key not in shard:
        missing.append(("action_key", action_key, ACTION_CANDIDATES))
    if image_key not in (None, "") and image_key not in shard:
        image_key = None

    if missing:
        available = ", ".join(sorted(shard.keys()))
        details = "; ".join(
            f"{name}={value!r}, tried {list(candidates)}"
            for name, value, candidates in missing
        )
        raise KeyError(f"Could not resolve required shard keys: {details}. Available keys: {available}")

    return rgb_feature_key, depth_feature_key, action_key, image_key


def _value_at(value: object, index: int) -> object:
    if torch.is_tensor(value):
        return value[index]
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


class ShardFieldIndex:
    """Maps sample ids to fields stored across .pt shard files."""

    def __init__(
        self,
        part_files: Iterable[Path],
        value_key: str,
        id_key: str = "auto",
        manifest: Optional[Dict[str, object]] = None,
        preload: bool = False,
        label: str = "feature",
    ):
        self.part_files = [Path(p) for p in part_files]
        if not self.part_files:
            raise ValueError(f"No .pt part files were found for {label}.")
        self.manifest = manifest or {}
        self.value_key = value_key
        self.id_key = id_key
        self.preload = preload
        self.label = label
        self._cached_part_index = None
        self._cached_shard = None
        self._values = {}
        self._locations = {}
        seen_ids = set()

        first_shard = torch.load(self.part_files[0], map_location="cpu")
        if self.id_key == "auto":
            self.id_key = (
                _manifest_key(self.manifest, ("id_key", "image_key", "image_path_key"))
                or _first_present_key(first_shard, ID_KEY_CANDIDATES)
            )
        if self.id_key not in first_shard:
            raise KeyError(f"Could not resolve id key for {label}. Available keys: {sorted(first_shard.keys())}")
        if self.value_key not in first_shard:
            raise KeyError(
                f"{label} key {self.value_key!r} was not found. "
                f"Available keys: {sorted(first_shard.keys())}"
            )

        for part_index, part_file in enumerate(self.part_files):
            shard = torch.load(part_file, map_location="cpu")
            if self.id_key not in shard or self.value_key not in shard:
                raise KeyError(
                    f"{part_file} must contain {self.id_key!r} and {self.value_key!r}. "
                    f"Available keys: {sorted(shard.keys())}"
                )
            ids = shard[self.id_key]
            values = shard[self.value_key]
            if _sequence_len(ids) != _sequence_len(values):
                raise ValueError(
                    f"{part_file} has mismatched id/value lengths: "
                    f"{_sequence_len(ids)} vs {_sequence_len(values)}"
                )
            for local_index, sample_id in enumerate(ids):
                sample_id = str(sample_id)
                if sample_id in seen_ids:
                    continue
                seen_ids.add(sample_id)
                if self.preload:
                    self._values[sample_id] = _value_at(values, local_index)
                else:
                    self._locations[sample_id] = (part_index, local_index)

    def __contains__(self, sample_id: str) -> bool:
        sample_id = str(sample_id)
        return sample_id in self._values or sample_id in self._locations

    def __len__(self) -> int:
        return len(self._values) if self.preload else len(self._locations)

    def _load_shard(self, part_index: int) -> Dict[str, object]:
        if self._cached_part_index != part_index:
            self._cached_shard = torch.load(self.part_files[part_index], map_location="cpu")
            self._cached_part_index = part_index
        return self._cached_shard

    def get(self, sample_id: str) -> object:
        sample_id = str(sample_id)
        if self.preload:
            return self._values[sample_id]
        part_index, local_index = self._locations[sample_id]
        shard = self._load_shard(part_index)
        return _value_at(shard[self.value_key], local_index)


class JsonlActionIndex:
    """Maps sample ids to action vectors from a JSONL demonstration file."""

    def __init__(self, path: Path, id_key: str = "id", action_key: str = "auto"):
        self.path = Path(path)
        self.id_key = id_key
        self.action_key = action_key
        self._actions = {}
        self._resolved_action_keys = set()
        with self.path.open("r") as fin:
            for line in fin:
                if not line.strip():
                    continue
                item = json.loads(line)
                if self.id_key not in item:
                    continue
                key = self.action_key
                if key == "auto":
                    key = _first_present_key(item, ACTION_CANDIDATES)
                if key is None or key not in item:
                    continue
                self._resolved_action_keys.add(key)
                self._actions[str(item[self.id_key])] = torch.as_tensor(item[key], dtype=torch.float32)
        if not self._actions:
            raise ValueError(
                f"No actions were loaded from {self.path}. "
                f"Check --action_jsonl_id_key and --action_jsonl_key."
            )
        if self.action_key == "auto":
            self.action_key = ",".join(sorted(self._resolved_action_keys))

    def __contains__(self, sample_id: str) -> bool:
        return str(sample_id) in self._actions

    def __len__(self) -> int:
        return len(self._actions)

    def get(self, sample_id: str) -> torch.Tensor:
        return self._actions[str(sample_id)]


class LiberoDepthFusionDataset(Dataset):
    """Dataset for colleague-provided LIBERO feature shards.

    Each .pt shard must contain RGB features, depth features, and actions. Keys can
    be supplied explicitly or discovered from the manifest/shard with "auto".
    """

    def __init__(
        self,
        part_files: Iterable[Path],
        manifest: Optional[Dict[str, object]] = None,
        rgb_part_files: Optional[Iterable[Path]] = None,
        rgb_manifest: Optional[Dict[str, object]] = None,
        action_part_files: Optional[Iterable[Path]] = None,
        action_manifest: Optional[Dict[str, object]] = None,
        action_jsonl: Optional[Path] = None,
        action_jsonl_id_key: str = "id",
        action_jsonl_key: str = "auto",
        rgb_feature_key: str = "auto",
        depth_feature_key: str = "auto",
        action_key: str = "auto",
        image_key: str = "auto",
        rgb_id_key: str = "auto",
        action_id_key: str = "auto",
        preload: bool = True,
        return_metadata: bool = False,
    ):
        self.part_files = [Path(p) for p in part_files]
        if not self.part_files:
            raise ValueError("No .pt shard files were found.")

        first_shard = torch.load(self.part_files[0], map_location="cpu")
        self.manifest = manifest or {}
        self.image_key = image_key
        if self.image_key == "auto":
            self.image_key = (
                _manifest_key(self.manifest, ("id_key", "image_key", "image_path_key"))
                or _first_present_key(first_shard, ID_KEY_CANDIDATES)
            )
        if self.image_key not in first_shard:
            raise KeyError(f"Could not resolve primary id/image key. Available keys: {sorted(first_shard.keys())}")

        self.depth_feature_key = depth_feature_key
        if self.depth_feature_key == "auto":
            self.depth_feature_key = (
                _manifest_key(self.manifest, ("feature_key", "feature_key_pred", "depth_feature_key"))
                or _first_present_key(first_shard, DEPTH_FEATURE_CANDIDATES)
            )
        if self.depth_feature_key not in first_shard:
            raise KeyError(f"Could not resolve depth feature key. Available keys: {sorted(first_shard.keys())}")

        self.rgb_feature_key = rgb_feature_key
        if self.rgb_feature_key == "auto":
            self.rgb_feature_key = (
                _manifest_key(self.manifest, ("rgb_feature_key_input", "rgb_feature_key"))
                or _first_present_key(first_shard, RGB_FEATURE_CANDIDATES)
            )
        self.action_key = action_key
        if self.action_key == "auto":
            self.action_key = _manifest_key(self.manifest, ("action_key",)) or _first_present_key(first_shard, ACTION_CANDIDATES)

        self.preload = preload
        self.return_metadata = return_metadata
        self.rgb_index = None
        self.action_index = None

        if self.rgb_feature_key not in first_shard:
            if rgb_part_files is None:
                raise KeyError(
                    f"Primary shards do not contain RGB key {self.rgb_feature_key!r}. "
                    "Pass --rgb_data_dir/--rgb_manifest to join RGB features by id."
                )
            rgb_first = torch.load(list(rgb_part_files)[0], map_location="cpu")
            if self.rgb_feature_key == "auto" or self.rgb_feature_key not in rgb_first:
                self.rgb_feature_key = (
                    _manifest_key(rgb_manifest or {}, ("rgb_feature_key_input", "rgb_feature_key"))
                    or _first_present_key(rgb_first, RGB_FEATURE_CANDIDATES)
                )
            self.rgb_index = ShardFieldIndex(
                rgb_part_files,
                value_key=self.rgb_feature_key,
                id_key=rgb_id_key,
                manifest=rgb_manifest,
                preload=preload,
                label="RGB feature",
            )

        if not self.action_key or self.action_key not in first_shard:
            if action_part_files is not None:
                action_first = torch.load(list(action_part_files)[0], map_location="cpu")
                if not self.action_key or self.action_key == "auto" or self.action_key not in action_first:
                    self.action_key = (
                        _manifest_key(action_manifest or {}, ("action_key",))
                        or _first_present_key(action_first, ACTION_CANDIDATES)
                    )
                self.action_index = ShardFieldIndex(
                    action_part_files,
                    value_key=self.action_key,
                    id_key=action_id_key,
                    manifest=action_manifest,
                    preload=True,
                    label="action",
                )
            elif action_jsonl is not None:
                self.action_index = JsonlActionIndex(
                    action_jsonl,
                    id_key=action_jsonl_id_key,
                    action_key=action_jsonl_key,
                )
                self.action_key = self.action_index.action_key
            else:
                raise KeyError(
                    "Primary shards do not contain actions. Pass --action_data_dir/--action_manifest "
                    "or --action_jsonl to join action labels by id."
                )

        self._lengths = []
        rgb_features = []
        depth_features = []
        actions = []
        image_names = []
        self._ids = []
        missing_rgb = 0
        missing_action = 0
        for part_file in self.part_files:
            shard = torch.load(part_file, map_location="cpu")
            self._validate_keys(shard, part_file)
            ids = shard[self.image_key]
            shard_kept = 0
            for local_index, sample_id in enumerate(ids):
                sample_id = str(sample_id)
                has_rgb = self.rgb_feature_key in shard or (self.rgb_index is not None and sample_id in self.rgb_index)
                has_action = self.action_key in shard or (self.action_index is not None and sample_id in self.action_index)
                if not has_rgb:
                    missing_rgb += 1
                    continue
                if not has_action:
                    missing_action += 1
                    continue
                self._ids.append((len(self._lengths), local_index, sample_id))
                shard_kept += 1
            self._lengths.append(shard_kept)
            if self.preload:
                for _, local_index, sample_id in self._ids[-shard_kept:]:
                    if self.rgb_feature_key in shard:
                        rgb_features.append(_value_at(shard[self.rgb_feature_key], local_index).float().clone())
                    else:
                        rgb_features.append(self.rgb_index.get(sample_id).float().clone())
                    depth_features.append(_value_at(shard[self.depth_feature_key], local_index).float().clone())
                    if self.action_key in shard:
                        actions.append(_value_at(shard[self.action_key], local_index).float().clone())
                    else:
                        actions.append(self.action_index.get(sample_id).float().clone())
                    if self.return_metadata:
                        image_names.append(sample_id)

        if missing_rgb or missing_action:
            print(
                json.dumps(
                    {
                        "dropped_samples": {
                            "missing_rgb": missing_rgb,
                            "missing_action": missing_action,
                        }
                    }
                )
            )
        if not self._ids:
            raise ValueError("No trainable samples remain after joining depth/RGB/action by id.")

        self._cached_part_index = None
        self._cached_shard = None
        self._preloaded = None
        if self.preload:
            self._preloaded = {
                "rgb_feature": torch.stack(rgb_features, dim=0),
                "depth_feature": torch.stack(depth_features, dim=0),
                "action": torch.stack(actions, dim=0),
            }
            if self.return_metadata and self.image_key is not None:
                self._preloaded["image_name"] = image_names

    def _validate_keys(self, shard: Dict[str, object], part_file: Path) -> None:
        missing = [
            key
            for key in (self.depth_feature_key, self.image_key)
            if key not in shard
        ]
        if self.rgb_index is None and self.rgb_feature_key not in shard:
            missing.append(self.rgb_feature_key)
        if self.action_index is None and self.action_key not in shard:
            missing.append(self.action_key)
        if missing:
            raise KeyError(f"{part_file} is missing required key(s): {missing}")

    def __len__(self) -> int:
        return len(self._ids)

    def _load_shard(self, part_index: int) -> Dict[str, object]:
        if self._cached_part_index != part_index:
            self._cached_shard = torch.load(self.part_files[part_index], map_location="cpu")
            self._cached_part_index = part_index
        return self._cached_shard

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        if self._preloaded is not None:
            item = {
                "rgb_feature": self._preloaded["rgb_feature"][index],
                "depth_feature": self._preloaded["depth_feature"][index],
                "action": self._preloaded["action"][index],
            }
            if self.return_metadata and "image_name" in self._preloaded:
                item["image_name"] = self._preloaded["image_name"][index]
            return item

        part_index, local_index, sample_id = self._ids[index]
        shard = self._load_shard(part_index)

        item = {
            "rgb_feature": (
                shard[self.rgb_feature_key][local_index]
                if self.rgb_feature_key in shard
                else self.rgb_index.get(sample_id)
            ).float(),
            "depth_feature": shard[self.depth_feature_key][local_index].float(),
            "action": (
                shard[self.action_key][local_index]
                if self.action_key in shard
                else self.action_index.get(sample_id)
            ).float(),
        }
        if self.return_metadata:
            item["image_name"] = sample_id
        return item
