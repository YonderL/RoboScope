"""Extract Spatial from the pinned, 256px LeRobot LIBERO release without resizing."""

import argparse
import json
from pathlib import Path

DATASET_ID = "HuggingFaceVLA/libero"
DATASET_REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"
# This release has incorrect data/file_index metadata (55..68 instead of 308..376).
# Pin the verified physical shards, then verify every episode against its contents.
SPATIAL_FILES = [f"data/chunk-000/file-{i:03d}.parquet" for i in range(308, 377)]


def spatial_episodes(episodes, task_names):
    """Match complete instructions, not task indices or a substring shared by suites."""
    selected, found = [], set()
    for episode in episodes:
        tasks = list(episode["tasks"])
        if len(tasks) == 1 and tasks[0] in task_names:
            selected.append(int(episode["episode_index"]))
            found.add(tasks[0])
    if len(task_names) != 10 or found != set(task_names):
        raise ValueError(f"Spatial task coverage mismatch: missing={set(task_names) - found}")
    return sorted(selected)


def validate_features(info):
    features = info["features"]
    for key in ("observation.images.image", "observation.images.image2"):
        if features[key]["shape"] != [256, 256, 3] or features[key]["dtype"] != "image":
            raise ValueError("Expected the official embedded 256x256 RGB images")
    if features["observation.state"]["shape"] != [8] or features["action"]["shape"] != [7]:
        raise ValueError("Expected 8D end-effector state and 7D OSC action")


def _episode_locations(source, selected):
    """Map each Spatial episode to the parquet shard that actually contains it.

    The pinned release stores wrong ``data/file_index`` values (55..68). Scanning
    the physical shards is the source of truth; an episode must not cross files.
    """
    import pyarrow.parquet as pq

    wanted = set(selected)
    locations = {}
    for filename in SPATIAL_FILES:
        path = source / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned shard {path}")
        ids = pq.read_table(path, columns=["episode_index"]).column(0).to_pylist()
        for episode in set(ids):
            if episode not in wanted:
                continue
            if episode in locations:
                raise ValueError(f"Episode {episode} spans files; extraction must not silently truncate it")
            locations[episode] = path
    missing = wanted - set(locations)
    if missing:
        raise ValueError(f"Missing source episodes: {sorted(missing)[:8]}")
    return locations


def _episode_table(table, positions, episodes, old, new_index, task_index, global_index, features):
    import numpy as np
    import pyarrow as pa

    expected = episodes[old]
    if positions != list(range(positions[0], positions[-1] + 1)):
        raise ValueError(f"Episode {old} is not stored contiguously")
    ep = table.take(pa.array(positions))
    if len(ep) != expected["length"] or not np.array_equal(np.asarray(ep["frame_index"]), np.arange(len(ep))):
        raise ValueError(f"Incomplete or unordered episode {old}")
    stats = {}
    replacements = {
        "episode_index": np.full(len(ep), new_index, dtype=np.int64),
        "task_index": np.full(len(ep), task_index, dtype=np.int64),
        "index": np.arange(global_index, global_index + len(ep), dtype=np.int64),
    }
    for key, feature in features.items():
        if feature["dtype"] == "image":
            stats[key] = {
                stat: np.asarray(expected[f"stats/{key}/{stat}"], dtype=np.float64)
                for stat in ("min", "max", "mean", "std", "count")
            }
            continue
        values = replacements[key] if key in replacements else np.asarray(ep[key].to_pylist())
        if values.ndim == 1:
            values = values[:, None]
        values = values.astype(np.float64)
        stats[key] = {
            "min": values.min(0),
            "max": values.max(0),
            "mean": values.mean(0),
            "std": values.std(0),
            "count": np.array([len(ep)], dtype=np.float64),
        }
        if key in ("observation.state", "action"):
            for stat in ("min", "max", "mean", "std"):
                if not np.allclose(stats[key][stat], expected[f"stats/{key}/{stat}"], rtol=1e-4, atol=1e-5):
                    raise ValueError(f"Source statistics mismatch: episode={old}, {key}/{stat}")
    for key, values in replacements.items():
        ep = ep.set_column(ep.schema.get_field_index(key), key, pa.array(values, type=pa.int64()))
    return ep, stats


def extract_spatial(source, destination, info, episodes, selected, task_names):
    """Copy Arrow image bytes unchanged and write episodes in ascending index order.

    LeRobot assigns ``dataset_from_index`` from save order, so parquet rows must
    follow that same order. Numeric statistics are recomputed; image statistics
    come from the verified source episodes. No rendering, JPEG or resizing occurs.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    locations = _episode_locations(source, selected)
    meta = LeRobotDatasetMetadata.create(
        repo_id="HuggingFaceVLA/libero_spatial",
        fps=info["fps"],
        features=info["features"],
        robot_type=info["robot_type"],
        root=destination,
        use_videos=False,
        chunks_size=info["chunks_size"],
        data_files_size_in_mb=info["data_files_size_in_mb"],
        video_files_size_in_mb=info["video_files_size_in_mb"],
    )
    meta.save_episode_tasks(sorted(task_names))
    tasks = pq.read_table(source / "meta/tasks.parquet").to_pandas()
    task_by_id = {int(row.task_index): name for name, row in tasks.iterrows()}
    limit = int(info["data_files_size_in_mb"]) * 1024 * 1024
    file_index = global_index = buffered = 0
    tables, records, cached_path, cached = [], [], None, None

    def flush():
        nonlocal file_index, tables, records, buffered
        if not tables:
            return
        target = destination / f"data/chunk-000/file-{file_index:03d}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.concat_tables(tables), target, compression="snappy")
        for new_index, length, episode_tasks, stats in records:
            # dataset_from/to are assigned here from this save order.
            meta.save_episode(
                new_index,
                length,
                episode_tasks,
                stats,
                {
                    "data/chunk_index": 0,
                    "data/file_index": file_index,
                },
            )
        print(f"Wrote shard {file_index} ({len(records)} episodes)", flush=True)
        file_index += 1
        tables, records, buffered = [], [], 0

    for new_index, old in enumerate(selected):
        path = locations[old]
        if path != cached_path:
            cached = pq.read_table(path)
            cached_path = path
        ids = cached.column("episode_index").to_pylist()
        positions = [i for i, episode in enumerate(ids) if episode == old]
        expected = episodes[old]
        source_task = task_by_id[int(cached.column("task_index")[positions[0]].as_py())]
        if [source_task] != expected["tasks"] or len(
            {cached.column("task_index")[i].as_py() for i in positions}
        ) != 1:
            raise ValueError(f"Task/episode mismatch: {old}")
        task_index = meta.get_task_index(expected["tasks"][0])
        if task_index is None:
            raise ValueError(f"Task was not registered: {expected['tasks'][0]}")
        ep, stats = _episode_table(
            cached,
            positions,
            episodes,
            old,
            new_index,
            task_index,
            global_index,
            info["features"],
        )
        tables.append(ep)
        records.append((new_index, len(ep), expected["tasks"], stats))
        buffered += ep.nbytes
        global_index += len(ep)
        if buffered >= limit:
            flush()
        if (new_index + 1) % 50 == 0:
            print(f"Validated {new_index + 1}/{len(selected)} episodes ({global_index} frames)", flush=True)
    flush()
    meta.finalize()


def prepare(source, destination, libero_root, revision=DATASET_REVISION):
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if revision != DATASET_REVISION:
        raise ValueError("Physical shard mapping must be re-verified for a different source revision")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError("Source and Spatial subset must use different directories")
    task_names = {
        p.stem.replace("_", " ") for p in (Path(libero_root) / "bddl_files" / "libero_spatial").glob("*.bddl")
    }
    snapshot_download(
        DATASET_ID,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["meta/*"],
        local_dir=source,
        max_workers=4,
    )
    info = json.loads((source / "meta/info.json").read_text())
    validate_features(info)
    rows = pq.read_table(source / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    episodes = {int(row["episode_index"]): row for row in rows}
    selected = spatial_episodes(rows, task_names)
    provenance = {
        "source_repo": DATASET_ID,
        "source_revision": revision,
        "source_files": SPATIAL_FILES,
        "source_episode_indices": selected,
        "tasks": sorted(task_names),
        "state": "eef_position(3), eef_axis_angle(3), gripper_qpos(2)",
        "images": "Original embedded 256x256 RGB bytes; no resize or flip during extraction",
        "statistics": "Numeric stats recomputed; Spatial episode image stats aggregated with LeRobot",
        "index_repair": "Source metadata points to wrong shards; rebuilt from validated frame contents",
    }
    receipt = destination / "spatial_provenance.json"
    if receipt.exists():
        if json.loads(receipt.read_text()) != provenance:
            raise ValueError("Existing Spatial subset has different provenance")
        return destination
    staging = destination.with_name(destination.name + ".building")
    if destination.exists() or staging.exists():
        raise FileExistsError(f"Incomplete/unverified subset: {destination} or {staging}")
    print(f"Spatial: {len(selected)} episodes, {len(SPATIAL_FILES)} source parquet files", flush=True)
    snapshot_download(
        DATASET_ID,
        repo_type="dataset",
        revision=revision,
        allow_patterns=SPATIAL_FILES,
        local_dir=source,
        max_workers=4,
    )
    extract_spatial(source, staging, info, episodes, selected, task_names)
    subset = LeRobotDataset("HuggingFaceVLA/libero_spatial", root=staging)
    if subset.meta.total_tasks != 10 or subset.num_episodes != len(selected):
        raise RuntimeError("Unexpected task or episode count after subset extraction")
    sample = subset[0]
    for key in ("observation.images.image", "observation.images.image2"):
        if tuple(sample[key].shape) != (3, 256, 256):
            raise RuntimeError("Decoded image shape differs from the official release")
    (staging / receipt.name).write_text(json.dumps(provenance, indent=2) + "\n")
    staging.rename(destination)
    print(f"Prepared {subset.num_episodes} episodes / {subset.num_frames} frames: {destination}", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.destination, args.libero_root)


if __name__ == "__main__":
    main()
