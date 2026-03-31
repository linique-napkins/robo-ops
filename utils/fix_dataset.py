"""Fix duplicate frame data in jhimmens/linique-v2-fold-pickup dataset.

Episodes 300 and 314 have extra frames prepended (from another episode's data)
that aren't reflected in the videos or episode metadata. This causes
FrameTimestampError during training because the frame indices get shifted,
making video timestamp queries land beyond the actual video duration.

This script:
1. Loads the dataset from HuggingFace Hub
2. Removes the duplicate prefix rows from episodes 300 and 314
3. Recomputes the global `index` column and episode metadata boundaries
4. Pushes the corrected dataset back to the Hub

Usage:
    uv run utils/fix_dataset.py                    # dry run (default)
    uv run utils/fix_dataset.py --push             # fix and push to hub
    uv run utils/fix_dataset.py --repo-id NEW_ID   # push to a different repo
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json

import pyarrow as pa
import pyarrow.parquet as pq
import typer

app = typer.Typer()

REPO_ID = "jhimmens/linique-v2-fold-pickup"

# Episodes with duplicate prefix data and how many extra rows to remove
DUPLICATES = {
    300: 1448,  # first 1448 rows are from a different episode
    314: 1349,  # first 1349 rows are from a different episode
}


def find_duplicates(hf_dataset) -> dict[int, int]:
    """Verify and detect duplicate prefix rows by finding frame_index resets."""
    found = {}
    # Build episode start/end indices
    ep_ranges = {}
    prev_ep = None
    for idx in range(len(hf_dataset)):
        ep = int(hf_dataset[idx]["episode_index"])
        if ep != prev_ep:
            if prev_ep is not None:
                ep_ranges[prev_ep] = (ep_ranges[prev_ep][0], idx)
            ep_ranges[ep] = (idx, None)
            prev_ep = ep
    ep_ranges[prev_ep] = (ep_ranges[prev_ep][0], len(hf_dataset))

    for ep_idx, (start, end) in ep_ranges.items():
        prev_fi = -1
        for offset in range(end - start):
            fi = int(hf_dataset[start + offset]["frame_index"])
            if fi <= prev_fi:
                # frame_index reset — everything before this is duplicate
                found[ep_idx] = offset
                break
            prev_fi = fi

    return found


@app.command()
def main(
    push: bool = typer.Option(False, "--push", help="Push corrected dataset to Hub"),
    repo_id: str = typer.Option(REPO_ID, "--repo-id", help="Target repo ID for push"),
    verify_only: bool = typer.Option(False, "--verify-only", help="Only verify, don't fix"),
) -> None:
    """Fix duplicate frame data in the linique-v2-fold-pickup dataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    typer.echo(f"Loading dataset: {REPO_ID}")
    ds = LeRobotDataset(REPO_ID)

    typer.echo("Scanning for duplicate prefix rows...")
    found = find_duplicates(ds.hf_dataset)

    if not found:
        typer.echo("No duplicates found. Dataset is clean.")
        raise typer.Exit(0)

    typer.echo(f"Found {len(found)} episodes with duplicate prefixes:")
    total_extra = 0
    for ep_idx, extra in sorted(found.items()):
        typer.echo(f"  Episode {ep_idx}: {extra} extra rows to remove")
        total_extra += extra

    if verify_only:
        typer.echo(f"\nTotal extra rows: {total_extra}")
        typer.echo(f"Current total frames: {len(ds.hf_dataset)}")
        typer.echo(f"Corrected total frames: {len(ds.hf_dataset) - total_extra}")
        raise typer.Exit(0)

    # Build set of global indices to remove
    typer.echo("\nIdentifying rows to remove...")
    remove_indices = set()
    prev_ep = None
    ep_start = 0
    for idx in range(len(ds.hf_dataset)):
        ep = int(ds.hf_dataset[idx]["episode_index"])
        if ep != prev_ep:
            ep_start = idx
            prev_ep = ep
        if ep in found:
            offset = idx - ep_start
            if offset < found[ep]:
                remove_indices.add(idx)

    typer.echo(f"  Removing {len(remove_indices)} rows (expected {total_extra})")
    assert len(remove_indices) == total_extra

    # Filter the HF dataset
    typer.echo("Filtering dataset...")
    keep_indices = [i for i in range(len(ds.hf_dataset)) if i not in remove_indices]
    filtered = ds.hf_dataset.select(keep_indices)

    # Recompute the global `index` column (must be sequential 0..N-1)
    typer.echo("Recomputing global index column...")
    new_indices = list(range(len(filtered)))
    filtered = filtered.remove_columns(["index"]).add_column("index", new_indices)

    typer.echo(f"  New total frames: {len(filtered)}")

    # Write corrected parquet data back to disk
    root = ds.root
    data_dir = root / "data"

    # Read existing parquet files to understand chunking
    existing_files = sorted(data_dir.rglob("*.parquet"))
    typer.echo(f"  Found {len(existing_files)} existing parquet data files")

    # Write as a single chunk (LeRobot will re-chunk on push if needed)
    # First, remove old data files
    for f in existing_files:
        f.unlink()

    # Write new parquet file
    chunk_dir = data_dir / "chunk-000"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    out_path = chunk_dir / "train-00000-of-00001.parquet"

    # Convert HF dataset to arrow table and write
    table = filtered.data.table if hasattr(filtered.data, "table") else filtered.to_parquet(out_path)
    if hasattr(filtered.data, "table"):
        pq.write_table(table, out_path)
    typer.echo(f"  Wrote {out_path}")

    # Update episode metadata
    typer.echo("Updating episode metadata...")
    meta_dir = root / "meta"
    info_path = meta_dir / "info.json"

    with open(info_path) as f:
        info = json.load(f)

    old_total = info["total_frames"]
    info["total_frames"] = len(filtered)
    typer.echo(f"  total_frames: {old_total} -> {info['total_frames']}")

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    # Update episodes parquet — recompute dataset_from_index and dataset_to_index
    episodes_dir = meta_dir / "episodes"
    episodes_files = sorted(episodes_dir.rglob("*.parquet"))

    # Read all episodes metadata
    ep_tables = []
    for f in episodes_files:
        ep_tables.append(pq.read_table(f))
    episodes_table = pa.concat_tables(ep_tables)
    ep_df = episodes_table.to_pandas()

    # Recompute boundaries from the corrected parquet data
    # Scan the filtered dataset to find actual episode boundaries
    typer.echo("  Recomputing episode boundaries from corrected data...")
    ep_boundaries = {}
    for idx in range(len(filtered)):
        ep = int(filtered[idx]["episode_index"])
        if ep not in ep_boundaries:
            ep_boundaries[ep] = {"from": idx, "to": idx + 1}
        else:
            ep_boundaries[ep]["to"] = idx + 1

    # Update the episodes dataframe
    for ep_idx in range(len(ep_df)):
        old_from = ep_df.loc[ep_idx, "dataset_from_index"]
        old_to = ep_df.loc[ep_idx, "dataset_to_index"]
        new_from = ep_boundaries[ep_idx]["from"]
        new_to = ep_boundaries[ep_idx]["to"]
        new_len = new_to - new_from
        if old_from != new_from or old_to != new_to:
            typer.echo(
                f"  ep {ep_idx}: [{old_from},{old_to}) -> [{new_from},{new_to}) "
                f"(len {old_to - old_from} -> {new_len})"
            )
        ep_df.loc[ep_idx, "dataset_from_index"] = new_from
        ep_df.loc[ep_idx, "dataset_to_index"] = new_to

    # Write back episodes metadata
    for f in episodes_files:
        f.unlink()
    out_ep = episodes_dir / "chunk-000" / "episodes-00000-of-00001.parquet"
    out_ep.parent.mkdir(parents=True, exist_ok=True)
    ep_arrow = pa.Table.from_pandas(ep_df)
    pq.write_table(ep_arrow, out_ep)
    typer.echo(f"  Wrote {out_ep}")

    # Verify the fix
    typer.echo("\nVerifying corrected dataset...")
    ds_fixed = LeRobotDataset(REPO_ID, root=root)
    verify_found = find_duplicates(ds_fixed.hf_dataset)
    if verify_found:
        typer.echo(f"ERROR: Still found duplicates after fix: {verify_found}")
        raise typer.Exit(1)

    typer.echo(f"  Total episodes: {ds_fixed.num_episodes}")
    typer.echo(f"  Total frames:   {len(ds_fixed)}")
    typer.echo("  No duplicates found — fix verified!")

    if push:
        typer.echo(f"\nPushing corrected dataset to {repo_id}...")
        if repo_id != REPO_ID:
            # Update repo_id in the dataset metadata
            ds_fixed.meta.repo_id = repo_id
        ds_fixed.push_to_hub()
        typer.echo("Push complete!")
    else:
        typer.echo(f"\nDry run complete. To push, run with --push")
        typer.echo(f"  Local corrected dataset at: {root}")


if __name__ == "__main__":
    app()
