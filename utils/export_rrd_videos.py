"""
Export camera videos from a Rerun .rrd recording file.

Single-pass extraction of all camera streams, encoded via ffmpeg.

Usage:
    uv run utils/export_rrd_videos.py <rrd_file>
    uv run utils/export_rrd_videos.py data/recordings/inference_20260330_165735.rrd
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import rerun as rr
import typer

app = typer.Typer()


def discover_cameras(rec: rr.dataframe.Recording) -> list[str]:
    """Find all camera entity paths in the recording."""
    cameras = []
    for col in rec.schema().component_columns():
        entity = col.entity_path
        if entity.startswith("/cameras/") and "Image:buffer" in str(col):
            name = entity.replace("/cameras/", "")
            if name not in cameras:
                cameras.append(name)
    return cameras


@app.command()
def main(
    rrd_file: Path = typer.Argument(..., help="Path to .rrd recording file"),
    output_dir: Path = typer.Option(
        None, "--output", "-o", help="Output directory (default: next to rrd)"
    ),
    fps: int = typer.Option(30, "--fps", help="Video frame rate"),
) -> None:
    """Export camera videos from a Rerun .rrd recording."""
    if not rrd_file.exists():
        typer.echo(f"File not found: {rrd_file}")
        raise typer.Exit(1)

    if output_dir is None:
        output_dir = rrd_file.parent / rrd_file.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Loading: {rrd_file}")
    rec = rr.dataframe.load_recording(str(rrd_file))

    cameras = discover_cameras(rec)
    if not cameras:
        typer.echo("No camera streams found in recording.")
        raise typer.Exit(1)

    typer.echo(f"Found {len(cameras)} cameras: {', '.join(cameras)}")

    # Single-pass: read all cameras at once
    view = rec.view(index="step", contents="/cameras/**")
    reader = view.select()

    # Get image dimensions from first frame
    first_batch = reader.read_next_batch()
    fmt_col_name = None
    for name in first_batch.schema.names:
        if "Image:format" in name:
            fmt_col_name = name
            break
    fmt = first_batch.column(fmt_col_name)[0].as_py()[0]
    w, h = fmt["width"], fmt["height"]
    typer.echo(f"Resolution: {w}x{h} @ {fps}fps")

    # Start one ffmpeg process per camera, piping raw RGB frames in
    procs: dict[str, subprocess.Popen] = {}
    for cam in cameras:
        out_path = output_dir / f"{cam}.mp4"
        procs[cam] = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", f"{w}x{h}",
                "-r", str(fps),
                "-i", "pipe:0",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                str(out_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # Stream frames through in a single pass
    counts: dict[str, int] = {cam: 0 for cam in cameras}

    def process_batch(batch) -> None:
        for cam in cameras:
            entity = f"/cameras/{cam}"
            buf_col = batch.column(f"{entity}:Image:buffer")
            for i in range(len(batch)):
                raw = buf_col[i].as_py()
                if raw is None:
                    continue
                frame_bytes = bytes(raw[0])
                procs[cam].stdin.write(frame_bytes)
                counts[cam] += 1

    # Process first batch (already read)
    process_batch(first_batch)

    # Process remaining batches
    frame_num = 1
    try:
        while True:
            batch = reader.read_next_batch()
            process_batch(batch)
            frame_num += 1
            if frame_num % 200 == 0:
                typer.echo(f"  {frame_num} frames processed...")
    except StopIteration:
        pass

    # Close ffmpeg pipes and wait
    for cam, proc in procs.items():
        proc.stdin.close()
        proc.wait()
        out_path = output_dir / f"{cam}.mp4"
        size_mb = out_path.stat().st_size / 1024 / 1024
        typer.echo(f"  {cam}: {counts[cam]} frames -> {out_path} ({size_mb:.1f}MB)")

    typer.echo(f"\nDone. Videos saved to: {output_dir}")


if __name__ == "__main__":
    app()
