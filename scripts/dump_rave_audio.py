#!/usr/bin/env python3
"""Dump RAVE validation audio from TensorBoard event files to WAV files.

Usage:
    python dump_rave_audio.py ./data/models/03_disanje --out ./audio_dump --every 5
"""
import argparse
import glob
import os

from tensorboard.backend.event_processing import event_accumulator as ea


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logdir", help="model folder; searched recursively for event files")
    p.add_argument("--out", default="audio_dump")
    p.add_argument("--every", type=int, default=1, help="keep every Nth audio event")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.logdir, "**", "events.out.tfevents.*"),
                             recursive=True))
    if not files:
        raise SystemExit(f"No event files found under {args.logdir}")

    for f in files:
        # size_guidance 0 = keep everything (default keeps only 4 audio events)
        acc = ea.EventAccumulator(f, size_guidance={ea.AUDIO: 0, ea.SCALARS: 0})
        acc.Reload()
        tags = acc.Tags()
        if not tags["audio"]:
            continue

        # Audio may be logged against a validation counter rather than the global
        # step, so map each clip to the nearest 'validation' scalar by wall time.
        scal = acc.Scalars("validation") if "validation" in tags["scalars"] else []

        run = os.path.relpath(os.path.dirname(f), args.logdir).replace(os.sep, "_")
        for tag in tags["audio"]:
            for i, ev in enumerate(acc.Audio(tag)):
                if i % args.every:
                    continue
                if scal:
                    nearest = min(scal, key=lambda s: abs(s.wall_time - ev.wall_time))
                    step = nearest.step
                else:
                    step = ev.step
                name = f"{run}_{tag}_step{step:08d}.wav"
                with open(os.path.join(args.out, name), "wb") as fh:
                    fh.write(ev.encoded_audio_string)
                print(name)


if __name__ == "__main__":
    main()