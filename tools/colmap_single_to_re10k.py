"""
Convert a single-camera COLMAP sequence (cameras.json + sparse/txt/images.txt,
as produced for the self-collected surround-view dataset, see DrivingForward's
dataset/colmap_util.py) into the RE10K-style `.torch` chunk format expected by
VolSplat's `DatasetRE10k` dataloader.

This treats ONE camera's frames as a monocular moving sequence (exactly like
RE10K): pick N context frames + some target frames for novel-view synthesis.

Outputs (under <out_dir>/test/):
    - 000000.torch    : [ {key, images, cameras} ]
    - index.json      : {scene_key: "000000.torch"}
And:
    - <eval_index>.json : {scene_key: {"context": [...], "target": [...]}}

The 18-dim `cameras` row layout required by DatasetRE10k.convert_poses:
    [0:4]  = fx, fy, cx, cy   (NORMALIZED by image width/height)
    [4:6]  = placeholder (near/far; unused, set 0)
    [6:18] = world-to-camera (w2c) 3x4 matrix, row-major flattened (OpenCV conv.)

COLMAP images.txt stores (Q,T) as world->cam directly, which is exactly what
VolSplat's convert_poses expects in [6:18] (it will .inverse() internally to get
c2w). So we DO NOT invert the pose here.

Usage (run inside the `volsplat` conda env):
    python tools/colmap_single_to_re10k.py \
        --root colmap_test_full \
        --camera mid_center_top_wide \
        --out_dir datasets/mycolmap \
        --scene_key mid_wide \
        --num_context 6 \
        --num_target 8
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def qvec2rotmat(qvec):
    """Quaternion (w, x, y, z) -> 3x3 rotation matrix (COLMAP convention)."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


def read_cameras_json(path, camera_name):
    """Return (width, height, fx, fy, cx, cy, source_model) for one camera."""
    with open(path, "r") as f:
        data = json.load(f)
    for entry in data:
        if entry["camera"] == camera_name:
            fx, fy, cx, cy = entry["params"][:4]
            return (int(entry["width"]), int(entry["height"]),
                    float(fx), float(fy), float(cx), float(cy),
                    entry.get("source_model", "unknown"))
    raise KeyError(f"Camera '{camera_name}' not found in {path}. "
                   f"Available: {[e['camera'] for e in data]}")


def read_images_poses(path, camera_name):
    """Parse images.txt, return list of (frame_name, w2c_3x4) for one camera,
    sorted by the leading frame index. Poses are world->cam (COLMAP native)."""
    prefix = camera_name + "/"
    entries = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            name = parts[-1]
            if not (name.endswith(".jpg") or name.endswith(".png")):
                continue
            if not name.startswith(prefix):
                continue
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            R = qvec2rotmat([qw, qx, qy, qz])          # world->cam
            t = np.array([tx, ty, tz], dtype=np.float64)
            w2c = np.concatenate([R, t.reshape(3, 1)], axis=1)  # 3x4
            frame = name[len(prefix):]                 # e.g. '0002_xxx.jpg'
            entries.append((frame, w2c))
    # sort by leading zero-padded index
    entries.sort(key=lambda e: e[0])
    return entries


def pick_indices(n, num_context, num_target, context_stride=None, context_start=None):
    """Select context + target frames for novel-view synthesis.

    Context frames are picked with a SMALL, controllable baseline (not spread
    across the whole sequence) so neighbouring context views overlap enough for
    feed-forward reconstruction. The context window spans
    (num_context-1)*stride frames; target frames are interleaved in the gaps
    between consecutive context frames.

    Args:
        n              : number of usable frames.
        num_context    : how many context views.
        num_target     : how many target (held-out) views.
        context_stride : frame gap between adjacent context views. If None,
                         auto = max(1, (n-1)//(num_context-1)//4) capped so the
                         window fits, giving a compact baseline.
        context_start  : first context frame index. If None, the context window
                         is centred in the sequence.
    """
    if num_context > n:
        num_context = n
    if num_context <= 1:
        return list(range(min(n, max(1, num_context)))), []

    # Auto stride: compact baseline. Use a quarter of the "even spread" step,
    # at least 1, and never so large the window overflows the sequence.
    if context_stride is None or context_stride <= 0:
        even_step = max(1, (n - 1) // (num_context - 1))
        context_stride = max(1, even_step // 4)
    # Ensure the whole context window fits in the sequence.
    max_stride = (n - 1) // (num_context - 1)
    context_stride = max(1, min(context_stride, max_stride))

    span = context_stride * (num_context - 1)  # last - first

    # Centre the context window by default.
    if context_start is None:
        context_start = (n - 1 - span) // 2
    context_start = max(0, min(context_start, n - 1 - span))

    ctx = [context_start + i * context_stride for i in range(num_context)]
    ctx = sorted(set(ctx))

    ctx_set = set(ctx)
    lo, hi = ctx[0], ctx[-1]
    candidates = [i for i in range(lo + 1, hi) if i not in ctx_set]
    if not candidates:
        return ctx, []
    if num_target >= len(candidates):
        tgt = candidates
    else:
        sel = np.linspace(0, len(candidates) - 1, num_target).round().astype(int)
        tgt = sorted(set(candidates[i] for i in sel))
    return ctx, tgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Dataset root containing cameras.json, sparse/txt/images.txt, <camera>/*.jpg")
    ap.add_argument("--camera", default="mid_center_top_wide")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scene_key", default="mid_wide")
    ap.add_argument("--num_context", type=int, default=6)
    ap.add_argument("--num_target", type=int, default=8)
    ap.add_argument("--context_stride", type=int, default=None,
                    help="Frame gap between adjacent context views. Smaller = "
                         "tighter baseline (more overlap). Default: auto-compact.")
    ap.add_argument("--context_start", type=int, default=None,
                    help="First context frame index. Default: centre the "
                         "context window in the sequence.")
    ap.add_argument("--eval_index", default=None,
                    help="Where to write eval index json. Default <out_dir>/eval_index.json")
    ap.add_argument("--pose_scale", type=float, default=1.0,
                    help="Multiply COLMAP translations to match near/far scale")
    ap.add_argument("--require_images", action="store_true",
                    help="Fail if an image file referenced by a pose is missing "
                         "(default: skip frames whose image is absent)")
    ap.add_argument("--images_subdir", default=None,
                    help="Where image files live relative to --root. "
                         "Default = the camera name (e.g. mid_center_top_wide/).")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    test_dir = out_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    W, H, fx, fy, cx, cy, model = read_cameras_json(root / "cameras.json", args.camera)
    print(f"Camera '{args.camera}': {W}x{H}, model={model}, "
          f"fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    if model.lower().startswith("kannala") or "fisheye" in model.lower():
        print("  [WARN] This camera is a FISHEYE (kannalaBrandt) model. "
              "VolSplat uses a pinhole model and ignores distortion. "
              "For best quality, undistort the images+intrinsics first.")

    poses = read_images_poses(root / "sparse" / "txt" / "images.txt", args.camera)
    print(f"Found {len(poses)} poses for camera '{args.camera}'.")
    if not poses:
        raise RuntimeError("No poses parsed. Check --camera matches the prefix in images.txt.")

    img_subdir = root / (args.images_subdir if args.images_subdir else args.camera)

    # normalized intrinsics (RE10K convention)
    fx_n, fy_n, cx_n, cy_n = fx / W, fy / H, cx / W, cy / H

    cam_rows = []
    img_bytes_list = []
    ordered_names = []
    missing = []
    for frame, w2c in poses:
        img_path = img_subdir / frame
        if not img_path.exists():
            missing.append(frame)
            if args.require_images:
                raise FileNotFoundError(f"Missing image: {img_path}")
            continue  # skip frames without image on disk
        row = np.zeros(18, dtype=np.float32)
        row[0:4] = [fx_n, fy_n, cx_n, cy_n]
        w2c = w2c.copy()
        w2c[:3, 3] *= args.pose_scale
        row[6:18] = w2c.astype(np.float32).reshape(-1)
        cam_rows.append(row)
        raw = np.frombuffer(img_path.read_bytes(), dtype=np.uint8).copy()
        img_bytes_list.append(torch.from_numpy(raw))
        ordered_names.append(frame)

    print(f"Usable frames (pose + image present): {len(ordered_names)}")
    if missing:
        print(f"  [INFO] {len(missing)} posed frames had no image on disk and were skipped.")
        print(f"         first few missing: {missing[:5]}")

    if len(ordered_names) == 0:
        print("\n[ABORT] No frame has BOTH a pose and an image file.")
        print("        You must copy the image files whose frame indices appear in")
        print("        images.txt into:", img_subdir)
        print("        (poses cover frames like:",
              f"{poses[0][0]} ... {poses[-1][0]})")
        return

    cameras_tensor = torch.from_numpy(np.stack(cam_rows, axis=0))  # [N,18]
    record = {"key": args.scene_key,
              "images": img_bytes_list,
              "cameras": cameras_tensor}
    chunk_name = "000000.torch"
    torch.save([record], test_dir / chunk_name)

    with open(test_dir / "index.json", "w") as f:
        json.dump({args.scene_key: chunk_name}, f, indent=2)

    n = len(ordered_names)
    ctx, tgt = pick_indices(n, args.num_context, args.num_target,
                            context_stride=args.context_stride,
                            context_start=args.context_start)
    if not tgt:  # too few frames -> degenerate; reuse context as target
        tgt = ctx
        print("  [WARN] Not enough frames for separate targets; "
              "targets reuse context (metrics will be meaningless).")
    eval_index = {args.scene_key: {"context": ctx, "target": tgt}}
    eval_path = Path(args.eval_index) if args.eval_index else (out_dir / "eval_index.json")
    with open(eval_path, "w") as f:
        json.dump(eval_index, f, indent=2)

    print("\nDone.")
    print(f"  chunk        : {test_dir / chunk_name}")
    print(f"  index.json   : {test_dir / 'index.json'}")
    print(f"  eval index   : {eval_path}")
    print(f"  usable count : {n}")
    print(f"  context idx  : {ctx}  -> {[ordered_names[i] for i in ctx]}")
    print(f"  target idx   : {tgt}")


if __name__ == "__main__":
    main()
