"""
Convert a COLMAP reconstruction into the RE10K-style `.torch` chunk format
expected by VolSplat's `DatasetRE10k` dataloader.

Outputs (under <out_dir>/test/):
    - 000000.torch    : a list with one dict {key, images, cameras}
    - index.json      : {scene_key: "000000.torch"}
And:
    - <eval_index>.json : {scene_key: {"context": [...], "target": [...]}}

The 18-dim `cameras` row layout required by DatasetRE10k.convert_poses:
    [0:4]  = fx, fy, cx, cy   (NORMALIZED by image width/height)
    [4:6]  = placeholder (near/far; unused here, set 0)
    [6:18] = world-to-camera (w2c) 3x4 matrix, row-major flattened (OpenCV convention)

Usage (run inside the `volsplat` conda env):
    python tools/colmap_to_re10k.py \
        --colmap_dir /path/to/colmap_test_full \
        --sparse_dir /path/to/colmap_test_full/sparse/0 \
        --images_dir /path/to/colmap_test_full/images \
        --out_dir datasets/mycolmap \
        --scene_key myscene \
        --num_context 6
"""
import argparse
import json
import struct
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


# ----------------------------------------------------------------------------
# COLMAP binary/text readers (self-contained, no pycolmap dependency)
# ----------------------------------------------------------------------------
def _read_next_bytes(fid, num_bytes, fmt, endian="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + fmt, data)


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num):
            cam_id, model_id, width, height = _read_next_bytes(fid, 24, "iiQQ")
            # number of params per model id (subset of COLMAP models)
            num_params = {
                0: 3,   # SIMPLE_PINHOLE
                1: 4,   # PINHOLE
                2: 4,   # SIMPLE_RADIAL
                3: 5,   # RADIAL
                4: 8,   # OPENCV
                5: 8,   # OPENCV_FISHEYE
                6: 12,  # FULL_OPENCV
            }[model_id]
            params = _read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[cam_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": np.array(params),
            }
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num):
            props = _read_next_bytes(fid, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5])   # qw, qx, qy, qz
            tvec = np.array(props[5:8])
            cam_id = props[8]
            name = ""
            c = _read_next_bytes(fid, 1, "c")[0]
            while c != b"\x00":
                name += c.decode("utf-8")
                c = _read_next_bytes(fid, 1, "c")[0]
            num_pts = _read_next_bytes(fid, 8, "Q")[0]
            _ = _read_next_bytes(fid, 24 * num_pts, "ddq" * num_pts)
            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": cam_id,
                "name": name,
            }
    return images


def read_cameras_text(path):
    cameras = {}
    model_params = {
        "SIMPLE_PINHOLE": 3, "PINHOLE": 4, "SIMPLE_RADIAL": 4,
        "RADIAL": 5, "OPENCV": 8, "OPENCV_FISHEYE": 8, "FULL_OPENCV": 12,
    }
    model_id_map = {
        "SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2,
        "RADIAL": 3, "OPENCV": 4, "OPENCV_FISHEYE": 5, "FULL_OPENCV": 6,
    }
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            els = line.split()
            cam_id = int(els[0])
            model = els[1]
            width, height = int(els[2]), int(els[3])
            params = np.array([float(x) for x in els[4:4 + model_params[model]]])
            cameras[cam_id] = {
                "model_id": model_id_map[model],
                "width": width, "height": height, "params": params,
            }
    return cameras


def read_images_text(path):
    images = {}
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    # image lines come in pairs; 2nd line (points) is ignored
    for i in range(0, len(lines), 2):
        els = lines[i].split()
        image_id = int(els[0])
        qvec = np.array([float(x) for x in els[1:5]])
        tvec = np.array([float(x) for x in els[5:8]])
        cam_id = int(els[8])
        name = els[9]
        images[image_id] = {
            "qvec": qvec, "tvec": tvec, "camera_id": cam_id, "name": name,
        }
    return images


def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def intrinsics_from_camera(cam):
    """Return (fx, fy, cx, cy) in pixels for the supported models."""
    p = cam["params"]
    model = cam["model_id"]
    if model == 0:   # SIMPLE_PINHOLE: f, cx, cy
        fx = fy = p[0]; cx, cy = p[1], p[2]
    elif model in (2, 3):  # SIMPLE_RADIAL / RADIAL: f, cx, cy, ...
        fx = fy = p[0]; cx, cy = p[1], p[2]
    else:            # PINHOLE / OPENCV / FULL_OPENCV: fx, fy, cx, cy, ...
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    return fx, fy, cx, cy


# ----------------------------------------------------------------------------
# Main conversion
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse_dir", required=True,
                    help="COLMAP sparse model dir containing cameras/images .bin or .txt")
    ap.add_argument("--images_dir", required=True, help="Directory with the actual image files")
    ap.add_argument("--out_dir", required=True, help="Output dataset root (will create <out_dir>/test/)")
    ap.add_argument("--scene_key", default="myscene")
    ap.add_argument("--num_context", type=int, default=6)
    ap.add_argument("--eval_index", default=None,
                    help="Path to write evaluation index json. Default: <out_dir>/eval_index.json")
    ap.add_argument("--pose_scale", type=float, default=1.0,
                    help="Multiply COLMAP translations by this to match near/far scale")
    args = ap.parse_args()

    sparse = Path(args.sparse_dir)
    images_root = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    test_dir = out_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    # --- read COLMAP model (prefer .bin, fallback .txt) ---
    if (sparse / "cameras.bin").exists():
        cameras = read_cameras_binary(sparse / "cameras.bin")
        images = read_images_binary(sparse / "images.bin")
    elif (sparse / "cameras.txt").exists():
        cameras = read_cameras_text(sparse / "cameras.txt")
        images = read_images_text(sparse / "images.txt")
    else:
        raise FileNotFoundError(f"No cameras.bin/.txt found under {sparse}")

    # sort images by filename for deterministic ordering
    img_items = sorted(images.values(), key=lambda im: im["name"])
    print(f"Found {len(img_items)} images, {len(cameras)} camera models.")

    cam_rows = []
    img_bytes_list = []
    ordered_names = []
    for im in img_items:
        cam = cameras[im["camera_id"]]
        W, H = cam["width"], cam["height"]
        fx, fy, cx, cy = intrinsics_from_camera(cam)

        # normalize intrinsics by image size (RE10K convention)
        fx_n, fy_n = fx / W, fy / H
        cx_n, cy_n = cx / W, cy / H

        # world-to-camera (COLMAP stores R, t as w2c in OpenCV convention)
        R = qvec2rotmat(im["qvec"])
        t = im["tvec"].reshape(3, 1) * args.pose_scale
        w2c_3x4 = np.concatenate([R, t], axis=1).astype(np.float32)  # 3x4

        row = np.zeros(18, dtype=np.float32)
        row[0:4] = [fx_n, fy_n, cx_n, cy_n]
        row[4:6] = 0.0  # placeholder
        row[6:18] = w2c_3x4.reshape(-1)
        cam_rows.append(row)

        # read raw image bytes
        img_path = images_root / im["name"]
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")
        raw = np.frombuffer(img_path.read_bytes(), dtype=np.uint8).copy()
        img_bytes_list.append(torch.from_numpy(raw))
        ordered_names.append(im["name"])

    cameras_tensor = torch.from_numpy(np.stack(cam_rows, axis=0))  # [N, 18]

    record = {
        "key": args.scene_key,
        "images": img_bytes_list,     # list of uint8 tensors (encoded bytes)
        "cameras": cameras_tensor,    # [N, 18]
    }
    chunk_name = "000000.torch"
    torch.save([record], test_dir / chunk_name)

    # index.json : scene_key -> chunk filename
    with open(test_dir / "index.json", "w") as f:
        json.dump({args.scene_key: chunk_name}, f, indent=2)

    # evaluation index: pick first num_context as context, the rest as target
    n = len(ordered_names)
    nc = min(args.num_context, n)
    context = list(range(nc))
    target = list(range(nc, n)) if n > nc else list(range(n))  # fallback: all as target
    eval_index = {args.scene_key: {"context": context, "target": target}}
    eval_path = Path(args.eval_index) if args.eval_index else (out_dir / "eval_index.json")
    with open(eval_path, "w") as f:
        json.dump(eval_index, f, indent=2)

    print("Done.")
    print(f"  chunk        : {test_dir / chunk_name}")
    print(f"  index.json   : {test_dir / 'index.json'}")
    print(f"  eval index   : {eval_path}")
    print(f"  image order  : {ordered_names}")
    print(f"  context idx  : {context}")
    print(f"  target idx   : {target}")


if __name__ == "__main__":
    main()
