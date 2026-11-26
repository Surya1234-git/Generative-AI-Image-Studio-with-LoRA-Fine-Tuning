"""Command line wrapper: build a COCO captions subset that the Dataset Explorer can open.

Example (Colab):
  !wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip -q annotations_trainval2017.zip
  !python tools/import_coco_subset.py --captions annotations/captions_val2017.json \\
          --instances annotations/instances_val2017.json --out /content/coco_subset --max-images 500
Then open the Dataset Explorer, choose "Folder path on this machine / Drive" and enter /content/coco_subset.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.coco_import import build_coco_subset  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--captions", required=True, help="captions_*.json from the official COCO annotations")
    p.add_argument("--instances", help="instances_*.json (optional): puts each image in the folder of its main object category")
    p.add_argument("--images-dir", help="folder with the COCO images (optional): without it the images are downloaded")
    p.add_argument("--out", required=True, help="output dataset folder")
    p.add_argument("--max-images", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    summary = build_coco_subset(a.captions, a.out, a.images_dir, a.instances, a.max_images, a.seed,
                                progress_cb=lambda f: print(f"\r{f:5.0%}", end="", flush=True))
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
