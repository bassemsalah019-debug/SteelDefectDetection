import json

AUG_LINES = [
    "    # --- grayscale + low-contrast-texture aware augmentation (targets crazing / rolled-in_scale) ---\n",
    "    hsv_h=0.0, hsv_s=0.0,    # NEU-DET is grayscale -> hue/saturation jitter is a literal no-op; drop it\n",
    "    hsv_v=0.4,               # brightness jitter IS meaningful on gray; keep mild (default)\n",
    "    flipud=0.5,              # steel defects have no canonical up/down -> vertical flip doubles orientations\n",
    "    # optional experiments (uncomment to try): degrees=5.0, scale=0.3 (less shrink of faint texture)\n",
]

for fn in ["updated_05_train_improved.ipynb", "updated_06_train_lzy.ipynb"]:
    nb = json.load(open(fn, encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = c["source"]
        # find the mixup line in the train cell and insert aug block right after it
        for i, line in enumerate(src):
            if "mixup=0.1," in line:
                # avoid double-insert
                if any("texture aware augmentation" in s for s in src):
                    break
                c["source"] = src[: i + 1] + AUG_LINES + src[i + 1 :]
                print(f"{fn}: inserted {len(AUG_LINES)} aug lines after mixup")
                break
    json.dump(nb, open(fn, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
