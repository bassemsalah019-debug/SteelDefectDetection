"""
build_space.py - Assemble a ready-to-push Hugging Face Space folder.

Creates deployment/huggingface/space/ containing everything the Space needs:
    app.py, requirements.txt, README.md (the Space card), src/, weights/best.pt, .gitattributes

The src/ package is bundled because the Space app imports it (preprocessing parity,
inference wrapper, Eigen-CAM, and the report stack), and so any custom checkpoint can
un-pickle its layers. The production model is the plain YOLOv8n baseline (stock).

Run:
    python deployment/huggingface/build_space.py

Then push the space/ folder to your Space (instructions are printed, and in docs/DEPLOYMENT.md).
The space/ folder is a build artifact and is git-ignored from the main repo.
"""
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SPACE = HERE / "space"
# Production model (2026-06 audit + 5-seed study + deployment benchmark): YOLOv8n baseline @640.
WEIGHTS = ROOT / "results" / "baseline_640" / "weights" / "best.pt"


def main() -> None:
    assert WEIGHTS.exists(), f"trained weights not found: {WEIGHTS} (train updated_03 / notebook 08 first)"
    if SPACE.exists():
        shutil.rmtree(SPACE)
    (SPACE / "weights").mkdir(parents=True)

    # Space entrypoint + config (live in this folder). Dockerfile because HF now
    # builds Streamlit apps via the Docker SDK (sdk: docker in the README card).
    for name in ("app.py", "requirements.txt", "README.md", "Dockerfile"):
        shutil.copy(HERE / name, SPACE / name)

    # Dark "Molten Graphite" theme so the Studio UI renders correctly on HF.
    shutil.copytree(ROOT / ".streamlit", SPACE / ".streamlit",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # Source package needed to un-pickle the custom checkpoint
    shutil.copytree(ROOT / "src", SPACE / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # The trained model
    shutil.copy(WEIGHTS, SPACE / "weights" / "best.pt")

    # Track the weight with git-lfs on the Space
    (SPACE / ".gitattributes").write_text("*.pt filter=lfs diff=lfs merge=lfs -text\n")

    size = sum(p.stat().st_size for p in SPACE.rglob("*") if p.is_file()) / 1e6
    print(f"Built Space at: {SPACE}  ({size:.1f} MB)")
    print("Contents:")
    for p in sorted(SPACE.rglob("*")):
        if p.is_file():
            print("  ", p.relative_to(SPACE).as_posix())
    print("\nNext steps (one-time):")
    print("  hf auth login                          # paste a WRITE token from hf.co/settings/tokens")
    print("  # create a Streamlit Space at https://huggingface.co/new-space (SDK = Streamlit), then:")
    print(f"  hf upload <username>/<space-name> {SPACE} . --repo-type space")
    print("  # ^ uploads the whole folder; watch it build on the Space page.")


if __name__ == "__main__":
    main()
