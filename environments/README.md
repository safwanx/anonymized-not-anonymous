# Environments and external sources

The completed GPU experiments used Python 3.10, PyTorch 2.1.2 + CUDA 12.1, and
torchvision 0.16.2 + CUDA 12.1. `accv-dp2-pip-freeze.txt` records the observed
packages, with machine-specific editable paths removed. It is an inventory,
**not a portable lockfile**: CUDA extensions, drivers, model assets, and source
installations need separate setup.

## CPU checks and paper

Protocol tests and `tools/check_project.py` need only Python 3.10+. The paper's
figure/verifier scripts use `requirements-paper.txt`, Tectonic, and `pdftotext`
(Poppler). The synthetic PBP test additionally needs CPU PyTorch, NumPy, h5py,
pytest, and the pinned PBP source.

## Experiment runtime

In a fresh Python 3.10 environment, install the PyTorch build appropriate to your
hardware. For the recorded CUDA runtime:

```bash
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r environments/requirements-core.txt
python tools/fetch_sources.py --list
python tools/fetch_sources.py
```

`fetch_sources.py` retrieves the exact revisions in `sources.json` and applies
the two recorded source patches. It does not download data or model weights,
install packages, or overwrite an existing source tree. Use `--only
PrivacyBeyondPixels` for just the lightweight PBP integration.

Source roots default to `~/.cache/accv/repos`; `REPOS_DIR` overrides this. Install
the required source packages in the activated environment: `deep-person-reid`,
`insightface/python-package`, and `deep_privacy2` using their build instructions.
Detectron2 and its `projects/DensePose` package must be built against the same
PyTorch/CUDA runtime. OpenGait and PBP are imported directly from their source
trees by the integration modules.

DeepPrivacy2 additionally uses the pinned `tops`, `face_detection`, and `motpy`
sources and packages recorded in the full inventory. Its historical setup
requirements may conflict with modern PyTorch; do not treat an unconstrained
`pip install -e` as a validated complete reconstruction. `--no-deps` is appropriate
only after you have satisfied and checked the recorded runtime dependencies.

No fresh end-to-end GPU installation is certified here. The release checks
exercise CPU audits, synthetic PBP evaluation, saved artifacts, and manuscript
builds. GPU synthesis cannot be revalidated without restoring data and pretrained
weights, which are intentionally not distributed.

VideoMAEv2 uses custom Hugging Face model code (`trust_remote_code=True` in the
extractor). Review that upstream code and cache the intended model version before
executing it. Dependency/model licenses are separate from this repository's MIT
license; see `../THIRD_PARTY.md`.
