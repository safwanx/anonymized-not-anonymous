# Third-party materials

The root MIT license covers the authors' original software and its documentation.
It does not replace upstream terms or grant rights in data, model weights,
publisher-controlled paper versions, or conference templates.

External implementations are fetched separately, not vendored here. Exact
revisions are recorded in `environments/sources.json`:

| Implementation | Upstream |
| --- | --- |
| DeepPrivacy2 | https://github.com/hukkelas/deep_privacy2 |
| Privacy Beyond Pixels | https://github.com/UCF-CRCV/PrivacyBeyondPixels |
| OpenGait / GaitBase | https://github.com/ShiqiYu/OpenGait |
| OSNet / torchreid | https://github.com/KaiyangZhou/deep-person-reid |
| InsightFace / ArcFace | https://github.com/deepinsight/insightface |
| Detectron2 / DensePose | https://github.com/facebookresearch/detectron2 |

The compatibility patches in `environments/patches/` modify upstream DeepPrivacy2
and InsightFace files and remain subject to upstream terms. They address NumPy
random integer dtype and ONNX session-option forwarding. Generated C++ build
changes are not distributed as source patches.

Other dependencies include PyTorch, Transformers, VideoMAEv2, and Ultralytics.
Check each project's terms before redistribution or commercial use. In
particular, InsightFace's upstream README distinguishes its code license from
the non-commercial research terms of its supplied pretrained models.

NTU RGB+D 120 must be obtained separately from its owners. Filename lists do not
substitute for dataset permission. Local aggregate features and trained artifacts
are excluded from the public source distribution.

`paper/accv.sty`, `paper/accvabbrv.sty`, `paper/llncs.cls`, and `paper/splncs04.bst`
retain their upstream notices. The manuscript and figures are research materials,
not a blanket grant of rights to third-party imagery or publisher content.
