# Source notices

This repository packages inference components of InfiniHand. Third-party components retain their respective licenses; no blanket license is asserted over their code or model assets.

- EgoPreprocess step2 / HaWoR SAM3 tracking: see `third_party/ego_step2/NOTICE.md` and `docs/SOURCE_MAP.md` within that directory. Source: https://github.com/cskrren/EgoPreprocess
- LingBot-Map: https://github.com/Robbyant/lingbot-map ; see `third_party/lingbot-map/LICENSE.txt`. The streaming implementation includes the source runtime's window/memory adaptations.
- WiLoR: https://github.com/rolpotamias/WiLoR ; see `third_party/WiLoR/LICENSE`.
- HaWoR DROID-SLAM backend: https://github.com/ThunderVVV/HaWoR ; upstream DROID-SLAM: https://github.com/princeton-vl/DROID-SLAM ; see `third_party/droid/LICENSE` and LieTorch/Eigen notices.
- SAM3: https://github.com/facebookresearch/sam3 . Install separately; source and gated checkpoint are not bundled.
- MANO: https://mano.is.tue.mpg.de/ . Obtain assets under the applicable license. No MANO asset download is supplied by this repository.

Only inference entrypoints are exposed. Small compatibility and path changes preserve tensor names so the merged state dictionary can be loaded strictly.
