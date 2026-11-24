"""
Custom nnUNet trainers for the ResInv paper (experiment E2: does a model's robust
resolution band follow the range of object scales it saw during training?).

nnUNetTrainer's default SpatialTransform uses scaling=(0.7, 1.4) with p_scaling=0.2.
In batchgeneratorsv2 larger scaling values mean SMALLER objects: a patch scaled by s shows
objects at 1/s of their size, i.e. what they would look like at s times the pixel size.
Control (trained at 2.36 nm) therefore sees objects as if imaged between ~1.7 and ~3.3 nm,
which is roughly its measured robust band.

These trainers change ONLY the scaling range (and, where needed, the crop that feeds it).
Everything else is the stock pipeline: rotation, mirroring, intensity augmentation,
deep supervision, epochs, optimizer. To stay robust across nnUNet versions the stock
get_training_transforms() is called unchanged and its SpatialTransform is patched in place,
instead of copying the transform list (which differs between 2.6 and 2.8).

Online zoom-out has a hard ceiling set by image size: to show objects at 1/s size the
dataloader must crop s x the patch from the image. TEM1 images are 3762x2286 and the paper
plans use a 768x1152 patch, so s beyond ~2.9 starts sampling padding. That is why the full
2.36 -> 16 nm range (s ~ 6.8) cannot be done online and needs offline multires copies.

nnUNetTrainerScaleAugLogWide is the exception: the continuous counterpart of the multi-resolution
model, it zooms every patch by a factor drawn log-uniformly over the whole 1.18 -> 16 nm range
(s from 0.5 to 6.78). Beyond s ~ 2.9 the zoomed-out patch is the image shrunk inside padding,
which is also what the offline 10 and 16 nm copies are (they are smaller than the patch). The
crop is capped (MAX_INITIAL_PATCH) instead of growing to the ~7800 px square that full rotation
at s = 6.78 would ask for. The cap per axis is 2 * image - patch, the smallest crop that always
contains the whole image wherever nnUNet's data loader places it (lb is drawn from
[-(crop - patch) / 2, image - (crop + patch) / 2]), so a strongly zoomed-out patch shows the
whole image inside padding, like the offline copies, rather than a cut-off part of it.

Install into the venv with install_custom_trainers.sh, then train with
    nnUNetv2_train <dataset> 2d <fold> -tr nnUNetTrainerScaleAug2p5 -p <plans>
"""

import numpy as np
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform

from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def _iter_transforms(t):
    """Yield every transform in a (possibly nested) batchgeneratorsv2 pipeline."""
    yield t
    for child in getattr(t, "transforms", None) or []:
        yield from _iter_transforms(child)
    inner = getattr(t, "transform", None)
    if inner is not None:
        yield from _iter_transforms(inner)


class LogUniform:
    """Scale factor drawn log-uniformly in [lo, hi], so every octave of pixel size is equally likely.

    A class rather than a lambda so the transform pipeline stays picklable for the
    data-augmentation workers. batchgeneratorsv2's sample_scalar calls it with image= and dim=.
    """

    def __init__(self, lo, hi):
        self.lo, self.hi = float(lo), float(hi)

    def __call__(self, *args, **kwargs):
        return float(np.exp(np.random.uniform(np.log(self.lo), np.log(self.hi))))

    def __repr__(self):
        return f"LogUniform({self.lo}, {self.hi})"


class nnUNetTrainerScaleAug(nnUNetTrainer):
    """Base class. Subclasses set SCALING (and optionally P_SCALING). Not meant to be trained directly."""

    SCALING = (0.7, 1.4)      # nnUNet default
    P_SCALING = 0.2           # nnUNet default
    LOG_UNIFORM = False       # False: uniform in SCALING (nnUNet's sampling); True: log-uniform
    MAX_INITIAL_PATCH = None  # cap on the crop fed to the spatial transform (pixels, int or one per axis)

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        # Stock nnUNet sizes the crop for a hardcoded (0.85, 1.25) scale range (old-style
        # semantics: it divides by the smallest value). Zooming out by SCALING[1] needs a crop
        # SCALING[1] x the patch, otherwise the extra field of view is filled with padding.
        dim = len(self.configuration_manager.patch_size)
        needed = get_patch_size(self.configuration_manager.patch_size[-dim:],
                                rotation_for_DA, rotation_for_DA, rotation_for_DA,
                                (min(0.85, 1.0 / max(self.SCALING)), 1.25))
        initial_patch_size = np.maximum(np.asarray(initial_patch_size), needed).astype(int)
        if self.MAX_INITIAL_PATCH is not None:
            initial_patch_size = np.minimum(initial_patch_size, np.asarray(self.MAX_INITIAL_PATCH)).astype(int)
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = self.configuration_manager.patch_size[0]
        self.print_to_log_file(f"ResInv scale aug: scaling={self.SCALING}, log_uniform={self.LOG_UNIFORM}, "
                               f"p_scaling={self.P_SCALING}, "
                               f"initial_patch_size={initial_patch_size.tolist()}")
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def get_training_transforms(self, *args, **kwargs):
        transforms = nnUNetTrainer.get_training_transforms(*args, **kwargs)
        patched = 0
        for t in _iter_transforms(transforms):
            if isinstance(t, SpatialTransform):
                t.scaling = LogUniform(*self.SCALING) if self.LOG_UNIFORM else self.SCALING
                t.p_scaling = self.P_SCALING
                patched += 1
        if patched != 1:
            raise RuntimeError(f"Expected exactly one SpatialTransform in the training pipeline, found {patched}. "
                               "The nnUNet version changed its augmentation layout; update this trainer.")
        return transforms


class nnUNetTrainerScaleAug2p5(nnUNetTrainerScaleAug):
    """Objects down to 1/2.5 of their size: control at 2.36 nm sees ~1.7 to ~5.9 nm equivalents."""
    SCALING = (0.7, 2.5)


class nnUNetTrainerScaleAug2p5_p05(nnUNetTrainerScaleAug):
    """Same range, scaling applied to half the patches instead of a fifth (secondary variant)."""
    SCALING = (0.7, 2.5)
    P_SCALING = 0.5


class nnUNetTrainerScaleAugLogWide(nnUNetTrainerScaleAug):
    """Continuous counterpart of the multi-resolution model (multires_fine: 1.18, 2.36, 7, 10, 16 nm).

    Every patch is zoomed by s drawn log-uniformly from 1.18/2.36 to 16/2.36, so a model trained at
    2.36 nm sees objects as if imaged anywhere from 1.18 to 16 nm, with nnUNet's own interpolation
    (no anti-aliasing when zooming out). Scaling stays synchronised across axes (stock setting).
    """
    SCALING = (1.18 / 2.36, 16.0 / 2.36)  # 0.5 to 6.78
    P_SCALING = 1.0
    LOG_UNIFORM = True
    # TEM1 images are 2286 x 3762 and the patch 768 x 1152: 2 * image - patch per axis. Stock crop is
    # 1355 x 1355, uncapped would be ~7800 x 7800.
    MAX_INITIAL_PATCH = (2 * 2286 - 768, 2 * 3762 - 1152)  # (3804, 6372)
