"""Official UltraSam box-prompt inference adapter."""

import copy
import os
import sys
import types
from functools import partial
from pathlib import Path

import numpy as np


class UltraSamSegmentor:
    """Load CAMMA UltraSam and run one box-prompted ultrasound mask."""

    def __init__(self, root, checkpoint, config, device):
        import torch

        # mmpretrain 1.2 imports optional BLIP code that is incompatible with
        # Transformers 5. UltraSam uses vision backbones only, so provide an
        # empty optional multimodal namespace before mmpretrain initializes.
        sys.modules.setdefault(
            'mmpretrain.models.multimodal',
            types.ModuleType('mmpretrain.models.multimodal'),
        )

        from mmengine.config import Config
        from mmdet.apis import init_detector

        self.root = Path(root).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.config = Path(config).resolve()
        for path, label in (
            (self.root, 'repository'),
            (self.checkpoint, 'checkpoint'),
            (self.config, 'config'),
        ):
            if not path.exists():
                raise FileNotFoundError(f'UltraSam {label} not found: {path}')

        root_string = str(self.root)
        if root_string not in sys.path:
            sys.path.insert(0, root_string)

        # UltraSam's official checkpoint contains MMEngine metadata. PyTorch
        # 2.6+ defaults torch.load to weights_only=True, so trusted official
        # checkpoints need the legacy mode while MMEngine loads this file.
        original_torch_load = torch.load
        torch.load = partial(original_torch_load, weights_only=False)
        previous_cwd = os.getcwd()
        try:
            os.chdir(self.root)
            self.model = init_detector(
                str(self.config), str(self.checkpoint), device=device)
        finally:
            os.chdir(previous_cwd)
            torch.load = original_torch_load

        cfg = Config.fromfile(str(self.config))
        pipeline = copy.deepcopy(cfg.test_dataloader.dataset.pipeline)
        pipeline[0] = dict(type='LoadImageFromNDArray')
        for transform in pipeline:
            if transform['type'] == 'GetPointBox':
                # The VLM box is the requested prompt; do not randomly jitter it.
                transform['max_jitter'] = 0.0
            elif transform['type'] == 'GetPromptType':
                # PromptType is [POINT, BOX] in the official UltraSam code.
                transform['prompt_probabilities'] = [0.0, 1.0]

        from mmcv.transforms import Compose
        self.pipeline = Compose(pipeline)
        self.device = device

    def segment(self, image, bbox):
        import torch
        import torch.nn.functional as torch_functional
        from mmengine.dataset import pseudo_collate

        # UltraSam's SAMAttention uses 256-d inputs with 128-d downsampled
        # attention projections. Its normal MMEngine Runner installs a custom
        # multi-head-attention function via MonkeyPatchHook to support that
        # layout. This adapter calls init_detector/test_step directly, so hooks
        # are never run. Apply the same compatibility function for this forward
        # pass only, rather than permanently changing PyTorch process-wide.
        from endosam.models.utils.custom_functional import (
            multi_head_attention_forward as ultrasam_attention_forward,
        )

        # DetDataPreprocessor is configured with bgr_to_rgb=True.
        array = np.asarray(image.convert('RGB'))[:, :, ::-1].copy()
        x0, y0, x1, y1 = map(float, bbox)
        polygon = [x0, y0, x1, y0, x1, y1, x0, y1]
        instance = {
            'bbox': [x0, y0, x1, y1],
            'bbox_label': 0,
            'ignore_flag': 0,
            # The official test pipeline loads masks before selecting prompt
            # type. This rectangle is pipeline metadata only; BOX is forced.
            'mask': [polygon],
        }
        packed = self.pipeline({
            'img': array,
            'img_id': 0,
            'instances': [instance],
        })
        original_attention_forward = torch_functional.multi_head_attention_forward
        try:
            torch_functional.multi_head_attention_forward = ultrasam_attention_forward
            with torch.inference_mode():
                result = self.model.test_step(pseudo_collate([packed]))[0]
        finally:
            torch_functional.multi_head_attention_forward = original_attention_forward

        predictions = result.pred_instances
        if len(predictions) == 0:
            return np.zeros((image.height, image.width), dtype=bool), 0.0
        scores = predictions.scores.detach().cpu().reshape(-1)
        best = int(torch.argmax(scores))
        mask = predictions.masks[best].detach().cpu().numpy().astype(bool)
        return mask, float(scores[best])
