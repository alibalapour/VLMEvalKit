"""SonoReason multi-turn VLM localization -> MedSAM -> VLM verification agent."""

import base64
import io
import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw

from .base import BaseModel


VERIFY_PROMPT = """Verify a mask proposed by an external medical image segmentor for
the exact segmentation target described in the ORIGINAL SEGMENTATION TASK below.
The first image is the original ultrasound and the second is an overlay: green is the
proposed mask and red is its prompting box. Accept only if green covers the complete
visible target tissue. Reject posterior acoustic shadowing beneath the target, normal
tissue, labels, or artifacts. Calipers may be localization clues. Return only JSON:
{"accept": true, "bbox": [x_min, y_min, x_max, y_max], "reason": "short reason"}
If incorrect, set accept=false and provide a corrected pixel-coordinate bbox."""


class SonoReasonSegmentationAgent(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, **kwargs):
        super().__init__()
        self.backend = os.environ.get('SONOREASON_VLM_BACKEND', 'openrouter').lower()
        if self.backend not in {'openrouter', 'local'}:
            raise ValueError('SONOREASON_VLM_BACKEND must be openrouter or local')
        self.vlm_model_name = os.environ.get(
            'SONOREASON_AGENT_VLM_MODEL',
            'qwen/qwen3.5-9b' if self.backend == 'openrouter'
            else 'google/medgemma-1.5-4b-it')
        self.segmentor_model = os.environ.get(
            'SONOREASON_SEGMENTOR_MODEL', 'wanglab/medsam-vit-base')
        self.segmentor_device = os.environ.get('SONOREASON_SEGMENTOR_DEVICE', 'auto')
        self.max_refinements = int(os.environ.get('SONOREASON_MAX_REFINEMENTS', '2'))
        self.max_tokens = int(os.environ.get('SONOREASON_VLM_MAX_TOKENS', '12000'))
        self.reasoning_effort = os.environ.get('SONOREASON_REASONING_EFFORT', 'low')
        self.output_dir = Path(os.environ.get(
            'SONOREASON_AGENT_OUTPUT_DIR', 'sonoreason_agent_masks')).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.local_vlm = None
        if self.backend == 'local':
            from .medgemma import MedGemma
            self.local_vlm = MedGemma(model_path=self.vlm_model_name)

    @staticmethod
    def _json(text):
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if not match:
                raise ValueError(f'VLM returned no JSON: {text[:300]}')
            value = json.loads(match.group(0))
        if not isinstance(value, dict):
            raise ValueError('VLM response is not a JSON object')
        return value

    @staticmethod
    def _data_url(image):
        stream = io.BytesIO()
        image.save(stream, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()

    def _openrouter(self, images, prompt):
        key = os.environ.get('OPENROUTER_API_KEY', '')
        if not key:
            raise RuntimeError('OPENROUTER_API_KEY is required for openrouter backend')
        content = [{'type': 'text', 'text': prompt}]
        content.extend({
            'type': 'image_url', 'image_url': {'url': self._data_url(image)}
        } for image in images)
        last = 'no response'
        for content_attempt in range(2):
            budget = self.max_tokens * (content_attempt + 1)
            for network_attempt in range(4):
                try:
                    response = requests.post(
                        'https://openrouter.ai/api/v1/chat/completions',
                        headers={'Authorization': f'Bearer {key}'},
                        json={
                            'model': self.vlm_model_name,
                            'messages': [{'role': 'user', 'content': content}],
                            'temperature': 0,
                            'max_completion_tokens': budget,
                            'reasoning': {'effort': self.reasoning_effort},
                        }, timeout=(30, 300))
                    response.raise_for_status()
                    body = response.json()
                    choice = (body.get('choices') or [{}])[0]
                    message = choice.get('message') or {}
                    text = message.get('content')
                    if isinstance(text, str) and text.strip():
                        return text.strip()
                    last = f"content={type(text).__name__}, finish={choice.get('finish_reason')}"
                    break
                except (requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout,
                        requests.exceptions.ChunkedEncodingError) as exc:
                    if network_attempt == 3:
                        raise
                    time.sleep(2 ** (network_attempt + 1))
        raise RuntimeError(f'OpenRouter returned no visible text ({last})')

    def _vlm(self, images, prompt):
        if self.backend == 'openrouter':
            return self._openrouter(images, prompt)
        message = [{'type': 'image', 'value': image} for image in images]
        message.append({'type': 'text', 'value': prompt})
        return self.local_vlm.generate_inner(message)

    @staticmethod
    def _bbox(value, width, height):
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        try:
            x0, y0, x1, y1 = map(float, value)
        except (TypeError, ValueError):
            return None
        x0, x1 = sorted((max(0, min(x0, width - 1)), max(0, min(x1, width - 1))))
        y0, y1 = sorted((max(0, min(y0, height - 1)), max(0, min(y1, height - 1))))
        return [x0, y0, x1, y1] if x1 - x0 >= 2 and y1 - y0 >= 2 else None

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_sam(name, device):
        from transformers import SamModel, SamProcessor
        processor = SamProcessor.from_pretrained(name)
        model = SamModel.from_pretrained(name).to(device).eval()
        return processor, model

    def _segment(self, image, bbox):
        import torch
        device = self.segmentor_device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        processor, model = self._load_sam(self.segmentor_model, device)
        inputs = processor(images=image, input_boxes=[[bbox]], return_tensors='pt')
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs, multimask_output='medsam' not in self.segmentor_model.lower())
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs['original_sizes'].cpu(),
            inputs['reshaped_input_sizes'].cpu())[0]
        scores = outputs.iou_scores.detach().cpu().reshape(-1)
        best = int(torch.argmax(scores))
        return masks.reshape(-1, *masks.shape[-2:])[best].numpy() > 0, float(scores[best])

    @staticmethod
    def _overlay(image, mask, bbox):
        array = np.asarray(image.convert('RGB'), dtype=np.float32)
        green = np.zeros_like(array)
        green[..., 1] = 255
        array[mask] = .62 * array[mask] + .38 * green[mask]
        overlay = Image.fromarray(array.astype(np.uint8))
        ImageDraw.Draw(overlay).rectangle(bbox, outline=(255, 40, 40), width=2)
        return overlay

    def generate_inner(self, message, dataset=None):
        image_paths = [item['value'] for item in message if item['type'] == 'image']
        prompt = '\n'.join(item['value'] for item in message if item['type'] == 'text')
        if not image_paths:
            raise ValueError('Segmentation agent requires one ultrasound image')
        image = Image.open(image_paths[0]).convert('RGB')
        try:
            localization = self._json(self._vlm([image], prompt))
        except Exception as exc:
            return json.dumps({
                'status': 'failed_localization',
                'error': f'{type(exc).__name__}: {exc}',
                'mask_path': None,
                'bbox': None,
                'accepted': False,
                'attempts': 0,
                'vlm_backend': self.backend,
                'vlm_model': self.vlm_model_name,
                'segmentor_model': self.segmentor_model,
            })
        if not localization.get('lesion_present', True):
            mask = np.zeros((image.height, image.width), bool)
            bbox, accepted, attempts = [], True, 0
        else:
            bbox = self._bbox(localization.get('bbox'), image.width, image.height)
            if bbox is None:
                return json.dumps({'status': 'skipped_invalid_bbox', 'bbox': None})
            accepted = False
            verification_error = None
            for attempt in range(self.max_refinements + 1):
                mask, confidence = self._segment(image, bbox)
                overlay = self._overlay(image, mask, bbox)
                verification_prompt = (
                    f'{VERIFY_PROMPT}\n\nORIGINAL SEGMENTATION TASK:\n{prompt}\n\n'
                    'Judge only whether the green mask segments that specified target.'
                )
                try:
                    verification = self._json(
                        self._vlm([image, overlay], verification_prompt))
                except Exception as exc:
                    # The current SAM mask is still a usable prediction. A VLM
                    # formatting/length/network failure must not abort all
                    # remaining dataset rows.
                    verification_error = f'{type(exc).__name__}: {exc}'
                    break
                accepted = verification.get('accept') is True
                attempts = attempt + 1
                if accepted or attempt == self.max_refinements:
                    break
                corrected = self._bbox(
                    verification.get('bbox'), image.width, image.height)
                if corrected is None:
                    break
                bbox = corrected
        source_path = Path(image_paths[0])
        dataset_folder = source_path.parent.name or str(dataset or 'unknown')
        dataset_folder = re.sub(r'[^A-Za-z0-9_.-]+', '_', dataset_folder).strip('._') or 'unknown'
        image_stem = re.sub(
            r'[^A-Za-z0-9_.-]+', '_', source_path.stem).strip('._') or 'image'
        dataset_output_dir = self.output_dir / dataset_folder
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        mask_path = dataset_output_dir / f'{image_stem}.png'
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        return json.dumps({
            'status': 'completed', 'mask_path': str(mask_path), 'bbox': bbox,
            'accepted': accepted, 'attempts': attempts,
            'segmentor_confidence': confidence if attempts else None,
            'verification_error': verification_error if attempts else None,
            'vlm_backend': self.backend, 'vlm_model': self.vlm_model_name,
            'segmentor_model': self.segmentor_model,
        })
