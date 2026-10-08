import ast
import base64
import importlib.util
import io
import json
import logging
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import pandas as pd
from PIL import Image


def _load_sonoreason():
    vlmeval = types.ModuleType('vlmeval')
    vlmeval.__path__ = ['vlmeval']
    dataset = types.ModuleType('vlmeval.dataset')
    dataset.__path__ = ['vlmeval/dataset']

    image_base = types.ModuleType('vlmeval.dataset.image_base')
    image_base.ImageBaseDataset = type('ImageBaseDataset', (), {})

    prompts = types.ModuleType('vlmeval.dataset.sonoreason_prompts')
    prompts.resolve_prompt_template = lambda **kwargs: ('test', {})
    prompts.build_prompt_variants = lambda template: {'direct_prompt': ''}
    prompts.load_prompt_file = (
        lambda filename: '{anatomy} {segmentation_target} {feature_name} {feature_prompt}')

    smp = types.ModuleType('vlmeval.smp')
    smp.get_logger = logging.getLogger
    smp.load = lambda path: None
    smp.dump = lambda value, path: None

    json_repair = types.ModuleType('json_repair')
    json_repair.loads = json.loads

    modules = {
        'vlmeval': vlmeval,
        'vlmeval.dataset': dataset,
        'vlmeval.dataset.image_base': image_base,
        'vlmeval.dataset.sonoreason_prompts': prompts,
        'vlmeval.smp': smp,
        'json_repair': json_repair,
    }
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            'vlmeval.dataset.sonoreason',
            'vlmeval/dataset/sonoreason.py',
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        sys.modules.pop(spec.name, None)
        return module


class TestSonoReasonLimits(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.module = _load_sonoreason()
        cls.dataset = cls.module.SonoReasonDD()

    def test_limit_aliases_resolve_to_one_value(self):
        with mock.patch.dict(
            os.environ,
            {'SONOREASON_LIMIT': '4', 'SONOREASON_SAMPLE_LIMIT': '4'},
            clear=True,
        ):
            name, value = self.dataset._configured_limit()
        self.assertEqual(name, 'SONOREASON_LIMIT/SONOREASON_SAMPLE_LIMIT')
        self.assertEqual(value, 4)

    def test_conflicting_limit_aliases_fail_loudly(self):
        with mock.patch.dict(
            os.environ,
            {'SONOREASON_LIMIT': '4', 'SONOREASON_SAMPLE_LIMIT': '2'},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, 'Conflicting SonoReason sample limits'):
                self.dataset._configured_limit()

    def test_stratified_limit_returns_exact_requested_count(self):
        frame = pd.DataFrame({
            'index': range(8),
            'answer': ['a'] * 4 + ['b'] * 3 + ['c'],
        })
        with mock.patch.dict(os.environ, {'SONOREASON_LIMIT': '5'}, clear=True):
            limited = self.dataset._apply_limit(frame)
        self.assertEqual(len(limited), 5)
        self.assertEqual(limited['answer'].value_counts().to_dict(), {'a': 2, 'b': 2, 'c': 1})
        self.assertEqual(limited['index'].tolist(), [0, 1, 4, 5, 7])

    def test_unstratified_segmentation_limit_uses_sample_alias(self):
        frame = pd.DataFrame({'index': range(5)})
        with mock.patch.dict(
            os.environ, {'SONOREASON_SAMPLE_LIMIT': '2'}, clear=True
        ):
            limited = self.dataset._apply_limit(frame, stratify_on=None)
        self.assertEqual(limited['index'].tolist(), [0, 1])


class TestSonoReasonFeatureExpansion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.module = _load_sonoreason()
        cls.dataset = cls.module.SonoReasonDD()

    def test_every_feature_row_keeps_the_real_image(self):
        specs = {
            'shape': {
                'heading': 'SHAPE',
                'label': 'shape_label',
                'measurements': {},
                'thresholds': {},
            },
            'orientation': {
                'heading': 'ORIENTATION',
                'label': 'orientation_label',
                'measurements': {},
                'thresholds': {},
            },
        }
        source = pd.DataFrame([{
            'img_data': 'base64-image-payload',
            'anatomy_location': 'breast',
            'shape_label': 'oval',
            'orientation_label': 'parallel',
        }])
        prompts = {'shape': 'shape prompt', 'orientation': 'orientation prompt'}
        with (
            mock.patch.object(self.module, 'FEATURE_SPECS', specs),
            mock.patch.object(self.module, '_load_feature_prompts', return_value=prompts),
            mock.patch.dict(
                os.environ, {'SONOREASON_FEATURE_PROMPTS_FILE': 'unused'}, clear=True
            ),
        ):
            expanded = self.dataset._build_feature_data(source)

        self.assertEqual(expanded['index'].tolist(), ['0__shape', '0__orientation'])
        self.assertEqual(
            expanded['image'].tolist(),
            ['base64-image-payload', 'base64-image-payload'],
        )

    def test_mask_measurement_rows_keep_aspect_ratio_and_scale_lengths(self):
        def encode(image):
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode()

        mask = Image.new('L', (40, 20))
        mask.paste(255, (10, 5, 30, 15))  # 20x10 lesion on a 40x20 landscape image
        columns = {
            column
            for spec in self.module.MASK_MEASUREMENT_COLUMNS.values()
            for column in spec.values()
        }
        source = pd.DataFrame([{
            **{column: 1.0 for column in columns},
            'long_axis_pixels': 20.0,
            'orientation_degrees': 10.0,
            'img_data': encode(Image.new('L', (40, 20), 90)),
            'mask': encode(mask),
            'anatomy_location': 'breast',
            'patient_id': 'p1',
            'dataset_name': 'd1',
        }])
        prompts = {feature: 'prompt' for feature in self.module.MASK_MEASUREMENT_COLUMNS}
        with tempfile.TemporaryDirectory() as tmp:
            targets = os.path.join(tmp, 'targets.json')
            with open(targets, 'w') as handle:
                json.dump({'anatomy_to_segmentation_target': {'breast': 'breast lesion'}}, handle)
            with (
                mock.patch.object(self.module, '_load_feature_prompts', return_value=prompts),
                mock.patch.dict(os.environ, {
                    'SONOREASON_FEATURE_PROMPTS_FILE': 'unused',
                    'SONOREASON_ANATOMY_TARGETS_FILE': targets,
                }, clear=True),
            ):
                rows = self.dataset._build_mask_measurement_data(source).set_index('feature')

        _, canvas_mask = ast.literal_eval(rows.loc['shape', 'image'])
        canvas_mask = Image.open(io.BytesIO(base64.b64decode(canvas_mask)))
        x0, y0, x1, y1 = canvas_mask.getbbox()
        self.assertEqual(canvas_mask.size, (896, 896))
        self.assertAlmostEqual((x1 - x0) / (y1 - y0), 2.0, places=2)
        shape = json.loads(rows.loc['shape', 'ground_truth_measurements'])
        orientation = json.loads(rows.loc['orientation', 'ground_truth_measurements'])
        self.assertAlmostEqual(shape['long_axis'], 20.0 * 896 / 40)
        self.assertEqual(orientation['orientation_degrees'], 10.0)


if __name__ == '__main__':
    unittest.main()
