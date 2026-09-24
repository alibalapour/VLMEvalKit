import importlib.util
import json
import logging
import os
import sys
import types
import unittest
from unittest import mock

import pandas as pd


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


if __name__ == '__main__':
    unittest.main()
