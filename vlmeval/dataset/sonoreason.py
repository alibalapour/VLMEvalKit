import ast
import json
import os
import re
from pathlib import Path

import pandas as pd
from json_repair import loads as load_json_repair

from .image_base import ImageBaseDataset
from ..smp import load, dump


FEATURE_SPECS = {
    'shape': {
        'heading': 'SHAPE', 'label': 'lesion_shape_label',
        'measurements': {
            'long_axis': 'long_axis_pixels', 'short_axis': 'short_axis_pixels',
            'axis_difference_percent': 'long_short_difference_percent'},
        'thresholds': {'oval_threshold_percent': 'shape_oval_threshold_percent'},
    },
    'orientation': {
        'heading': 'ORIENTATION', 'label': 'lesion_orientation_label',
        'measurements': {'orientation_degrees': 'orientation_degrees'},
        'thresholds': {'parallel_threshold_degrees': 'orientation_parallel_threshold_degrees'},
    },
    'margin': {
        'heading': 'MARGIN', 'label': 'lesion_margin_label',
        'measurements': {
            'short_axis': 'short_axis_pixels',
            'boundary_contrast_proxy': 'margin_boundary_contrast',
            'local_noise_estimate': 'margin_local_noise_estimate',
            'contrast_to_noise_proxy': 'margin_contrast_to_noise'},
        'thresholds': {'circumscribed_threshold': 'margin_circumscribed_threshold'},
    },
    'border': {
        'heading': 'BORDER MORPHOLOGY', 'label': 'lesion_border_label',
        'measurements': {
            'solidity_proxy': 'border_solidity',
            'perimeter_to_hull_ratio_proxy': 'border_perimeter_to_hull_ratio'},
        'thresholds': {
            'smooth_solidity_threshold': 'border_smooth_solidity_threshold',
            'smooth_perimeter_ratio_threshold': 'border_smooth_perimeter_ratio_threshold',
            'spiky_perimeter_ratio_threshold': 'border_spiky_perimeter_ratio_threshold'},
    },
    'echo_pattern': {
        'heading': 'ECHO PATTERN', 'label': 'lesion_echo_pattern_label',
        'measurements': {
            'short_axis': 'short_axis_pixels',
            'lesion_core_median': 'echo_lesion_median',
            'echo_reference_median': 'echo_reference_median',
            'lesion_to_reference_ratio_proxy': 'echo_lesion_to_reference_ratio',
            'lesion_iqr': 'echotexture_lesion_iqr'},
        'thresholds': {
            'anechoic_ratio_threshold': 'echo_anechoic_ratio_threshold',
            'anechoic_iqr_threshold': 'echo_anechoic_iqr_threshold',
            'hypoechoic_ratio_threshold': 'echo_hypoechoic_ratio_threshold',
            'hyperechoic_ratio_threshold': 'echo_hyperechoic_ratio_threshold'},
    },
    'echotexture': {
        'heading': 'ECHOTEXTURE', 'label': 'lesion_echotexture_label',
        'measurements': {
            'lesion_iqr': 'echotexture_lesion_iqr',
            'echo_reference_median': 'echo_reference_median',
            'lesion_iqr_to_reference_ratio_proxy': 'echotexture_lesion_iqr_to_reference_ratio'},
        'thresholds': {
            'heterogeneous_iqr_ratio_threshold': 'echotexture_heterogeneous_iqr_ratio_threshold'},
    },
    'posterior_feature': {
        'heading': 'POSTERIOR FEATURE', 'label': 'lesion_posterior_feature_label',
        'measurements': {
            'posterior_to_reference_ratio_proxy': 'posterior_to_reference_ratio',
            'shadowing_fraction': 'posterior_shadowing_fraction',
            'enhancement_fraction': 'posterior_enhancement_fraction'},
        'thresholds': {
            'posterior_shadowing_threshold': 'posterior_shadowing_threshold',
            'posterior_enhancement_threshold': 'posterior_enhancement_threshold',
            'posterior_combined_min_fraction': 'posterior_combined_min_fraction',
            'posterior_combined_min_run_columns': 'posterior_combined_min_run_columns'},
    },
}


def _load_feature_prompts(path):
    text = Path(path).read_text()
    pattern = re.compile(
        r'^=+\n\d+\.\s+(.+?) PROMPT\n=+\n\n(.*?)(?=^=+\n\d+\.|\Z)',
        flags=re.MULTILINE | re.DOTALL,
    )
    by_heading = {heading.strip(): prompt.strip() for heading, prompt in pattern.findall(text)}
    prompts = {}
    for feature, spec in FEATURE_SPECS.items():
        heading = spec['heading']
        if heading not in by_heading:
            raise ValueError(f'Missing {heading!r} section in breast feature prompt file: {path}')
        prompts[feature] = by_heading[heading]
    return prompts


def _json_scalar(value):
    if pd.isna(value):
        return None
    if hasattr(value, 'item'):
        value = value.item()
    return value


class SonoReasonDD(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {'sonoreason_dd_breast': ''}   # local TSV in LMUData
    DATASET_MD5 = {'sonoreason_dd_breast': ''}
    PROMPT_COLUMNS = {
        'direct_diagnosis': 'direct_prompt',
        'reasoning_diagnosis': 'reasoning_prompt',
    }

    def _build_feature_data(self, data):
        prompt_file = os.environ.get('SONOREASON_FEATURE_PROMPTS_FILE')
        if not prompt_file:
            raise ValueError(
                'feature_diagnosis requires SONOREASON_FEATURE_PROMPTS_FILE')
        prompts = _load_feature_prompts(prompt_file)

        required = {'img_data', 'anatomy_location', 'bbox'}
        for spec in FEATURE_SPECS.values():
            required.add(spec['label'])
            required.update(spec['measurements'].values())
            required.update(spec['thresholds'].values())
        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f'SonoReason TSV is missing feature-diagnosis columns: {sorted(missing)}')

        data = data[
            data['anatomy_location'].astype(str).str.lower().eq('breast')
        ].reset_index(drop=True)
        rows = []
        for source_index, source in data.iterrows():
            primary_image_index = f'{source_index}__shape'
            for feature, spec in FEATURE_SPECS.items():
                label = _json_scalar(source[spec['label']])
                if label is None:
                    continue
                measurements = {
                    target: _json_scalar(source[column])
                    for target, column in spec['measurements'].items()
                    if _json_scalar(source[column]) is not None
                }
                if feature == 'posterior_feature':
                    try:
                        _, _, width, height = ast.literal_eval(str(source['bbox']))
                        measurements['bounding_box_width'] = _json_scalar(width)
                        measurements['bounding_box_height'] = _json_scalar(height)
                    except (TypeError, ValueError, SyntaxError):
                        pass
                thresholds = {
                    target: _json_scalar(source[column])
                    for target, column in spec['thresholds'].items()
                    if _json_scalar(source[column]) is not None
                }
                rows.append({
                    'index': f'{source_index}__{feature}',
                    'source_index': source_index,
                    'patient_id': _json_scalar(source.get('patient_id')),
                    'dataset_name': _json_scalar(source.get('dataset_name')),
                    'anatomy_location': 'breast',
                    'feature': feature,
                    # Store base64 once per source image. Other feature rows use
                    # VLMEvalKit's short-index image reference mechanism.
                    'image': (
                        source['img_data'] if feature == 'shape' else primary_image_index),
                    'question': prompts[feature],
                    'answer': str(label),
                    'ground_truth_measurements': json.dumps(measurements),
                    'ground_truth_thresholds': json.dumps(thresholds),
                })
        if not rows:
            raise ValueError('No labeled breast rows are available for feature_diagnosis')
        return pd.DataFrame(rows)

    def load_data(self, dataset):
        data_root = os.environ.get('LMUData', os.path.expanduser('~/LMUData'))
        relative_path = os.environ.get('SONOREASON_DATASET_FILE', f'{dataset}.tsv')
        path = os.path.join(data_root, relative_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f'SonoReason dataset file not found: {path}. '
                'Set LMUData to the dataset root and SONOREASON_DATASET_FILE '
                'to the relative TSV path.'
            )

        prompt_strategy = os.environ.get(
            'SONOREASON_PROMPT_STRATEGY', 'direct_diagnosis')
        data = pd.read_csv(path, sep='\t')
        if prompt_strategy == 'feature_diagnosis':
            return self._build_feature_data(data)
        if prompt_strategy not in self.PROMPT_COLUMNS:
            choices = ', '.join(sorted([*self.PROMPT_COLUMNS, 'feature_diagnosis']))
            raise ValueError(
                f'Unsupported SonoReason prompt strategy {prompt_strategy!r}. '
                f'Choose one of: {choices}.'
            )

        prompt_column = self.PROMPT_COLUMNS[prompt_strategy]
        required_columns = {'img_data', prompt_column, 'class_label'}
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(
                f'SonoReason TSV is missing required columns: {sorted(missing_columns)}')

        # Map the published SonoReason schema to VLMEvalKit's canonical fields.
        data = data.rename(columns={
            'img_data': 'image',
            prompt_column: 'question',
            'class_label': 'answer',
        })
        data['index'] = range(len(data))
        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt = self.dump_image(line)              # decodes base64 -> image path(s)
        msgs = []
        if isinstance(tgt, list):
            msgs += [dict(type='image', value=p) for p in tgt]
        else:
            msgs.append(dict(type='image', value=tgt))
        msgs.append(dict(type='text', value=line['question']))
        return msgs

    @staticmethod
    def _normalize_label(value):
        return str(value).strip().lower().replace('-', '_').replace(' ', '_')

    @staticmethod
    def _parse_feature_prediction(value):
        try:
            parsed = load_json_repair(str(value))
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _evaluate_features(self, data, eval_file):
        parsed_labels = []
        parse_ok = []
        hits = []
        row_maes = []
        row_measurement_counts = []
        measurement_records = []

        for _, row in data.iterrows():
            parsed = self._parse_feature_prediction(row['prediction'])
            valid = parsed is not None
            predicted_label = parsed.get('label') if valid else None
            parsed_labels.append(predicted_label)
            parse_ok.append(valid)
            hits.append(
                valid
                and self._normalize_label(predicted_label)
                == self._normalize_label(row['answer'])
            )

            truth = json.loads(row['ground_truth_measurements'])
            predicted_measurements = parsed.get('measurements', {}) if valid else {}
            if not isinstance(predicted_measurements, dict):
                predicted_measurements = {}
            errors = []
            for measurement, target in truth.items():
                predicted = predicted_measurements.get(measurement)
                try:
                    error = abs(float(predicted) - float(target))
                except (TypeError, ValueError):
                    continue
                errors.append(error)
                measurement_records.append({
                    'feature': row['feature'],
                    'measurement': measurement,
                    'absolute_error': error,
                })
            row_maes.append(sum(errors) / len(errors) if errors else None)
            row_measurement_counts.append(len(errors))

        data['predicted_label'] = parsed_labels
        data['prediction_parse_ok'] = parse_ok
        data['hit'] = hits
        data['measurement_mae'] = row_maes
        data['measurement_n'] = row_measurement_counts
        dump(data, eval_file.replace('.xlsx', '_parsed.xlsx'))

        details = pd.DataFrame(measurement_records)
        metric_rows = []
        for feature, group in data.groupby('feature', sort=False):
            feature_errors = details[details['feature'] == feature] if len(details) else details
            metric_rows.append({
                'feature': feature,
                'label_accuracy': group['hit'].mean(),
                'json_parse_rate': group['prediction_parse_ok'].mean(),
                'measurement_mae': (
                    feature_errors['absolute_error'].mean() if len(feature_errors) else None),
                'measurement_n': len(feature_errors),
                'n': len(group),
            })
        metric_rows.append({
            'feature': 'overall',
            'label_accuracy': data['hit'].mean(),
            'json_parse_rate': data['prediction_parse_ok'].mean(),
            'measurement_mae': details['absolute_error'].mean() if len(details) else None,
            'measurement_n': len(details),
            'n': len(data),
        })
        metrics = pd.DataFrame(metric_rows)
        dump(metrics, eval_file.replace('.xlsx', '_acc.csv'))
        if len(details):
            measurement_metrics = details.groupby(
                ['feature', 'measurement'], as_index=False)['absolute_error'].agg(['mean', 'count'])
            dump(measurement_metrics, eval_file.replace('.xlsx', '_measurement_mae.csv'))
        return metrics

    def evaluate(self, eval_file, **kwargs):
        data = load(eval_file)
        if 'feature' in data.columns:
            return self._evaluate_features(data, eval_file)

        CLASSES = ['malignant', 'benign', 'normal']
        def norm(x):
            s = str(x).lower()
            for c in CLASSES:
                if c in s: return c
            return s.strip()
        data['hit'] = [norm(p) == norm(a) for p, a in zip(data['prediction'], data['answer'])]
        acc = data['hit'].mean()
        res = pd.DataFrame([{'accuracy': acc, 'n': len(data)}])
        dump(res, eval_file.replace('.xlsx', '_acc.csv'))
        return res
