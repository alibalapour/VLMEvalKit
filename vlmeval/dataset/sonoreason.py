import ast
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from json_repair import loads as load_json_repair
from PIL import Image

from .image_base import ImageBaseDataset
from .sonoreason_prompts import resolve_prompt_template, build_prompt_variants
from ..smp import get_logger, load, dump

logger = get_logger(__name__)

# Metadata used to pick the right prompt template per dataset (see sonoreason_prompts.py).
# Only consulted for non-zero_shot prompt strategies -- zero_shot keeps using the
# `question` column exactly as before, so existing baseline runs are unaffected.
_BREAST_MALIGNANCY_METADATA = dict(
    keywords=['breast', 'ultrasound', 'malignancy', 'benign', 'malignant', 'normal'],
    modality='ultrasound',
    anatomy='breast',
)
# BI-RADS metadata, fine-grained (2/3/4A/4B/4C/5) -- resolves to breast_ultrasound_birads.
_BREAST_BIRADS_METADATA = dict(
    keywords=['breast', 'ultrasound', 'us', 'birads', 'bi-rads', 'acr', 'assessment'],
    modality='ultrasound',
    anatomy='breast',
)
# BI-RADS metadata, BUSBRA's coarse scheme (2/3/4/5) -- the extra 'busbra' keyword
# is what breaks the tie against _BREAST_BIRADS_METADATA and resolves to
# breast_ultrasound_birads_busbra instead (verified via resolve_prompt_template).
_BREAST_BIRADS_BUSBRA_METADATA = dict(
    keywords=['breast', 'ultrasound', 'us', 'birads', 'bi-rads', 'acr', 'assessment', 'busbra'],
    modality='ultrasound',
    anatomy='breast',
)

DATASET_PROMPT_METADATA = {
    'sonoreason_dd_breast': _BREAST_MALIGNANCY_METADATA,
    'sonoreason_dd_bus_cot': _BREAST_MALIGNANCY_METADATA,
    'sonoreason_dd_bus_uclm': _BREAST_MALIGNANCY_METADATA,
    'sonoreason_dd_bus_uc': _BREAST_MALIGNANCY_METADATA,
    'sonoreason_dd_open_access_breast': _BREAST_MALIGNANCY_METADATA,
    'sonoreason_dd_ultrasoundcases_breast': _BREAST_MALIGNANCY_METADATA,
    'sonoreason_dd_bus_cot_birads': _BREAST_BIRADS_METADATA,
    'sonoreason_dd_busbra_birads': _BREAST_BIRADS_BUSBRA_METADATA,
}

ANSWER_TAG_RE = re.compile(r'<answer>(.*?)</answer>', re.IGNORECASE | re.DOTALL)
REASONING_TAG_RE = re.compile(r'<reasoning>', re.IGNORECASE)
CLOSING_REASONING_RE = re.compile(r'</reasoning>', re.IGNORECASE)
# Structured arm only: the decision block's own label field, used when the
# generation ran out of tokens before it could close with an <answer> tag.
DECISION_LABEL_RE = re.compile(
    r'"decision"\s*:\s*\{[^{}]*?"label"\s*:\s*"([^"]*)"', re.IGNORECASE | re.DOTALL
)
# Marker that the response is structured-arm JSON. Its presence disables the
# last-resort whole-string scan: that scan returns the first label word it finds
# anywhere in the text, and a structured response mentions label words inside
# `evidence` strings, so scanning it would fabricate an answer from prose.
STRUCTURED_MARKER = '"feature_labels"'
PARSE_FAILED = '__parse_failed__'

# Gemma thought-channel delimiters (MedGemma-1.5). Registered special=False in
# the tokenizer, so skip_special_tokens=True leaves them in the prediction text
# and the trace has to be removed here instead. See vlm/medgemma.py.
THINK_OPEN = '<unused94>'
THINK_CLOSE = '<unused95>'


def strip_thinking(s):
    """Remove Gemma's thought channel before any label scanning.

    The trace is deliberation, not an answer, and it is dense with class words
    ("...so this leans benign, but the spiculated margin suggests malignant...").
    Scanning it fabricates answers the model never committed to, which is the
    same failure mode the STRUCTURED_MARKER guard below exists to prevent.

    An unclosed trace collapses to '' rather than to the raw text: a generation
    that hit the token cap mid-thought never produced an answer, so it must land
    as a parse failure instead of being mined for whichever class word appears.
    """
    if THINK_OPEN not in s:
        return s
    end = s.rfind(THINK_CLOSE)
    if end == -1:
        return ''
    return s[end + len(THINK_CLOSE):]

# Two strategy vocabularies are in use: the ablation-grid names in
# experiments/configs (zero_shot / reasoning / structured) and the names added
# alongside the feature arm (direct_diagnosis / reasoning_diagnosis /
# feature_diagnosis). Both are accepted and normalized here so neither set of
# configs breaks on the other's code. Collapse to one vocabulary before
# publication -- this map is a merge bridge, not a design.
STRATEGY_ALIASES = {
    'direct_diagnosis': 'zero_shot',
    'reasoning_diagnosis': 'reasoning',
    'feature_diagnosis': 'feature',
}


def canonical_strategy(value=None):
    """Normalize a prompt-strategy name, defaulting to the env var."""
    if value is None:
        value = os.environ.get('SONOREASON_PROMPT_STRATEGY', 'zero_shot')
    return STRATEGY_ALIASES.get(value, value)


# A run above this parse-failure rate is a formatting/decoding fault, not a
# model result -- the scores it produces are not comparable to a clean run.
# evaluate() logs a loud warning rather than raising, so a sweep still
# completes and every cell stays inspectable.
PARSE_FAIL_WARN_THRESHOLD = 0.10


def norm_label(x, valid_labels):
    """Normalize a ground-truth label (plain class string, no tags).

    valid_labels is the set of classes actually present in this dataset's
    ground truth (derived at evaluate()-time from the `answer` column), not a
    hardcoded vocabulary -- this is what lets the same function handle both
    the malignant/benign/normal datasets and the BI-RADS ones ('2'..'5',
    '4A'..'4C') without per-task special-casing.
    """
    s = str(x).lower().strip()
    for lb in valid_labels:
        if lb == s or lb in s:
            return lb
    return s


def extract_prediction(pred, valid_labels, strategy=None):
    """Extract the model's stated class from a raw completion.

    Resolution order, most trustworthy first:

    1. a well-formed <answer>...</answer> pair;
    2. the tail after the last </reasoning>, which is where the answer lands
       when the model emits a malformed tag (e.g. '</reasoning></answer>benign'
       or '</reasoning>**benign**'). This is still safe because the reasoning
       trace itself has been excluded;
    3. for a completion with no reasoning structure at all (the zero_shot
       case), a plain scan of the whole string.

    Step 2 exists because a strict reading of step 1 alone was discarding ~88%
    of MedGemma's reasoning rows and scoring them 0, which is what made
    reasoning look catastrophically worse than zero-shot. Note the trace text
    is never scanned: doing so picks up unrelated class words ("posterior
    acoustic features are normal") and silently corrupts accuracy, so a
    generation that loops until the token cap without answering still counts
    as a parse failure -- correctly, since it never gave an answer.

    `strategy` gates how far the fall-through is allowed to go; it defaults to
    SONOREASON_PROMPT_STRATEGY so in-run evaluation needs no plumbing, and can
    be passed explicitly when re-scoring an old result file offline.
    """
    strategy = canonical_strategy(strategy)
    s = strip_thinking(str(pred).lower())
    match = ANSWER_TAG_RE.search(s)
    if match:
        tag_text = match.group(1).strip()
        for lb in valid_labels:
            if lb == tag_text or lb in tag_text:
                return lb

    # Structured arm: the JSON carries the answer in decision.label, so a run
    # that was truncated before the closing <answer> tag is still recoverable
    # without falling through to the whole-string scan below.
    decision = DECISION_LABEL_RE.search(s)
    if decision:
        decision_text = decision.group(1).strip()
        for lb in valid_labels:
            if lb == decision_text or lb in decision_text:
                return lb
    if STRUCTURED_MARKER in s:
        return None

    # On the structured arm the answer is only ever trustworthy if it came from
    # the structured output itself -- the <answer> tag or decision.label, both
    # already tried above. Everything past this point mines free text, which on
    # this arm means mining a model that failed to produce the format at all.
    #
    # LLaVA-Med is why this gate is keyed on the strategy rather than on
    # STRUCTURED_MARKER being present: it emitted no JSON on any of its 1875
    # rows, only a description of the format it had been asked for ('...the
    # final decision will be either "benign" or "malignant."'). With no marker
    # to catch it, that text fell through to the whole-string scan below, which
    # walks valid_labels in sorted order and so returned 'benign' every time --
    # 923 invented predictions, an all-zero pred_malignant column,
    # parsed_balanced_accuracy of exactly 0.5, and parsed_accuracy 0.678 merely
    # restating the 0.676 majority baseline while reading as a result.
    #
    # Deliberately scoped to 'structured'. The zero_shot arm has no output
    # contract to fall back on -- free-text scanning is the only thing it can
    # do, and gating it there would rewrite every existing baseline.
    if strategy == 'structured':
        return None

    if REASONING_TAG_RE.search(s) or CLOSING_REASONING_RE.search(s):
        parts = CLOSING_REASONING_RE.split(s)
        if len(parts) > 1:
            tail = parts[-1]
            # last mention wins: the answer is the final thing written
            hits = [(tail.rfind(lb), lb) for lb in valid_labels if lb in tail]
            if hits:
                return max(hits)[1]
        return None

    for lb in valid_labels:
        if lb == s or lb in s:
            return lb
    return None



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
    # local TSVs in LMUData -- no MD5/URL since these aren't hosted the way
    # VLMEvalKit expects (load_data below reads straight from $LMUData)
    DATASET_URL = {
        'sonoreason_dd_breast': '',
        'sonoreason_dd_bus_cot': '',
        'sonoreason_dd_bus_uclm': '',
        'sonoreason_dd_bus_uc': '',
        'sonoreason_dd_open_access_breast': '',
        'sonoreason_dd_ultrasoundcases_breast': '',
        'sonoreason_dd_bus_cot_birads': '',
        'sonoreason_dd_busbra_birads': '',
    }
    DATASET_MD5 = {k: '' for k in DATASET_URL}

    _prompt_logged = False  # log the resolved prompt text once per run, not per-row
    # Canonical strategy -> TSV column carrying the pre-baked prompt. Only
    # consulted for datasets with no template registered above; build_prompt
    # prefers the registry (see the BI-RADS note there).
    PROMPT_COLUMNS = {
        'zero_shot': 'direct_prompt',
        'reasoning': 'reasoning_prompt',
    }

    def _build_segmentation_data(self, data):
        required = {
            'img_data', 'mask', 'segmentation_bbox_xyxy',
            'image_width', 'image_height', 'segmentation_prompt',
        }
        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f'SonoReason TSV is missing segmentation columns: {sorted(missing)}')

        # A source diagnosis table may contain only partial segmentation
        # coverage. Segmentation evaluation must never fail the whole run or,
        # worse, score a row with no ground truth. Keep only rows carrying an
        # encoded mask and report exactly how many were skipped. Intentionally
        # do not reject a decoded all-zero mask here: the published SEG tables
        # use those for verified normal/no-lesion examples.
        has_mask = data['mask'].notna() & data['mask'].astype(str).str.strip().ne('')
        skipped = int((~has_mask).sum())
        if skipped:
            logger.warning(
                f'[SonoReasonDD] skipping {skipped}/{len(data)} segmentation rows '
                'because no ground-truth mask is present'
            )
        data = data.loc[has_mask].reset_index(drop=True)
        if data.empty:
            raise ValueError(
                'No rows with segmentation masks remain. Select a file under SEG/; '
                'datasets without masks cannot be evaluated for segmentation.')

        rows = []
        for source_index, source in data.iterrows():
            width = int(source['image_width'])
            height = int(source['image_height'])
            sample_id = str(source.get('sample_id') or source.get('patient_id') or source_index)
            image_name = Path(sample_id).name
            if not Path(image_name).suffix:
                image_name += '.png'
            dataset_folder = re.sub(
                r'[^A-Za-z0-9_.-]+', '_', str(source.get('dataset_name') or 'unknown')
            ).strip('._') or 'unknown'
            question = (
                f"{source['segmentation_prompt']}\n"
                f'Original image dimensions: width={width}, height={height}.'
            )
            rows.append({
                'index': source_index,
                'patient_id': _json_scalar(source.get('patient_id')),
                'sample_id': sample_id,
                'dataset_name': _json_scalar(source.get('dataset_name')),
                'anatomy_location': _json_scalar(source.get('anatomy_location')),
                'feature': 'segmentation',
                'image': source['img_data'],
                'image_path': f'{dataset_folder}/{image_name}',
                'question': question,
                'segmentation_prompt': source['segmentation_prompt'],
                'segmentation_verification_prompt': _json_scalar(
                    source.get('segmentation_verification_prompt')),
                'segmentation_target': _json_scalar(source.get('segmentation_target')),
                'answer': source['segmentation_bbox_xyxy'],
                'ground_truth_mask': source['mask'],
                'ground_truth_bbox': source['segmentation_bbox_xyxy'],
                'image_width': width,
                'image_height': height,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _apply_sample_limit(data):
        raw_limit = os.environ.get('SONOREASON_SAMPLE_LIMIT', '').strip()
        if not raw_limit:
            return data
        try:
            limit = int(raw_limit)
        except ValueError as exc:
            raise ValueError(
                f'SONOREASON_SAMPLE_LIMIT must be an integer, got {raw_limit!r}'
            ) from exc
        if limit < 0:
            raise ValueError('SONOREASON_SAMPLE_LIMIT must be non-negative')
        if limit == 0:
            return data
        return data.head(limit).reset_index(drop=True)

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

        strategy = canonical_strategy()
        df = pd.read_csv(path, sep='\t')
        if strategy == 'segmentation':
            return self._apply_sample_limit(self._build_segmentation_data(df))
        if strategy == 'feature':
            return self._apply_sample_limit(self._build_feature_data(df))

        # Two TSV shapes are in play: the published SonoReason schema read
        # straight from u2_ext_data/data/DD/ (img_data/direct_prompt/class_label),
        # and the pre-built canonical TSVs under $LMUData. Reading the source
        # directly is preferred -- it keeps the mask-derived measurement and
        # lesion_*_label columns the structured/feature arms score against,
        # which the pre-built files drop -- but both are accepted so existing
        # $LMUData files keep working.
        if 'image' not in df.columns:
            prompt_column = self.PROMPT_COLUMNS.get(strategy, 'direct_prompt')
            required_columns = {'img_data', prompt_column, 'class_label'}
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(
                    f'SonoReason TSV is missing required columns: {sorted(missing_columns)}')
            df = df.rename(columns={
                'img_data': 'image',
                prompt_column: 'question',
                'class_label': 'answer',
            })
            df['index'] = range(len(df))

        # SONOREASON_LIMIT=<n> cuts the run down to a smoke test: enough rows to
        # tell whether the output format parses at all, cheap enough to turn
        # around in minutes. Sampling is stratified by `answer` and seeded, so
        # every class is represented (a plain head(n) on these files can return
        # a single class) and two models see byte-identical rows.
        #
        # A smoke run is a format check, never a result: n is far too small for
        # the per-class metrics to mean anything.
        limit = os.environ.get('SONOREASON_LIMIT')
        if limit:
            n = int(limit)
            if n < len(df):
                n_classes = max(1, df['answer'].nunique())
                per_class = max(1, n // n_classes)
                # groupby().head() keeps the file's original row order and does
                # not touch the `index` column VLMEvalKit keys its cache on.
                df = (df.groupby('answer', sort=True)
                        .head(per_class)
                        .sort_index()
                        .reset_index(drop=True))
                logger.warning(
                    f'[SonoReasonDD] SONOREASON_LIMIT={n} -> {len(df)} rows '
                    f'({per_class} x {n_classes} classes). SMOKE TEST: this is a '
                    f'format check, the metrics below are not results.'
                )
        return self._apply_sample_limit(df)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt = self.dump_image(line)              # decodes base64 -> image path(s)
        msgs = []
        if isinstance(tgt, list):
            msgs += [dict(type='image', value=p) for p in tgt]
        else:
            msgs.append(dict(type='image', value=tgt))

        # The feature arm builds a distinct per-feature prompt per row in
        # load_data. sonoreason_dd_breast has a template registered below, so
        # without this guard the registry would overwrite all seven of them.
        if 'feature' in line.index:
            text = line['question']
            if not self._prompt_logged:
                logger.info(
                    f'[SonoReasonDD] dataset={self.dataset_name} '
                    f'prompt_strategy={canonical_strategy()} -- prompt sent to the model '
                    f'(first sample):\n{text}'
                )
                self._prompt_logged = True
            msgs.append(dict(type='text', value=text))
            return msgs

        prompt_strategy = canonical_strategy()
        meta = DATASET_PROMPT_METADATA.get(self.dataset_name)
        if meta is not None:
            # Both arms are built from the same template so the two conditions
            # differ ONLY in reasoning format, never in the option list.
            #
            # This previously read line['question'] for zero_shot, which comes
            # from the upstream `direct_prompt` column. For the 6 malignancy
            # datasets that column is byte-identical to the template, but for
            # the two BI-RADS sets it is *swapped*: bus_cot_birads has
            # fine-grained ground truth (2/3/4A/4B/4C/5) while its question
            # column offered only coarse ['2','3','4','5'] -- making every
            # 4A/4B/4C row unwinnable -- and busbra_birads had the mirror
            # problem. That asymmetry, not reasoning, is what produced the
            # apparent reasoning "wins" on exactly those two datasets.
            _, template = resolve_prompt_template(
                keywords=meta.get('keywords', []),
                modality=meta.get('modality', ''),
                anatomy=meta.get('anatomy', ''),
            )
            variants = build_prompt_variants(template)
            text = variants.get(f'{prompt_strategy}_prompt', variants['direct_prompt'])
        else:
            # No template registered for this dataset -- fall back to the TSV.
            logger.warning(
                f'[SonoReasonDD] no prompt template registered for {self.dataset_name}; '
                f'falling back to the TSV question column (prompt_strategy={prompt_strategy} '
                f'will have no effect)'
            )
            text = line['question']

        if not self._prompt_logged:
            logger.info(
                f'[SonoReasonDD] dataset={self.dataset_name} prompt_strategy={prompt_strategy} '
                f'-- prompt sent to the model (first sample, identical for the rest of the run):\n{text}'
            )
            self._prompt_logged = True

        msgs.append(dict(type='text', value=text))
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

    @staticmethod
    def _parse_segmentation_prediction(value, width, height):
        parsed = SonoReasonDD._parse_feature_prediction(value)
        if parsed is None or not parsed.get('lesion_present', True):
            return None
        bbox = parsed.get('bbox')
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            x0, y0, x1, y1 = [float(x) for x in bbox]
        except (TypeError, ValueError):
            return None
        x0, x1 = sorted((
            min(max(0.0, x0), float(width - 1)),
            min(max(0.0, x1), float(width - 1)),
        ))
        y0, y1 = sorted((
            min(max(0.0, y0), float(height - 1)),
            min(max(0.0, y1), float(height - 1)),
        ))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        return [x0, y0, x1, y1]

    @staticmethod
    @lru_cache(maxsize=2)
    def _load_segmentor(model_name, device):
        import torch
        from transformers import SamModel, SamProcessor

        processor = SamProcessor.from_pretrained(model_name)
        model = SamModel.from_pretrained(model_name).to(device)
        model.eval()
        return processor, model

    @staticmethod
    def _segment_with_box(image, bbox):
        import torch

        requested_device = os.environ.get('SONOREASON_SEGMENTOR_DEVICE', 'auto')
        if requested_device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = requested_device
        model_name = os.environ.get(
            'SONOREASON_SEGMENTOR_MODEL', 'wanglab/medsam-vit-base')
        processor, model = SonoReasonDD._load_segmentor(model_name, device)
        inputs = processor(images=image, input_boxes=[[bbox]], return_tensors='pt')
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        is_medsam = 'medsam' in model_name.lower()
        with torch.inference_mode():
            outputs = model(**inputs, multimask_output=not is_medsam)
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs['original_sizes'].cpu(),
            inputs['reshaped_input_sizes'].cpu(),
        )[0]
        scores = outputs.iou_scores.detach().cpu().reshape(-1)
        best = int(torch.argmax(scores).item())
        candidates = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
        return candidates[best].numpy() > 0, float(scores[best].item()), model_name, device

    @staticmethod
    def _compute_monai_metrics(prediction, target):
        import torch
        from monai.metrics import (
            DiceMetric, HausdorffDistanceMetric, MeanIoU,
            compute_confusion_matrix_metric, get_confusion_matrix,
        )

        pred = torch.as_tensor(prediction.astype(bool))
        truth = torch.as_tensor(target.astype(bool))
        y_pred = torch.stack((~pred, pred), dim=0).unsqueeze(0).float()
        y = torch.stack((~truth, truth), dim=0).unsqueeze(0).float()

        dice = DiceMetric(include_background=False, reduction='mean', ignore_empty=False)
        iou = MeanIoU(include_background=False, reduction='mean', ignore_empty=False)
        hausdorff95 = HausdorffDistanceMetric(
            include_background=False, percentile=95, reduction='mean')
        dice(y_pred, y)
        iou(y_pred, y)
        hausdorff95(y_pred, y)
        confusion = get_confusion_matrix(y_pred, y, include_background=False)

        def scalar(value):
            return float(torch.as_tensor(value).detach().cpu().nanmean().item())

        return {
            'dice': scalar(dice.aggregate()),
            'iou': scalar(iou.aggregate()),
            'precision': scalar(compute_confusion_matrix_metric('precision', confusion)),
            'recall_sensitivity': scalar(
                compute_confusion_matrix_metric('sensitivity', confusion)),
            'specificity': scalar(
                compute_confusion_matrix_metric('specificity', confusion)),
            'pixel_accuracy': scalar(
                compute_confusion_matrix_metric('accuracy', confusion)),
            'hausdorff_distance_95_pixels': scalar(hausdorff95.aggregate()),
        }

    def _evaluate_segmentation(self, data, eval_file):
        source_by_index = {
            str(row['index']): row for _, row in self.data.iterrows()
        }
        output_dir = Path(eval_file).with_suffix('').with_name(
            Path(eval_file).stem + '_predicted_masks')
        output_dir.mkdir(parents=True, exist_ok=True)
        records = []

        for _, result in data.iterrows():
            index = str(result['index'])
            source = source_by_index.get(index)
            record = {
                'index': index,
                'patient_id': result.get('patient_id'),
                'prediction_parse_ok': False,
                'segmentation_success': False,
                'error': None,
            }
            if source is None:
                record['error'] = f'No source row for index {index}'
                records.append(record)
                continue
            try:
                from ..smp.vlm import decode_base64_to_image

                truth_image = decode_base64_to_image(
                    source['ground_truth_mask']).convert('L')
                truth = np.asarray(truth_image) > 0
                agent_result = self._parse_feature_prediction(result['prediction'])
                if agent_result and 'mask_path' in agent_result:
                    mask_path = Path(agent_result['mask_path'])
                    if agent_result.get('status') != 'completed' or not mask_path.is_file():
                        raise ValueError(
                            f"Agent mask unavailable: status={agent_result.get('status')}, "
                            f"path={mask_path}")
                    prediction = np.asarray(Image.open(mask_path).convert('L')) > 0
                    bbox = agent_result.get('bbox')
                    confidence = agent_result.get('segmentor_confidence')
                    model_name = agent_result.get('segmentor_model')
                    device = agent_result.get('segmentor_device')
                    record.update({
                        'prediction_parse_ok': True,
                        'predicted_bbox': json.dumps(bbox),
                        'agent_accepted': agent_result.get('accepted'),
                        'agent_attempts': agent_result.get('attempts'),
                        'verification_error': agent_result.get('verification_error'),
                        'vlm_backend': agent_result.get('vlm_backend'),
                        'vlm_model': agent_result.get('vlm_model'),
                    })
                else:
                    # Backward-compatible path for older single-box predictions.
                    bbox = self._parse_segmentation_prediction(
                        result['prediction'], int(source['image_width']),
                        int(source['image_height']))
                    if bbox is None:
                        raise ValueError(
                            'Prediction contains neither an agent mask nor a valid bbox')
                    image = decode_base64_to_image(source['image']).convert('RGB')
                    prediction, confidence, model_name, device = self._segment_with_box(
                        image, bbox)
                    mask_path = output_dir / f'{source.get("patient_id", index)}.png'
                    Image.fromarray(prediction.astype(np.uint8) * 255).save(mask_path)
                    record.update({
                        'prediction_parse_ok': True,
                        'predicted_bbox': json.dumps(bbox),
                    })
                if prediction.shape != truth.shape:
                    raise ValueError(
                        f'Prediction shape {prediction.shape} != mask shape {truth.shape}')
                metrics = self._compute_monai_metrics(prediction, truth)
                record.update(metrics)
                record.update({
                    'segmentation_success': True,
                    'segmentor_confidence': confidence,
                    'segmentor_model': model_name,
                    'segmentor_device': device,
                    'predicted_mask_path': str(mask_path),
                })
            except Exception as exc:
                record['error'] = f'{type(exc).__name__}: {exc}'
            records.append(record)

        details = pd.DataFrame(records)
        dump(details, eval_file.replace('.xlsx', '_segmentation_details.csv'))
        metric_names = [
            'dice', 'iou', 'precision', 'recall_sensitivity', 'specificity',
            'pixel_accuracy', 'hausdorff_distance_95_pixels', 'segmentor_confidence',
        ]
        summary = {
            'prediction_parse_rate': details['prediction_parse_ok'].mean(),
            'segmentation_success_rate': details['segmentation_success'].mean(),
            'n': len(details),
            'successful_n': int(details['segmentation_success'].sum()),
        }
        for metric in metric_names:
            summary[metric] = details[metric].mean() if metric in details else None
        metrics = pd.DataFrame([summary])
        dump(metrics, eval_file.replace('.xlsx', '_segmentation_metrics.csv'))
        return metrics

    def evaluate(self, eval_file, **kwargs):
        data = load(eval_file)
        if 'feature' in data.columns and data['feature'].eq('segmentation').all():
            return self._evaluate_segmentation(data, eval_file)
        if 'feature' in data.columns:
            return self._evaluate_features(data, eval_file)

        from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                                     balanced_accuracy_score)

        valid_labels = sorted(set(str(a).lower().strip() for a in data['answer'].unique()))
        data['norm_answer'] = data['answer'].apply(lambda x: norm_label(x, valid_labels))
        data['norm_pred'] = data['prediction'].apply(lambda x: extract_prediction(x, valid_labels))
        data['parse_failed'] = data['norm_pred'].isna()
        data['hit'] = data['norm_pred'] == data['norm_answer']

        n = len(data)
        n_parse_failed = int(data['parse_failed'].sum())
        acc = data['hit'].mean()

        y_true = data['norm_answer']
        y_pred = data['norm_pred'].fillna(PARSE_FAILED)
        labels = sorted(y_true.unique())  # ground-truth classes only -- excludes the parse-failure bucket

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average='macro', zero_division=0)

        # Prior-aware metrics. Plain accuracy is not interpretable on these
        # class-imbalanced sets: a model that always answers 'benign' scores
        # 0.63 on sonoreason_dd_breast while doing no diagnosis at all, so a
        # degenerate constant predictor can outscore a genuinely discriminating
        # one. majority_baseline is the score to beat; balanced accuracy over
        # the rows that actually parsed is the "when it answers, is it better
        # than chance?" number.
        majority_baseline = y_true.value_counts(normalize=True).max() if n else float('nan')
        parsed = data['norm_pred'].notna()
        n_parsed = int(parsed.sum())
        if n_parsed > 1:
            parsed_acc = float((data.loc[parsed, 'norm_pred'] == y_true[parsed]).mean())
            parsed_balanced_acc = float(
                balanced_accuracy_score(y_true[parsed], data.loc[parsed, 'norm_pred']))
            _, _, parsed_f1, _ = precision_recall_fscore_support(
                y_true[parsed], data.loc[parsed, 'norm_pred'],
                average='macro', zero_division=0)
        else:
            parsed_acc = parsed_balanced_acc = parsed_f1 = float('nan')

        pct_parse_failed = n_parse_failed / n if n else 0.0
        if pct_parse_failed > PARSE_FAIL_WARN_THRESHOLD:
            logger.warning(
                f'[SonoReasonDD] {self.dataset_name}: {pct_parse_failed:.1%} of rows '
                f'({n_parse_failed}/{n}) could not be parsed into a label. Scores below are '
                f'NOT comparable to a clean run -- a parse failure counts as wrong, so this '
                f'depresses accuracy independently of model quality. Check the generation '
                f'config (see vlmeval/vlm/sonoreason_gen.py) and a few raw predictions '
                f'before interpreting these numbers.'
            )

        res = pd.DataFrame([{
            'accuracy': acc,
            'n': n,
            'n_parse_failed': n_parse_failed,
            'pct_parse_failed': pct_parse_failed,
            'precision_macro': macro_p,
            'recall_macro': macro_r,
            'f1_macro': macro_f1,
            'majority_baseline': majority_baseline,
            'n_parsed': n_parsed,
            'parsed_accuracy': parsed_acc,
            'parsed_balanced_accuracy': parsed_balanced_acc,
            'parsed_f1_macro': parsed_f1,
        }])
        dump(res, eval_file.replace('.xlsx', '_acc.csv'))

        per_class = pd.DataFrame({
            'class': labels,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
        })
        dump(per_class, eval_file.replace('.xlsx', '_per_class.csv'))

        cm_labels = labels + ([PARSE_FAILED] if PARSE_FAILED in y_pred.values else [])
        cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
        cm_df = pd.DataFrame(cm, columns=[f'pred_{lb}' for lb in cm_labels])
        cm_df.insert(0, 'true_label', [f'true_{lb}' for lb in cm_labels])
        dump(cm_df, eval_file.replace('.xlsx', '_confusion.csv'))

        return res
