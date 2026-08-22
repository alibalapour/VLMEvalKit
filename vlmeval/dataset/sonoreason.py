import os, re, pandas as pd
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
    if strategy is None:
        strategy = os.environ.get('SONOREASON_PROMPT_STRATEGY', 'zero_shot')
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

    def load_data(self, dataset):
        import os, pandas as pd
        data_root = os.environ.get('LMUData', os.path.expanduser('~/LMUData'))
        path = os.path.join(data_root, f'{dataset}.tsv')
        df = pd.read_csv(path, sep='\t')

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
        return df

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt = self.dump_image(line)              # decodes base64 -> image path(s)
        msgs = []
        if isinstance(tgt, list):
            msgs += [dict(type='image', value=p) for p in tgt]
        else:
            msgs.append(dict(type='image', value=tgt))

        prompt_strategy = os.environ.get('SONOREASON_PROMPT_STRATEGY', 'zero_shot')
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

    def evaluate(self, eval_file, **kwargs):
        from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                                     balanced_accuracy_score)

        data = load(eval_file)
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
