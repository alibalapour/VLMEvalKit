"""Shared generation defaults for the SonoReason adapters.

Background -- why this file exists
----------------------------------
The MedGemma / LLaVA-Med / Mistral adapters originally each hardcoded

    dict(max_new_tokens=1024, do_sample=False,
         repetition_penalty=1.3, no_repeat_ngram_size=3)

`no_repeat_ngram_size=3` bans any 3-gram that already occurs in the *full*
sequence -- and HF counts the prompt, not just the generated tokens. The
reasoning prompt contains the literals `<answer>`, `</answer>`, `<reasoning>`
and `</reasoning>`, so the exact tags the evaluator parses for were the tokens
the decoder was forbidden to produce. Measured consequences:

  * LLaVA-Med (Mistral tokenizer): `<answer>` -> ['<','answer','>'], and that
    trigram is present in the prompt => permanently banned. Every reasoning
    output was the 23-char fragment `<reasoning><</reasoning` (the trailing
    '>' is banned via the ['reason','ing','>'] trigram). 100% parse failure.
  * MedGemma: emitted `<answer>` in only 20.7% of rows while emitting
    `</answer>` in 84.7% -- trying to close a tag it was never allowed to open.

The penalty was not gratuitous, though: with the constraints removed, ~45% of
MedGemma's reasoning generations run away in a verbatim repetition loop and hit
the token cap without ever answering. So the fix is not "drop the penalty" but
"replace it with mechanisms that don't collide with the output format":

  * `stop_strings=['</answer>']` -- terminate as soon as the answer is written,
    so anything the model would ramble afterwards costs nothing.
  * a mild `repetition_penalty` (default 1.05, well below the original 1.3),
    which discourages loops without materially suppressing the tag tokens.
  * a `max_new_tokens` cap as a backstop.

Every value is overridable per-run via SONOREASON_GEN_* environment variables
so the decoding config itself can be ablated without editing code.
"""
from __future__ import annotations

import os

# Emitted by the reasoning prompt template; generation can stop as soon as the
# answer tag closes. Harmless for zero_shot, which never produces these.
DEFAULT_STOP_STRINGS = ['</answer>']


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == '':
        return default
    return float(raw)


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == '':
        return default
    return int(raw)


def default_gen_kwargs(tokenizer=None, **overrides):
    """Build the generation kwargs shared by all SonoReason model adapters.

    Env overrides (all optional):
      SONOREASON_GEN_MAX_NEW_TOKENS   (default 768)
      SONOREASON_GEN_REPETITION_PENALTY (default 1.05; 1.0 disables)
      SONOREASON_GEN_NO_REPEAT_NGRAM  (default 0 = disabled -- do NOT set this
                                       to a small value, see module docstring)
      SONOREASON_GEN_STOP_STRINGS     comma-separated; empty string disables

    `tokenizer` is required by HF whenever `stop_strings` is used; if it is not
    supplied the stop-string mechanism is silently skipped rather than raising,
    so this stays safe for callers that do not have one handy.
    """
    # The structured arm emits a ~45-field JSON object before its answer tag,
    # which does not fit in the 768 the reasoning arm needs. Truncation here is
    # not a visible error -- it lands as a parse failure scored 0 -- so the
    # default is raised for that strategy rather than left to be set by hand.
    strategy = os.environ.get('SONOREASON_PROMPT_STRATEGY', 'zero_shot')
    # The feature arm asks for one small JSON object per call (3-5 measurements,
    # a label, a short reasoning string) but has no stop string that can fire --
    # its output ends with '}', which appears mid-object too. Left at the
    # reasoning default every one of the 7 calls per image would run to the cap,
    # so the budget is set to what the schema actually needs.
    if strategy == 'feature':
        default_max_new = 384
    else:
        default_max_new = 1600 if strategy == 'structured' else 768

    kwargs = dict(
        max_new_tokens=_env_int('SONOREASON_GEN_MAX_NEW_TOKENS', default_max_new),
        do_sample=False,
    )

    rep = _env_float('SONOREASON_GEN_REPETITION_PENALTY', 1.05)
    if rep and rep != 1.0:
        kwargs['repetition_penalty'] = rep

    # Disabled by default. Kept configurable only so the pathology documented
    # above can be reproduced deliberately in an ablation.
    ngram = _env_int('SONOREASON_GEN_NO_REPEAT_NGRAM', 0)
    if ngram and ngram > 0:
        kwargs['no_repeat_ngram_size'] = ngram

    raw_stop = os.environ.get('SONOREASON_GEN_STOP_STRINGS')
    if raw_stop is None:
        # Only the reasoning arm can ever emit </answer>, and HF's
        # StopStringCriteria costs a string check on every decode step. Attaching
        # it to zero_shot would be pure overhead (measured ~4x slower) for a
        # stop condition that can never fire.
        stop = list(DEFAULT_STOP_STRINGS) if strategy in ('reasoning', 'structured') else []
    else:
        stop = [s for s in (part.strip() for part in raw_stop.split(',')) if s]
    if stop and tokenizer is not None:
        kwargs['stop_strings'] = stop
        kwargs['tokenizer'] = tokenizer

    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# Assistant-turn prefill.
#
# Why this exists -- measured on the structured arm (run 55209947, 2026-08-17):
# LLaVA-Med produced no JSON on any of its 24 rows. Every output was instead a
# future-tense paraphrase of the output contract it had just been given:
#
#   "The output will be a single JSON object containing the various
#    measurements and labels derived from the ultrasound image. The JSON object
#    will be followed by the final decision, which is either "benign" or
#    "malignant.""
#
# This is not truncation and not a plumbing fault. The generations terminate on
# their own at a mean of 287 characters against a 1600-token cap, the model's
# chat template is the correct mistral_instruct '[INST] <image>\n... [/INST]',
# and its context window is 32k against a ~1k-token prompt. The model completes
# a full, fluent turn -- it simply answers in the wrong register.
#
# The cause is the instruction-tuning distribution. LLaVA-Med v1.5 is tuned on
# short conversational biomedical QA over figure captions, so a prompt whose
# final section *describes* an output format has, as its highest-likelihood
# continuation, a paraphrase of that description rather than the artifact. The
# model never enters "emit the object" register because prose about the task is
# what its tuning data looks like.
#
# Prefilling the assistant turn with the opening brace removes the choice: with
# '{' already in the context, "The output will be..." is not a reachable
# continuation. Same mechanism as medgemma.py's thought-channel prefill, with
# one difference -- that prefill is scaffolding and gets sliced off, whereas
# this one is the first character of the intended answer and is prepended back
# onto the decoded text so the parser receives a complete object.
#
# Whether this is enough is an empirical question -- it forces the format but
# cannot create measurement ability the model lacks. It is ablatable per-run
# via SONOREASON_JSON_PREFILL precisely so the "did the format barrier move?"
# question can be asked separately from "are the numbers any good?".
# ---------------------------------------------------------------------------
DEFAULT_JSON_PREFILL = '{'

# Arms whose output contract is a JSON object. zero_shot and reasoning emit
# prose and tags, so a brace prefill there would corrupt every response.
JSON_CONTRACT_STRATEGIES = ('structured', 'feature')


def json_prefill_text():
    """The assistant-turn prefill string for the current strategy, or ''.

    SONOREASON_JSON_PREFILL overrides:
      unset            -> '{' on the structured/feature arms, '' elsewhere
      ''               -> disabled, model formats freely
      any other string -> used verbatim
    """
    raw = os.environ.get('SONOREASON_JSON_PREFILL')
    if raw is not None:
        return raw
    strategy = os.environ.get('SONOREASON_PROMPT_STRATEGY', 'zero_shot')
    return DEFAULT_JSON_PREFILL if strategy in JSON_CONTRACT_STRATEGIES else ''


def resolve_prefill(tokenizer, prefill=None):
    """Tokenize the prefill, returning (ids, text_actually_encoded).

    The text is taken back out of the tokenizer rather than trusted as given:
    sentencepiece round-trips are not always exact, and the string prepended to
    the output has to be what the model actually saw, or the reassembled JSON
    silently disagrees with the context that produced it.

    Returns ([], '') when there is no prefill, so callers can treat the feature
    as off with a plain falsiness check.
    """
    if prefill is None:
        prefill = json_prefill_text()
    if not prefill:
        return [], ''
    ids = tokenizer(prefill, add_special_tokens=False)['input_ids']
    if not ids:
        return [], ''
    return ids, tokenizer.decode(ids)


def apply_prefill(inputs, prefill_ids):
    """Append prefill token ids to a tokenized batch, in place.

    Extends attention_mask to match. Returns the new input length so the caller
    can slice the generation correctly.
    """
    import torch
    ids = inputs['input_ids']
    pre = torch.tensor([prefill_ids], device=ids.device, dtype=ids.dtype)
    inputs['input_ids'] = torch.cat([ids, pre], dim=-1)
    if 'attention_mask' in inputs:
        inputs['attention_mask'] = torch.cat(
            [inputs['attention_mask'], torch.ones_like(pre)], dim=-1)
    return inputs['input_ids'].shape[-1]
