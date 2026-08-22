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
