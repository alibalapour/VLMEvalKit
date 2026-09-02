import os

import torch
from PIL import Image
from .base import BaseModel
from .sonoreason_gen import apply_prefill, default_gen_kwargs, resolve_prefill

# Gemma's thought-channel delimiters, as used by MedGemma-1.5. Both are
# registered with special=False in tokenizer_config.json, so decoding with
# skip_special_tokens=True does NOT remove them -- they land verbatim in the
# prediction string (see strip_thinking() in dataset/sonoreason.py).
THINK_OPEN_TOKEN = '<unused94>'
THINK_CLOSE_TOKEN = '<unused95>'

# Default prefill appended to the model turn to suppress the thought channel.
# Empty string disables suppression.
#
# Why this exists -- measured on run 55013062 (structured arm, 2026-08-16):
# MedGemma-1.5 opens the thought channel whenever the prompt asks for explicit
# multi-step work. The structured prompt's STAGE 1/2/3 cascade triggers it on
# 100% of rows (460/460 on dd_breast, 470/470 on busbra_birads), and the trace
# runs ~5.2k chars -- the entire max_new_tokens=1600 budget. Consequences:
#   * 0/460 rows ever reached decision.label or <answer>; the run would have
#     scored 100% parse failure even with unlimited wall clock;
#   * stop_strings=['</answer>'] could never fire, so every row ran to the token
#     cap: 45.9 s/it, ~24h projected against a 6h limit -> killed at 25%.
# The reasoning arm on the same model/adapter never opened the channel (0/1875)
# and parsed at 99.9%, so this is a structured-prompt-specific regression, not a
# standing property of the model.
#
# The chat template exposes no thinking toggle (no enable_thinking kwarg), so
# the only lever is prefilling the model turn with the closing delimiter.
DEFAULT_THINK_PREFILL = THINK_CLOSE_TOKEN


class MedGemma(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True          # supports interleaved image+text

    def __init__(self, model_path='google/medgemma-4b-it', **kwargs):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map='auto', low_cpu_mem_usage=True).eval()
        # See sonoreason_gen.py: the previous no_repeat_ngram_size=3 made the
        # <answer>/</answer> tags unemittable, which is what produced the ~88%
        # parse-failure rate on every reasoning run.
        self.gen_kwargs = default_gen_kwargs(tokenizer=self.processor.tokenizer, **kwargs)
        self.think_prefill_ids = self._resolve_think_prefill()
        # Composed after the thought-channel closer: '<unused95>' ends the
        # trace, '{' then opens the object. Order matters -- a brace before
        # the closer would land inside the trace and be stripped with it.
        self.prefill_ids, self.prefill_text = resolve_prefill(self.processor.tokenizer)

    def _resolve_think_prefill(self):
        """Token ids to append after the generation prompt, or [] to disable.

        Overridable per-run with SONOREASON_THINK_PREFILL so the suppression
        itself stays ablatable without a code edit:
          unset            -> '<unused95>' (suppress thinking)
          ''               -> disabled, model thinks freely
          any other string -> tokenized and used verbatim
        """
        raw = os.environ.get('SONOREASON_THINK_PREFILL')
        prefill = DEFAULT_THINK_PREFILL if raw is None else raw
        if not prefill:
            return []
        ids = self.processor.tokenizer(
            prefill, add_special_tokens=False)['input_ids']
        # A model whose vocabulary lacks these tokens (e.g. plain MedGemma-4B,
        # which is not a thinking model) tokenizes them into unrelated pieces;
        # only apply the prefill when it round-trips exactly.
        if self.processor.tokenizer.decode(ids) != prefill:
            return []
        return ids

    def generate_inner(self, message, dataset=None):
        content = []
        for msg in message:
            if msg['type'] == 'image':
                source = msg['value']
                image = (source.copy().convert('RGB') if isinstance(source, Image.Image)
                         else Image.open(source).convert('RGB'))
                content.append({'type': 'image', 'image': image})
            elif msg['type'] == 'text':
                content.append({'type': 'text', 'text': msg['value']})
        messages = [{'role': 'user', 'content': content}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors='pt'
        ).to(self.model.device, dtype=torch.bfloat16)

        # in_len is recomputed after each prefill so the slice below drops both:
        # they are scaffolding we supplied, not something the model produced.
        # The JSON prefill is the exception -- it is the first character of the
        # answer, so it is added back onto the decoded text.
        in_len = inputs['input_ids'].shape[-1]
        if self.think_prefill_ids:
            in_len = apply_prefill(inputs, self.think_prefill_ids)
        if self.prefill_ids:
            in_len = apply_prefill(inputs, self.prefill_ids)
        with torch.inference_mode():
            out = self.model.generate(**inputs, **self.gen_kwargs)
        text = self.processor.decode(out[0][in_len:], skip_special_tokens=True)
        return (self.prefill_text + text).strip()
