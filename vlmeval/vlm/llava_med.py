import torch
from PIL import Image
from .base import BaseModel
from .sonoreason_gen import apply_prefill, default_gen_kwargs, resolve_prefill

class LLaVAMed(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True          # supports interleaved image+text

    def __init__(self, model_path='chaoyinshe/llava-med-v1.5-mistral-7b-hf', **kwargs):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map='auto', low_cpu_mem_usage=True).eval()
        # See sonoreason_gen.py. This model was hit hardest by the old
        # no_repeat_ngram_size=3: with the Mistral tokenizer '<answer>' is the
        # trigram ['<','answer','>'], which occurs in the prompt and was
        # therefore banned outright -- every reasoning run scored exactly 0.
        self.gen_kwargs = default_gen_kwargs(tokenizer=self.processor.tokenizer, **kwargs)
        # This model answers a JSON request with prose *about* the JSON it was
        # asked for -- 0/24 objects on the structured arm. Prefilling the
        # assistant turn with '{' makes that continuation unreachable. See the
        # long note in sonoreason_gen.py.
        self.prefill_ids, self.prefill_text = resolve_prefill(self.processor.tokenizer)

    def generate_inner(self, message, dataset=None):
        content = []
        for msg in message:
            if msg['type'] == 'image':
                content.append({'type': 'image',
                                'image': Image.open(msg['value']).convert('RGB')})
            elif msg['type'] == 'text':
                content.append({'type': 'text', 'text': msg['value']})
        messages = [{'role': 'user', 'content': content}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors='pt'
        ).to(self.model.device, dtype=torch.bfloat16)
        if self.prefill_ids:
            in_len = apply_prefill(inputs, self.prefill_ids)
        else:
            in_len = inputs['input_ids'].shape[-1]
        with torch.inference_mode():
            out = self.model.generate(**inputs, **self.gen_kwargs)
        text = self.processor.decode(out[0][in_len:], skip_special_tokens=True)
        # The prefill is the first character of the answer, not scaffolding, so
        # it goes back on -- otherwise the parser gets an object with no opening
        # brace. (Contrast medgemma.py, where the prefill is dropped.)
        return (self.prefill_text + text).strip()
