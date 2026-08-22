import torch
from PIL import Image
from .base import BaseModel
from .sonoreason_gen import default_gen_kwargs

class MistralSmall(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True          # supports interleaved image+text

    def __init__(self, model_path='mistralai/Mistral-Small-3.1-24B-Instruct-2503', **kwargs):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map='auto', low_cpu_mem_usage=True).eval()
        # See sonoreason_gen.py -- same tag-banning pathology as the other two
        # adapters, milder here only because this model formats more reliably.
        self.gen_kwargs = default_gen_kwargs(tokenizer=self.processor.tokenizer, **kwargs)

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
        in_len = inputs['input_ids'].shape[-1]
        with torch.inference_mode():
            out = self.model.generate(**inputs, **self.gen_kwargs)
        return self.processor.decode(out[0][in_len:], skip_special_tokens=True).strip()
