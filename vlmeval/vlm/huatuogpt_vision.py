import torch
from PIL import Image
from .base import BaseModel
from .sonoreason_gen import apply_prefill, default_gen_kwargs, resolve_prefill

# HuatuoGPT-Vision via its authors' transformers-native conversions
# (FreedomIntelligence/HuatuoGPT-Vision-{7B,34B}-hf: LLaVA on Qwen2-7B / Yi-34B,
# both with CLIP-L/336). Both use the same prompt format and '<|endoftext|>' EOS.
# Prompt format, image padding and EOS follow upstream cli.py
# (https://github.com/FreedomIntelligence/HuatuoGPT-Vision), which is the
# reference inference path for the original checkpoint.


def _expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    side = max(width, height)
    result = Image.new(pil_img.mode, (side, side), background_color)
    result.paste(pil_img, ((side - width) // 2, (side - height) // 2))
    return result


class HuatuoGPTVision(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True          # supports interleaved image+text

    # Upstream cli.py caps a turn at 6 images.
    MAX_IMAGES = 6

    def __init__(self, model_path='FreedomIntelligence/HuatuoGPT-Vision-7B-hf', **kwargs):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_path)
        # The 7B -hf repo's processor_config.json has patch_size=null (the 34B
        # ships none, so the same defaults apply), which leaves
        # '<image>' unexpanded and the model then finds 1 image token where it
        # expects 576. CLIP-L/14 at 336px: 24*24 patches + CLS, CLS dropped by
        # the 'default' feature strategy -> 576 (= config.image_seq_length).
        self.processor.patch_size = 14
        self.processor.num_additional_image_tokens = 1
        self.processor.vision_feature_select_strategy = 'default'
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map='auto', low_cpu_mem_usage=True).eval()

        tokenizer = self.processor.tokenizer
        # Upstream sets pad = eos = <|endoftext|>; the repo's
        # generation_config.json carries a stale pad_token_id=32001.
        eos = tokenizer.convert_tokens_to_ids('<|endoftext|>')
        self.gen_kwargs = default_gen_kwargs(
            tokenizer=tokenizer, **{'eos_token_id': eos, 'pad_token_id': eos, **kwargs})
        self.prefill_ids, self.prefill_text = resolve_prefill(tokenizer)
        image_mean = self.processor.image_processor.image_mean
        self.pad_color = tuple(int(x * 255) for x in image_mean)

    def generate_inner(self, message, dataset=None):
        images, texts = [], []
        for msg in message:
            if msg['type'] == 'image':
                image = Image.open(msg['value']).convert('RGB')
                # Upstream pads to square (image_aspect_ratio='pad'); the CLIP
                # processor's centre crop is then a no-op and nothing is cut away.
                images.append(_expand2square(image, self.pad_color))
            elif msg['type'] == 'text':
                texts.append(msg['value'])
        if len(images) > self.MAX_IMAGES:
            raise ValueError(
                f'HuatuoGPT-Vision takes at most {self.MAX_IMAGES} images, got {len(images)}')

        # Upstream: one '<image>\n' per image, all before the text.
        question = '<image>\n' * len(images) + '\n'.join(texts).strip()
        prompt = f'<|user|>\n{question}\n<|assistant|>\n'
        inputs = self.processor(
            images=images or None, text=prompt, return_tensors='pt'
        ).to(self.model.device)
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
        if self.prefill_ids:
            in_len = apply_prefill(inputs, self.prefill_ids)
        else:
            in_len = inputs['input_ids'].shape[-1]
        with torch.inference_mode():
            out = self.model.generate(**inputs, **self.gen_kwargs)
        text = self.processor.decode(out[0][in_len:], skip_special_tokens=True)
        return (self.prefill_text + text).strip()
