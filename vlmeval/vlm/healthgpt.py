import os
import sys

import torch
from PIL import Image
from .base import BaseModel
from .sonoreason_gen import default_gen_kwargs, resolve_prefill

# HealthGPT is not a transformers-native model: it is an H-LoRA adapter plus a
# LLaVA-style vision stack on top of Phi-4, assembled by the upstream repo's own
# code (https://github.com/DCDmllm/HealthGPT). That code is cloned, unmodified,
# under SonoReason/third_party/ (gitignored, like UltraSam).
DEFAULT_HEALTHGPT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    'third_party', 'HealthGPT', 'HealthGPT')

# Upstream's phi4_instruct template ships the system prompt "You are a medieval
# knight and must provide explanations to modern people." -- a leftover that
# would bias every answer's register. Replaced with a neutral one; overridable
# with SONOREASON_HEALTHGPT_SYSTEM (empty string keeps the upstream prompt).
DEFAULT_SYSTEM = 'You are a helpful medical assistant.'


def _add_special_tokens_and_resize_model(tokenizer, model, vq_idx_nums):
    """Upstream utils.add_special_tokens_and_resize_model, for transformers 5.

    Upstream reads tokenizer.additional_special_tokens, which v5 removed. Same
    tokens in the same order (so they land on the same ids, appended after
    Phi-4's 100352-row embedding) and the same mean initialisation. The H-LoRA
    checkpoint carries no embedding rows, so these only need to exist for the
    model shape to match.
    """
    tokens = (['<start_index>'] + [f'<idx_{i}>' for i in range(vq_idx_nums)]
              + ['<end_index>', '<pixel_newline>'])
    num_new_tokens = tokenizer.add_tokens(tokens, special_tokens=True)
    # mean_resizing (default since transformers 4.46) fits a 5120-dim Gaussian to
    # the whole embedding matrix on CPU -- minutes of work that the plain-mean
    # overwrite below discards anyway. Upstream (4.41) predates it.
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    if num_new_tokens > 0:
        for emb in (model.get_input_embeddings().weight.data,
                    model.get_output_embeddings().weight.data):
            emb[-num_new_tokens:] = emb[:-num_new_tokens].mean(dim=0, keepdim=True)
    return num_new_tokens


class HealthGPT(BaseModel):
    INSTALL_REQ = True         # needs the upstream repo, see DEFAULT_HEALTHGPT_ROOT
    INTERLEAVE = True          # supports interleaved image+text

    def __init__(self, model_path='microsoft/phi-4',
                 hlora_repo='lintw/HealthGPT-L14',
                 hlora_file='com_hlora_weights_phi4.bin',
                 vit_path='openai/clip-vit-large-patch14-336',
                 hlora_r=32, hlora_alpha=64, hlora_nums=4, vq_idx_nums=8192,
                 instruct_template='phi4_instruct', **kwargs):
        root = os.environ.get('SONOREASON_HEALTHGPT_ROOT', DEFAULT_HEALTHGPT_ROOT)
        if not os.path.isdir(os.path.join(root, 'llava')):
            raise FileNotFoundError(
                f'HealthGPT code not found at {root}. Clone '
                'https://github.com/DCDmllm/HealthGPT into SonoReason/third_party/ '
                'or set SONOREASON_HEALTHGPT_ROOT.')
        for path in (root, os.path.join(root, 'llava', 'demo')):
            if path not in sys.path:
                sys.path.insert(0, path)

        import transformers
        from huggingface_hub import hf_hub_download, snapshot_download
        from llava import conversation as conversation_lib
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import tokenizer_image_token
        from llava.model.language_model.llava_phi3 import LlavaPhiForCausalLM
        from llava.peft import LoraConfig, get_peft_model
        from utils import com_vision_args, expand2square, find_all_linear_names

        class _LlavaPhiForCausalLM(LlavaPhiForCausalLM):
            # Upstream targets transformers 4.41. Newer generate() also passes
            # cache_position / logits_to_keep / ..., which upstream's explicit
            # forward() signature rejects with a TypeError on the first step.
            def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                        past_key_values=None, inputs_embeds=None, labels=None,
                        images=None, image_sizes=None, **kwargs):
                if inputs_embeds is None:
                    (input_ids, position_ids, attention_mask, past_key_values,
                     inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(
                        input_ids, position_ids, attention_mask, past_key_values,
                        labels, images, image_sizes)
                return super(LlavaPhiForCausalLM, self).forward(
                    input_ids=input_ids, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds, labels=labels, **kwargs)

        # Resolve every component from the local HF cache: compute nodes are offline.
        # Local directories / files are used as-is.
        base_path = model_path if os.path.isdir(model_path) else snapshot_download(model_path)
        vit_local = vit_path if os.path.isdir(vit_path) else snapshot_download(vit_path)
        hlora_path = (hlora_file if os.path.isfile(hlora_file)
                      else hf_hub_download(hlora_repo, hlora_file))

        # Upstream com_infer_phi4.sh runs FP16.
        self.dtype = torch.float16
        model = _LlavaPhiForCausalLM.from_pretrained(base_path, torch_dtype=self.dtype)
        lora_config = LoraConfig(
            r=hlora_r, lora_alpha=hlora_alpha,
            target_modules=find_all_linear_names(model), lora_dropout=0.0,
            bias='none', task_type='CAUSAL_LM', lora_nums=hlora_nums)
        model = get_peft_model(model, lora_config)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_path, padding_side='right', use_fast=False)
        _add_special_tokens_and_resize_model(self.tokenizer, model, vq_idx_nums)

        com_vision_args.model_name_or_path = base_path
        com_vision_args.vision_tower = vit_local
        com_vision_args.version = instruct_template
        model.get_model().initialize_vision_modules(model_args=com_vision_args)
        model.get_vision_tower().to(dtype=self.dtype)

        # Replaces upstream load_weights(): same strict=False load, but mapped to
        # CPU and weights_only, which torch >= 2.6 defaults to anyway.
        state = torch.load(hlora_path, map_location='cpu', weights_only=True)
        unexpected = model.load_state_dict(state, strict=False).unexpected_keys
        if unexpected:
            raise RuntimeError(
                f'{len(unexpected)} H-LoRA keys did not match the model, '
                f'e.g. {unexpected[:3]}')
        self.model = model.to(self.dtype).cuda().eval()

        self.image_processor = self.model.get_vision_tower().image_processor
        self.pad_color = tuple(int(x * 255) for x in self.image_processor.image_mean)
        self.expand2square = expand2square
        self.conv_template = conversation_lib.conv_templates[instruct_template]
        system = os.environ.get('SONOREASON_HEALTHGPT_SYSTEM', DEFAULT_SYSTEM)
        if system:
            self.conv_template = self.conv_template.copy()
            self.conv_template.system = f'<|im_start|>system<|im_sep|>\n{system}'
        self.tokenizer_image_token = tokenizer_image_token
        self.image_token_index = IMAGE_TOKEN_INDEX

        self.gen_kwargs = default_gen_kwargs(tokenizer=self.tokenizer, **kwargs)
        self.prefill_ids, self.prefill_text = resolve_prefill(self.tokenizer)

    def generate_inner(self, message, dataset=None):
        qs, images = '', []
        for msg in message:
            if msg['type'] == 'image':
                image = Image.open(msg['value']).convert('RGB')
                # LLaVA-1.5 recipe (as upstream): pad to square, then the CLIP
                # processor's centre crop is a no-op and nothing is cut away.
                images.append(self.expand2square(image, self.pad_color))
                qs += '<image>\n'
            elif msg['type'] == 'text':
                qs += msg['value']

        conv = self.conv_template.copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        input_ids = self.tokenizer_image_token(
            conv.get_prompt(), self.tokenizer, self.image_token_index,
            return_tensors='pt').unsqueeze(0)
        if self.prefill_ids:
            input_ids = torch.cat(
                [input_ids, torch.tensor([self.prefill_ids], dtype=input_ids.dtype)], dim=1)
        input_ids = input_ids.cuda()

        image_tensor = None
        if images:
            image_tensor = self.image_processor.preprocess(
                images, return_tensors='pt')['pixel_values'].to(
                dtype=self.dtype, device='cuda')

        with torch.inference_mode():
            # Generating from inputs_embeds, so the output holds only new tokens.
            out = self.model.base_model.model.generate(
                input_ids, images=image_tensor,
                image_sizes=[im.size for im in images] if images else None,
                use_cache=True, **self.gen_kwargs)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return (self.prefill_text + text).strip()
