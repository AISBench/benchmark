import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = os.environ.get(
    'CORESET_MODEL_PATH',
    os.environ.get('MODEL_PATH', '/workspace/raw_models/Qwen3-4B-Instruct-2507'),
)


def load_model(model_path=None):
    if model_path is None:
        model_path = os.environ.get('MODEL_PATH', DEFAULT_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
    model.eval()
    return model, tokenizer


def generate_logits(model, tokenizer, prompts_generator):
    for prompt in prompts_generator:
        model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                model_inputs.input_ids,
                max_new_tokens=2,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        # first generated step, last layer, last position
        last_layer_idx = model.config.num_hidden_layers
        first_token_hidden = outputs.hidden_states[1][last_layer_idx][:, -1, :]
        yield first_token_hidden.squeeze(0)
