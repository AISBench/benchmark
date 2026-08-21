import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path):
    """Load the Hugging Face model used for hidden-state feature extraction."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
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
