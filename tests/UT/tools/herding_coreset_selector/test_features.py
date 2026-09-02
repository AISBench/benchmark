from types import SimpleNamespace
from unittest.mock import Mock

import torch

from herding import features


def test_load_model(monkeypatch):
    tokenizer = object()
    model = Mock()
    tokenizer_loader = Mock(return_value=tokenizer)
    model_loader = Mock(return_value=model)
    monkeypatch.setattr(features.AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(features.AutoModelForCausalLM, "from_pretrained", model_loader)

    assert features.load_model("/models/example") == (model, tokenizer)
    tokenizer_loader.assert_called_once_with("/models/example")
    model_loader.assert_called_once_with("/models/example", device_map="auto")
    model.eval.assert_called_once_with()


def test_generate_logits_extracts_hidden_state():
    class Inputs:
        input_ids = torch.tensor([[1, 2]])

        def to(self, device):
            self.device = device
            return self

    inputs = Inputs()
    tokenizer = Mock(return_value=inputs)
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    model = Mock(
        device=torch.device("cpu"),
        config=SimpleNamespace(num_hidden_layers=1),
    )
    model.generate.return_value = SimpleNamespace(
        hidden_states=[None, [torch.zeros_like(hidden), hidden]]
    )

    result = list(features.generate_logits(model, tokenizer, ["prompt"]))

    tokenizer.assert_called_once_with("prompt", return_tensors="pt")
    model.generate.assert_called_once_with(
        inputs.input_ids,
        max_new_tokens=2,
        do_sample=False,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    assert inputs.device == model.device
    torch.testing.assert_close(result[0], torch.tensor([3.0, 4.0]))
