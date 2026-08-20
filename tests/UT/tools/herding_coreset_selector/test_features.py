from types import SimpleNamespace
from unittest.mock import Mock

import torch

from herding import features


def test_load_model_uses_auto_device_map_and_eval(monkeypatch):
    tokenizer = object()
    model = Mock()
    tokenizer_loader = Mock(return_value=tokenizer)
    model_loader = Mock(return_value=model)
    monkeypatch.setattr(features.AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(
        features.AutoModelForCausalLM,
        "from_pretrained",
        model_loader,
    )

    loaded_model, loaded_tokenizer = features.load_model("/models/example")

    assert loaded_model is model
    assert loaded_tokenizer is tokenizer
    tokenizer_loader.assert_called_once_with("/models/example")
    model_loader.assert_called_once_with("/models/example", device_map="auto")
    model.eval.assert_called_once_with()


def test_generate_logits_extracts_first_generated_hidden_state():
    class FakeInputs:
        input_ids = torch.tensor([[1, 2]])

        def __init__(self):
            self.target_device = None

        def to(self, device):
            self.target_device = device
            return self

    inputs = FakeInputs()
    tokenizer = Mock(return_value=inputs)
    expected = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    generated_step = [torch.zeros_like(expected), torch.ones_like(expected), expected]
    model = Mock()
    model.device = torch.device("cpu")
    model.config.num_hidden_layers = 2
    model.generate.return_value = SimpleNamespace(
        hidden_states=[None, generated_step]
    )

    results = list(features.generate_logits(model, tokenizer, ["first prompt"]))

    assert inputs.target_device == model.device
    tokenizer.assert_called_once_with("first prompt", return_tensors="pt")
    model.generate.assert_called_once_with(
        inputs.input_ids,
        max_new_tokens=2,
        do_sample=False,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    assert len(results) == 1
    torch.testing.assert_close(results[0], torch.tensor([3.0, 4.0]))


def test_generate_logits_handles_multiple_prompts():
    class FakeInputs:
        input_ids = torch.tensor([[1]])

        def to(self, _device):
            return self

    tokenizer = Mock(side_effect=lambda *_args, **_kwargs: FakeInputs())
    hidden = torch.tensor([[[5.0]]])
    model = Mock()
    model.device = torch.device("cpu")
    model.config.num_hidden_layers = 0
    model.generate.return_value = SimpleNamespace(hidden_states=[None, [hidden]])

    results = list(features.generate_logits(model, tokenizer, ["one", "two"]))

    assert tokenizer.call_count == 2
    assert model.generate.call_count == 2
    assert [result.item() for result in results] == [5.0, 5.0]
