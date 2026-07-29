"""
Global test configuration with consistent mocks for all ML dependencies.

All test files share these mocks to avoid conflicts from
sys.modules state being shared across test files.
"""

import sys
import types
from unittest.mock import MagicMock


class _MockModule(types.ModuleType):
    pass


def _install_mocks():
    # Only install if not already done
    if 'torch' in sys.modules and hasattr(sys.modules['torch'], '_mocked'):
        return

    _torch = _MockModule("torch")
    _torch._mocked = True
    _torch.Tensor = type("Tensor", (), {"__init__": lambda s: None, "to": lambda s, d: s})
    _torch.device = lambda x: x
    _torch.float16 = "float16"
    _torch.bfloat16 = "bfloat16"
    _torch.cuda = type("cuda", (), {"is_available": staticmethod(lambda: False), "empty_cache": staticmethod(lambda: None)})()
    _torch.no_grad = lambda: type('no_grad', (), {'__enter__': lambda s: None, '__exit__': lambda s, *a: None})()
    _torch.zeros = MagicMock(return_value=MagicMock())
    _torch.stack = MagicMock()

    # nn
    _torch.nn = _MockModule("torch.nn")
    _torch.nn.Module = type("Module", (), {"__init__": lambda s: None, "parameters": lambda s: []})
    _torch.nn.Linear = MagicMock(return_value=MagicMock())
    _torch.nn.Dropout = MagicMock(return_value=MagicMock())
    _mock_reward_tensor = MagicMock()
    _mock_reward_tensor.item = lambda: 0.5
    _mock_reward_head = MagicMock()
    _mock_reward_head.side_effect = lambda x: _mock_reward_tensor
    _torch.nn.Sequential = MagicMock(return_value=_mock_reward_head)
    _torch.nn.functional = _MockModule("torch.nn.functional")
    _torch.nn.functional.logsigmoid = lambda x: x

    # optim
    _torch.optim = _MockModule("torch.optim")

    # utils.data
    _torch.utils = _MockModule("torch.utils")
    _torch.utils.data = _MockModule("torch.utils.data")
    _torch.utils.data.Dataset = type("Dataset", (), {"__len__": lambda s: 0})

    # transformers
    _transformers = _MockModule("transformers")
    _mock_causal_lm = MagicMock()
    _mock_causal_lm.print_trainable_parameters = lambda: None
    _mock_causal_lm.gradient_checkpointing_enable = lambda: None
    _transformers.AutoModelForCausalLM = type("AutoModelForCausalLM", (), {
        "from_pretrained": staticmethod(lambda *a, **kw: _mock_causal_lm)
    })()
    _mock_tokenizer = MagicMock()
    _mock_tokenizer.pad_token = None
    _mock_tokenizer.eos_token = "<|endoftext|>"
    _mock_tokenizer.encode = MagicMock(return_value=[101, 102, 103])
    _mock_tokenizer.__call__ = MagicMock(return_value=MagicMock())
    _transformers.AutoTokenizer = type("AutoTokenizer", (), {
        "from_pretrained": staticmethod(lambda *a, **kw: _mock_tokenizer)
    })()
    _transformers.TrainingArguments = MagicMock

    # AutoModel for reward model
    _mock_output = MagicMock()
    _mock_output.last_hidden_state = MagicMock()
    
    class _FakeTensor:
        """Tensor-like with real .item() returning float."""
        def item(self):
            return 0.5
        def __getitem__(self, key):
            return _FakeTensor()
    
    _mock_output.last_hidden_state.__getitem__ = lambda s, k: _FakeTensor()
    _mock_output.last_hidden_state.__class__ = type('MockTensor', (), {'__getitem__': lambda s, k: _FakeTensor()})
    _mock_model_instance = MagicMock()
    _mock_model_instance.__call__ = lambda **kw: _mock_output
    _transformers.AutoModel = type("Auto", (), {
        "from_pretrained": staticmethod(lambda *a, **kw: _mock_model_instance)
    })()

    # peft
    _peft = _MockModule("peft")
    _peft.LoraConfig = MagicMock
    _peft.get_peft_model = MagicMock(return_value=MagicMock())

    # trl
    _trl = _MockModule("trl")
    _trl.DPOTrainer = MagicMock

    # Install
    for name, mod in [
        ("torch", _torch), ("torch.nn", _torch.nn), ("torch.nn.functional", _torch.nn.functional),
        ("torch.optim", _torch.optim), ("torch.utils", _torch.utils), ("torch.utils.data", _torch.utils.data),
        ("transformers", _transformers), ("peft", _peft), ("trl", _trl),
    ]:
        if name not in sys.modules:
            sys.modules[name] = mod


_install_mocks()
