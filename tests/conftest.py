import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="session")
def test_model_and_tokenizer(device):
    # Use a small model for testing
    model_name = "distilbert/distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

@pytest.fixture(scope="session")
def test_string():
    return "This is a test string for STGSDiffString"

@pytest.fixture(scope="session")
def target_test_string():
    return "This string is a target test string for STGSDiffString"

@pytest.fixture(scope="session")
def vocab_size():
    return 50257
    