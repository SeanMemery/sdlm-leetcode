# SDLM: Straight-Through Gumbel-Softmax Differentiable Language Modelling

SDLM is a Python library that provides differentiable text generation capabilities using the Straight-Through Gumbel-Softmax trick. It allows for gradient-based optimization through discrete token generation, making it useful for tasks that require end-to-end training with discrete text outputs.

## Features

- `STGS`: Core implementation of the Straight-Through Gumbel-Softmax operation
- `STGSDiffModel`: Wrapper for HuggingFace models to enable differentiable text generation
- `STGSDiffString`: A differentiable string representation for gradient-based text manipulation

## Installation

```bash
pip install -e .
```

## Usage

```python
from sdlm import STGSDiffModel, STGSDiffString
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize with a pretrained model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)

# Create differentiable model
diff_model = STGSDiffModel(
    model=base_model,
    tokenizer=tokenizer,
    temperature=0.7,
    hard=True,
    learnable_temperature=True
)

# Generate text with gradient flow
outputs = diff_model.generate(input_ids=input_ids, max_length=50)
```

## Project Structure

```
sdlm/
├── sdlm/                      # Package source
│   ├── __init__.py
│   ├── stgs_diff_model.py     # STGSDiffModel implementation
│   └── stgs_diff_string.py    # STGSDiffString implementation
├── tests/                     # Unit tests
├── examples/                  # Example scripts
└── README.md                 # This file
```

## Development

### Testing

Run the test suite with:

```bash
pytest tests/
```

### TODO List

#### STGS Class Tests
- [ ] Test initialization with different parameters
- [ ] Test forward pass with fixed temperature
- [ ] Test forward pass with learnable temperature
- [ ] Test gradient flow through STGS operation
- [ ] Test hard vs soft sampling modes
- [ ] Test device handling (CPU/GPU)

#### STGSDiffModel Tests
- [ ] Test initialization with different model types
- [ ] Test forward pass with and without labels
- [ ] Test gradient flow through generation
- [ ] Test temperature scheduling
- [ ] Test batch processing
- [ ] Test attention mask handling
- [ ] Test with different model architectures

#### STGSDiffString Tests
- [ ] Test initialization with different strings
- [ ] Test string manipulation operations
- [ ] Test gradient flow through string operations
- [ ] Test device handling
- [ ] Test serialization/deserialization

## License

MIT

## Citation

If you use SDLM in your research, please cite:

```
@misc{sdlm,
  author = {Kevin Denamganaï},
  title = {SDLM: Straight-Through Gumbel-Softmax Differentiable Language Modelling},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Near32/sdlm}}
}
```
