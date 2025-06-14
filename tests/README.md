# SDLM Test Suite

This directory contains the test suite for the Straight-Through Gumbel-Softmax Differentiable Language Modeling (SDLM) package. The tests are designed to ensure the correctness and robustness of the core components.

## Test Files

### 1. `test_stgs.py`
Tests for the `STGS` (Straight-Through Gumbel-Softmax) module.

**Test Cases:**
- `test_forward_shape`: Verifies output shapes for different input dimensions
- `test_hard_sampling`: Tests hard sampling mode (one-hot outputs)
- `test_soft_sampling`: Tests soft sampling mode (probability distributions)
- `test_temperature_effect`: Verifies temperature parameter's effect on sampling
- `test_gradient_flow`: Ensures proper gradient flow through the module

### 2. `test_stgs_diff_string.py`
Tests for the `STGSDiffString` class which handles differentiable string operations.

**Test Cases:**
- `test_initialization`: Verifies proper initialization and properties
- `test_forward_pass`: Tests the forward pass and output formats
- `test_hard_vs_soft`: Compares hard and soft sampling modes
- `test_gradient_flow`: Ensures gradients flow correctly
- `test_to_device`: Tests device management (CPU/GPU)

### 3. `test_stgs_diff_model.py`
Tests for the `STGSDiffModel` class which integrates with language models.

**Test Cases:**
- `test_initialization`: Verifies model initialization
- `test_forward_pass`: Tests the forward pass and output structure
- `test_generate_standard`: Tests standard text generation
- `test_generate_with_bpttoken`: Tests generation with Backpropagation Through Tokens
- `test_gradient_flow`: Verifies gradient propagation
- `test_save_load_state_dict`: Tests model serialization/deserialization

## Running the Tests

### Prerequisites
- Python 3.7+
- PyTorch
- Transformers
- Pytest
- Pytest-cov (for coverage reports)

### Installation

```bash
# Install the package in development mode
pip install -e .

# Install test dependencies
pip install pytest pytest-cov
```

### Running All Tests

```bash
# From the project root
pytest sdlm/tests/ -v

# With coverage report
pytest sdlm/tests/ --cov=sdlm --cov-report=term-missing
```

### Running Specific Tests

```bash
# Run a specific test file
pytest sdlm/tests/test_stgs.py -v

# Run a specific test function
pytest sdlm/tests/test_stgs.py::TestSTGS::test_forward_shape -v

# Run tests with a marker
pytest sdlm/tests/ -m "not slow"  # Skip slow tests
```

### Test Coverage

To generate a coverage report:

```bash
pytest sdlm/tests/ --cov=sdlm --cov-report=html
```

This will generate an HTML report in the `htmlcov` directory.

## Fixtures

The `conftest.py` file provides the following fixtures:

- `device`: The computation device (CPU/CUDA)
- `test_model_and_tokenizer`: A pre-trained GPT-2 model and tokenizer
- `test_string`: A sample test string

## Writing New Tests

When adding new tests:

1. Follow the existing naming convention (`test_*` for test functions)
2. Use the provided fixtures when possible
3. Add appropriate assertions to verify behavior
4. Consider edge cases and error conditions
5. Add appropriate docstrings and comments

## Continuous Integration

Consider setting up a CI/CD pipeline to run these tests automatically on pushes and pull requests. A basic GitHub Actions configuration might include:

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[test]
    - name: Run tests
      run: |
        pytest sdlm/tests/ --cov=sdlm --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```
