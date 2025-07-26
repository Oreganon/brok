## Features

- **Code Quality**: Configured with `ruff` for linting and formatting
- **Type Safety**: Full type hints with `mypy` type checking
- **Testing**: Comprehensive test suite with `pytest` and coverage reporting
- **CI/CD**: GitHub Actions workflow for automated testing and quality checks
- **Pre-commit Hooks**: Automated code quality checks before commits

## Setup

### Requirements

- Python 3.13+
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd brok
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development

### Code Quality

This project uses several tools to maintain code quality:

- **ruff**: Fast Python linter and formatter
- **mypy**: Static type checker
- **bandit**: Security linter
- **pytest**: Testing framework with coverage

### Running Quality Checks

```bash
# Run linting
ruff check .

# Run formatting
ruff format .

# Run type checking
mypy .

# Run tests with coverage
pytest

# Run security checks
bandit -r .
```

### Pre-commit Hooks

Pre-commit hooks automatically run quality checks before each commit:

```bash
# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### Testing

Write tests for all new functionality following these guidelines:

- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test names that explain the scenario
- Follow the Arrange-Act-Assert pattern
- Aim for >90% code coverage

Run tests:
```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_main.py
```

### Project Structure

```
brok/
├── .cursor/rules/          # Cursor IDE rules
├── .github/workflows/      # GitHub Actions CI/CD
├── tests/                  # Test files
├── main.py                 # Main application code
├── pyproject.toml         # Project configuration
├── .pre-commit-config.yaml # Pre-commit hooks
├── .gitignore             # Git ignore patterns
└── README.md              # This file
```

## Configuration

### pyproject.toml

The `pyproject.toml` file contains configuration for:

- Project metadata and dependencies
- ruff linting and formatting rules
- mypy type checking settings
- pytest and coverage configuration
