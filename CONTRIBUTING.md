# 🤝 Contributing to Blood Cancer Detection (ALL)

Thank you for your interest in contributing to this project! We welcome contributions from researchers, developers, and medical professionals to improve leukemia detection technology.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Workflow](#development-workflow)
- [Reporting Issues](#reporting-issues)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)

## 📜 Code of Conduct

This project adheres to a code of conduct to ensure a welcoming environment for all contributors. By participating, you agree to:

- Be respectful and inclusive
- Focus on constructive feedback
- Accept responsibility for mistakes
- Show empathy towards other contributors
- Help create a positive community

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Development Environment

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/blood-cancer-detection.git
   cd blood-cancer-detection
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate     # Windows
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up pre-commit hooks** (optional):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## 💡 How to Contribute

### Types of Contributions

- 🐛 **Bug fixes** - Report and fix issues
- ✨ **New features** - Add new functionality
- 📚 **Documentation** - Improve docs and tutorials
- 🧪 **Testing** - Add or improve tests
- 🎨 **UI/UX** - Improve user interfaces
- 🔬 **Research** - New models, techniques, or analysis

### Areas for Contribution

#### 🔬 Research Contributions
- Model architecture improvements
- New evaluation metrics
- Transfer learning experiments
- Data augmentation techniques

#### 💻 Technical Contributions
- Code optimization
- Performance improvements
- New preprocessing methods
- API development

#### 📊 Data Science Contributions
- Statistical analysis
- Visualization improvements
- Feature engineering
- Model interpretation

#### 🌐 Web Development
- Streamlit app enhancements
- New UI components
- Accessibility improvements
- Mobile responsiveness

## 🔄 Development Workflow

### 1. Choose an Issue
- Check [open issues](../../issues) for tasks
- Comment on the issue to indicate you're working on it
- Wait for maintainer approval if needed

### 2. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number-description
```

### 3. Make Changes
- Write clear, concise commit messages
- Test your changes thoroughly
- Update documentation if needed
- Add tests for new functionality

### 4. Test Your Changes
```bash
# Run the complete pipeline
python cancer.py
python limpieza.py
python diccionario_datos.py
python entrenamiento_y_metricas.py

# Test the web app
streamlit run app.py
```

### 5. Submit a Pull Request
- Push your branch to your fork
- Create a Pull Request with a clear description
- Reference any related issues
- Request review from maintainers

## 🐛 Reporting Issues

### Bug Reports
When reporting bugs, please include:

- **Clear title** describing the issue
- **Steps to reproduce** the problem
- **Expected behavior** vs actual behavior
- **Environment details** (OS, Python version, etc.)
- **Error messages** and stack traces
- **Screenshots** if applicable

### Feature Requests
For new features, please provide:

- **Clear description** of the proposed feature
- **Use case** and benefits
- **Implementation ideas** if you have them
- **Mockups** or examples if applicable

## 🔍 Pull Request Process

### Before Submitting
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

### PR Template
Please use this template for pull requests:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Performance improvement

## Testing
Describe how you tested your changes

## Screenshots (if applicable)
Add screenshots of UI changes

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
```

## 🎨 Style Guidelines

### Python Code Style

We follow PEP 8 with some modifications:

```python
# Good: Clear variable names, consistent spacing
def preprocess_image(image_path: str) -> np.ndarray:
    """Preprocess a single image for model input."""
    image = Image.open(image_path).convert('RGB')
    # ... processing logic ...
    return processed_image

# Avoid: Unclear names, inconsistent style
def proc(img):
    i = Image.open(img).convert('RGB')
    # ... processing logic ...
    return i
```

### Key Guidelines

- **Use type hints** for function parameters and return values
- **Write docstrings** for all public functions
- **Use meaningful variable names** (avoid single letters except in loops)
- **Keep functions small** and focused on single responsibility
- **Add comments** for complex logic
- **Use relative imports** within the project

### Documentation Style

```python
def complex_function(param1: str, param2: int) -> dict:
    """
    Perform complex data processing operation.

    Args:
        param1 (str): Description of first parameter
        param2 (int): Description of second parameter

    Returns:
        dict: Processed data with keys 'result' and 'metadata'

    Raises:
        ValueError: If param1 is empty
        TypeError: If param2 is not an integer

    Example:
        >>> result = complex_function("data", 42)
        >>> print(result['result'])
        processed_data
    """
```

## 🧪 Testing Guidelines

### Unit Tests
```python
import pytest
from your_module import your_function

def test_your_function():
    # Arrange
    input_data = "test_input"
    expected_output = "expected_result"

    # Act
    result = your_function(input_data)

    # Assert
    assert result == expected_output

def test_your_function_edge_cases():
    # Test edge cases
    with pytest.raises(ValueError):
        your_function("")
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest test_your_module.py

# Run with coverage
pytest --cov=your_module --cov-report=html
```

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For general questions and ideas
- **Email**: For private matters or sensitive issues

## 🎯 Recognition

Contributors will be:
- Listed in the README acknowledgments
- Mentioned in release notes
- Credited in academic publications
- Invited to join as maintainers for significant contributions

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the same MIT License that covers the project.

---

Thank you for contributing to Blood Cancer Detection (ALL)! 🩸🤖

Your work helps bring accurate leukemia diagnosis to communities worldwide.