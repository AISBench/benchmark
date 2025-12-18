# Contributing Guide

Thank you for your interest in the AISBench evaluation tool! We welcome all forms of contributions, including but not limited to code, documentation, issue reports, and suggestions.

## Table of Contents

- [Task Board](#task-board)
- [License](#license)
- [Development](#development)
- [Code Checking (Linting)](#code-checking-linting)
- [Documentation](#documentation)
- [Testing](#testing)
- [Issues](#issues)
- [Pull Requests (PRs) and Code Reviews](#pull-requests-prs-and-code-reviews)
- [DCO and Signed-off-by](#dco-and-signed-off-by)
- [PR Title and Classification](#pr-title-and-classification)
- [Code Quality](#code-quality)
- [Adding or Changing Components](#adding-or-changing-components)
- [Breaking Changes](#breaking-changes)
- [Review Expectations](#review-expectations)
- [Acknowledgments](#acknowledgments)

## Task Board

We use GitHub Issues to track tasks and issues. Before starting to contribute, we recommend:

1. Check existing [Issues](https://github.com/AISBench/benchmark/issues) to understand current work priorities
2. For new features or major changes, it is recommended to create an Issue for discussion first
3. For bug fixes, you can directly submit a Pull Request

## License

AISBench uses the [Apache License 2.0](../../../LICENSE) license. By contributing code to this project, you agree that your contributions will be released under the same license.

All contributors must ensure:

- You own or have the right to license the code you submit
- Your contributions do not infringe any third-party intellectual property rights
- You agree to license your contributions to the project maintainers

## Development

### Environment Setup

1. **Python Version Requirements**: Only supports Python 3.10, 3.11, or 3.12

2. **Clone Repository**:

```bash
git clone https://github.com/AISBench/benchmark.git
cd benchmark/
```

3. **Create Development Environment** (recommended to use Conda):

```bash
conda create --name ais_bench_dev python=3.10 -y
conda activate ais_bench_dev
```

4. **Install Development Dependencies**:

```bash
# Install core dependencies
pip3 install -e ./ --use-pep517

# Install development dependencies (if needed)
pip3 install -r requirements/api.txt
pip3 install -r requirements/extra.txt
```

### Code Structure

AISBench adopts a registry-based plugin architecture:

- **Core Plugin Modules**: Located in the `ais_bench/benchmark/` directory
  - `models/`: Model runtime classes
  - `datasets/`: Dataset loaders
  - `openicl/icl_evaluator`: ICL evaluators
  - `openicl/icl_inferencer`: ICL inferencers
  - `calculators/`: Performance metric calculators
  - `registry.py`: Registry definitions

- **Configuration Files**: Located in the `ais_bench/benchmark/configs/` directory
  - `models/`: Model configurations
  - `datasets/`: Dataset configurations
  - `summarizers/`: Result summarizer configurations

- **Plugin System**: Supports extending functionality through entry_point mechanism, see [Plugin Development Guide](../../../plugin_examples/README.md) for details

### Development Workflow

1. **Create Branch**:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

2. **Develop**: Write code, add tests, update documentation

3. **Commit Changes**:

```bash
git add .
git commit -m "feat: Add new feature description"
```

4. **Push and Create Pull Request**

## Code Checking (Linting)

Before submitting code, please ensure the code complies with the project's code standards:

### Python Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) code style guidelines
- Use 4 spaces for indentation
- Line length recommended not to exceed 120 characters
- Use meaningful variable and function names

### Code Formatting Tools

It is recommended to use the following tools for code formatting:

```bash
# Use black for code formatting (if project is configured)
black ais_bench/

# Use isort for import sorting (if project is configured)
isort ais_bench/
```

### Static Checking

Run basic Python syntax checking before submitting:

```bash
python -m py_compile your_file.py
```

## Documentation

### Documentation Structure

Project documentation is located in the `docs/` directory:

- `source_zh_cn/`: Chinese documentation
- `source_en/`: English documentation

### Documentation Writing Standards

1. **Markdown Format**: Use Markdown to write documentation
2. **Chinese Documentation**: Use simplified Chinese, language should be concise and clear
3. **Code Examples**: Provide complete runnable code examples
4. **Link Checking**: Ensure all links are valid

### Updating Documentation

- When adding new features, please synchronously update related documentation
- When modifying APIs, please update corresponding documentation
- When adding dataset or model configurations, please provide README.md instructions

## Testing

### Test Structure

Test files are located in the `tests/` directory:

- `UT/`: Unit tests (integrated into CI)

### Running Tests

Install test-related dependencies

```bash
# Run all tests
python tests/run_tests.py

# Run all tests with multiple processes
python tests/run_tests.py -p 4 # 4 processes

# Run specific test file
pytest tests/UT/test_example.py
```

### Writing Tests

- **Unit Tests**: Test functionality of single functions or classes
- **Integration Tests**: Test interactions between multiple components
- **Test Naming**: Use `test_` prefix to name test functions
- **Test Coverage**: New features should include corresponding test cases, target coverage not less than 80%

## Issues

### Reporting Bugs

When reporting bugs, please provide the following information:

1. **Problem Description**: Clearly describe the problem phenomenon
2. **Reproduction Steps**: Provide reproducible steps
3. **Expected Behavior**: Describe expected behavior
4. **Actual Behavior**: Describe actual behavior
5. **Environment Information**:
   - Python version
   - Operating system
   - AISBench version
   - Related dependency versions
6. **Log Information**: Provide relevant error logs or stack traces

### Feature Requests

When proposing feature requests, please explain:

1. **Feature Description**: Describe in detail the feature you want to add
2. **Use Cases**: Explain the use cases and value of this feature
3. **Possible Implementation**: If you have ideas, you can propose implementation suggestions

## Pull Requests (PRs) and Code Reviews

### Creating Pull Requests

1. **Preparation**:
   - Ensure code has passed all tests
   - Ensure code complies with code standards
   - Update related documentation

2. **PR Description**:
   - Clearly describe the purpose and changes of the PR
   - If it resolves an Issue, please use the `Fixes #123` format to associate
   - Provide test results or screenshots (if applicable)

3. **Code Review**:
   - Keep PR size moderate for easy review
   - Respond to review comments promptly
   - Make modifications based on feedback

### Review Process

1. **Automated Checks**: CI checks will automatically run after PR creation
2. **Code Review**: Maintainers will review the code
3. **Modification Feedback**: Make modifications based on review comments
4. **Merge**: After review passes, maintainers will merge the PR

## DCO and Signed-off-by

AISBench uses the [Developer Certificate of Origin (DCO)](https://developercertificate.org/) to ensure the legality of contributions.

### How to Sign Commits

Add a `Signed-off-by` line to the commit message:

```bash
git commit -s -m "feat: Add new feature"
```

The `-s` parameter will automatically add the `Signed-off-by` line.

### Signed-off-by Format

```
Signed-off-by: Your Name <your.email@example.com>
```

### Meaning

By adding `Signed-off-by`, you confirm:

1. The contribution is your original work, or you have the right to submit it under an open source license
2. You agree to release the contribution under Apache License 2.0

## PR Title and Classification

### Title Format

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <subject>
```

### Types

- `[feat]`: New feature
- `[fix]`: Bug fix
- `[docs]`: Documentation update
- `[style]`: Code format adjustment (does not affect functionality)
- `[refactor]`: Code refactoring
- `[test]`: Test related
- `[chore]`: Build process or auxiliary tool changes
- `[perf]`: Performance optimization

### Examples

```
[feat][models]Add new model backend support
[fix][datasets]Fix dataset loading error
[docs][readme]Update installation instructions
[test][calculators]Add performance calculator tests
```

## Code Quality

### Code Review Standards

Code reviews will focus on the following aspects:

1. **Functional Correctness**: Whether the code implements expected functionality
2. **Code Quality**: Whether the code is clear and maintainable
3. **Test Coverage**: Whether there is sufficient test coverage
4. **Documentation Completeness**: Whether related documentation has been updated
5. **Performance Impact**: Whether there is a negative impact on performance
6. **Backward Compatibility**: Whether existing functionality has been broken

### Best Practices

1. **Single Responsibility**: Each function/class should do only one thing
2. **DRY Principle**: Avoid duplicate code
3. **Clear Comments**: Add comments for complex logic
4. **Error Handling**: Properly handle exception cases
5. **Type Hints**: It is recommended to use type hints to improve code readability

## Adding or Changing Components

### Adding New Model Backends

Refer to [Supporting New Models](./new_model.md)

### Adding New Datasets and Accuracy Evaluators

Refer to [Supporting New Datasets and Accuracy Evaluators](./new_dataset.md)

### Adding New Inferencers

Refer to [Supporting New Inferencers](./new_inferencer.md)

## Breaking Changes

Please try to keep changes concise. For major architectural changes (>500 lines of code, excluding data/config/tests), we expect you to:

1. **Discuss in Advance**: Create a discussion in GitHub Issues, explaining the reason for changes and scope of impact
2. **Backward Compatibility**: Try to maintain backward compatibility, or provide migration guides
3. **Documentation Updates**: Update all related documentation
4. **Test Coverage**: Ensure sufficient test coverage
5. **Version Notes**: Detail changes in CHANGELOG

### Breaking Change Examples

- API interface changes
- Configuration file format changes
- Default behavior changes
- Removal of deprecated features

## Review Expectations

### As a Contributor

- **Be Patient**: Reviews may take some time, please be patient
- **Respond Actively**: Respond to review comments promptly and make necessary modifications
- **Stay Polite**: Maintain a professional and polite attitude in discussions
- **Continuous Improvement**: Treat reviews as learning opportunities and continuously improve code quality

### As a Reviewer

- **Respond Promptly**: Review PRs as soon as possible and provide constructive feedback
- **Friendly Communication**: Provide feedback in a friendly and professional manner
- **Explain Reasons**: Explain why modifications are needed to help contributors understand
- **Recognize Contributions**: Recognize good contributions and encourage continued participation

## Acknowledgments

Thank you to all developers who have contributed to AISBench! Your contributions make this project better.

### Ways to Contribute

You can contribute in the following ways:

- 🐛 Report Bugs
- 💡 Propose Feature Suggestions
- 📝 Improve Documentation
- 💻 Submit Code
- 🧪 Write Tests
- 🔍 Code Review
- 📢 Promote the Project

### Contributor List

All contributors will be recognized in the project. Thank you for every contributor's efforts!

---

If you have any questions, please feel free to raise them in [GitHub Issues](https://github.com/AISBench/benchmark/issues).

