# Development Setup

Welcome to the development guide for nickyspatial! This guide will help you get up and running quickly, while also explaining the tools and processes we use.

## 1. Install UV

We use [UV](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) because it is **freaking fast** and simplifies dependency management. UV streamlines the installation and synchronization of dependencies, making development smoother and more efficient.

Install UV by running:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```


## 2. Sync Project Dependencies

Once UV is installed, install the project dependencies directly into your virtual environment (`.venv`) with:

```bash
uv sync
```

This command reads project's configuration (i.e. `pyproject.toml`) and ensures that all required libraries are installed with the correct versions.

## 3. Install Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) hooks to ensure code quality and consistency across the project. Our pre-commit configuration includes:

- **UV Lock**: Ensures locking of dependency versions.
- **Ruff Hooks (linter and formatter)**: Ruff is used for linting and formatting. It helps catch issues and enforces a consistent code style.
- **Commitizen**: Helps enforce conventional commit messages for better project history.

To set up these hooks, run:

```bash
pre-commit install
```

This will automatically run the following on every commit:

- uv-lock: Validates your UV lock file.
- ruff: Checks code style and formatting.
- commitizen: Validates commit messages against the conventional commits specification.


When you make changes you can run
```bash
uv run pre-commit run --all-files
```
To check you pass all the precommit checks.

## 4. Getting Started

Once you have UV installed, dependencies synced, and pre-commit hooks set, you’re ready for development. A typical workflow might look like:

- Work on a **feature** or **bug fix**. Just tell other people what you will be working on in issues
- **Run your tests** – our project uses Pytest for testing.
- **Commit your changes** – pre-commit hooks ensure that your code meets our quality standards and that your commit messages follow the Conventional Commits guidelines.
- **Submit your PR** - Create a branch with suitable name as per as your changes and raise PR

#### Bring up frontend

```bash
uv run streamlit run streamlit.py
```
