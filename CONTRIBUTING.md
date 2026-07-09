# Contributing to VetGraph

Thanks for your interest in contributing to VetGraph. Contributions of all sizes are welcome.

Please consider starring and forking the repository to help increase project visibility and make it easier for others to discover the project.

## Ways to contribute

- Report bugs
- Suggest features or improvements
- Improve documentation
- Submit code fixes and new features
- Review open pull requests

## Development setup

1. Fork the repository on GitHub.
2. Clone your fork locally:

```bash
git clone https://github.com/<your-username>/vetgraph.git
cd vetgraph
```

3. Add the upstream remote:

```bash
git remote add upstream https://github.com/Vishesh0-7/vetgraph.git
```

4. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

5. Install the project in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

6. Run checks before pushing:

```bash
pytest tests/ -v
ruff check .
black --check .
```

## Coding standards

- Follow PEP 8 and keep code readable and well-structured.
- Use type hints where practical.
- Keep functions focused and small.
- Add or update tests for any behavior change.
- Format code with Black and lint with Ruff.
- Do not commit secrets, API keys, or local environment files.

## Branch naming conventions

Create branches from latest `main` and use descriptive names:

- `feat/<short-description>` for new features
- `fix/<short-description>` for bug fixes
- `docs/<short-description>` for documentation updates
- `test/<short-description>` for test-only changes
- `chore/<short-description>` for maintenance tasks

Examples:

- `feat/provider-retry-support`
- `fix/schema-validation-edge-case`

## Conventional Commits (required)

VetGraph uses automated Semantic Versioning and publishing. Commit messages in merged PRs drive version bumps and release notes.

Use Conventional Commits format:

- `fix:` -> PATCH bump (e.g., 0.1.2 -> 0.1.3)
- `feat:` -> MINOR bump (e.g., 0.1.2 -> 0.2.0)
- `BREAKING CHANGE:` footer or `feat!:`/`fix!:` -> MAJOR bump (e.g., 0.1.2 -> 1.0.0)
- `docs:`, `chore:`, `refactor:`, `test:`, `ci:` are allowed and may not trigger a release unless configured as release types

Examples:

- `fix: handle empty extraction response`
- `feat: add schema validation for predicates`
- `feat!: redesign graph export API`

If your PR contains multiple commits, keep each commit message conventional, or squash-merge with a conventional commit title.

## How to pick up an open issue

1. Check open issues in the repository.
2. Start with beginner-friendly issues labeled `good first issue`.
3. Use this filter to find them quickly:
   - `is:open is:issue label:"good first issue"`
4. Comment on the issue that you want to work on it (for example, "assign me" or "I'd like to work on this").
5. Wait for assignment confirmation before starting if the issue is already active.

## Pull request process

1. Sync your fork with upstream:

```bash
git fetch upstream
git checkout main
git merge upstream/main
```

2. Create your feature branch:

```bash
git checkout -b feat/my-change
```

3. Make your changes and add tests.
4. Run local checks (`pytest`, `ruff`, `black --check`).
5. Commit with a clear message:

```bash
git commit -m "feat: add support for ..."
```

6. Push to your fork:

```bash
git push -u origin feat/my-change
```

7. Open a pull request against `main` and complete the PR template.

Note: Please make sure your PR title and/or squash commit message follows Conventional Commits. This is required for automated versioning, changelog generation, and PyPI publishing.

## Submitting a strong pull request

- Explain what changed and why.
- Reference the related issue number (for example, `Closes #123`).
- Include tests for new logic.
- Keep changes scoped and focused.

Thanks again for contributing, and please remember to star and fork the repository to help VetGraph grow.
