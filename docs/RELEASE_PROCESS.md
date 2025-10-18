# Release Process

This document describes the release process for pplx-kernels, including how to create stable tags and releases.

## Overview

The pplx-kernels project follows [Semantic Versioning](https://semver.org/) and provides stable releases through Git tags and GitHub releases. This allows consumers to pin to specific versions instead of using commit SHAs.

## Version Format

Versions follow the format `vX.Y.Z` where:
- `X` = Major version (breaking changes)
- `Y` = Minor version (new features, backwards compatible)
- `Z` = Patch version (bug fixes, backwards compatible)

Examples: `v0.1.0`, `v1.0.0`, `v1.2.3`

## Release Types

### 1. Automated Release (Recommended)

Use GitHub Actions workflow for consistent releases:

1. Go to the [Actions tab](https://github.com/perplexityai/pplx-kernels/actions)
2. Select "Release" workflow
3. Click "Run workflow"
4. Specify:
   - Tag (e.g., `v0.2.0`)
   - Whether it's a pre-release
5. The workflow will:
   - Update `pyproject.toml` version
   - Create a Git tag
   - Generate release notes
   - Create GitHub release

### 2. Manual Release

For manual releases:

```bash
# 1. Update version in pyproject.toml
sed -i 's/version = "[^"]*"/version = "0.2.0"/' pyproject.toml

# 2. Update CHANGELOG.md
# Add new version section with changes

# 3. Commit changes
git add pyproject.toml CHANGELOG.md
git commit -m "Prepare release v0.2.0"

# 4. Create and push tag
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0

# 5. Create GitHub release manually or let the workflow handle it
```

## Pre-release Process

For testing releases before stable versions:

1. Use pre-release tags: `v0.2.0-alpha.1`, `v0.2.0-beta.1`, `v0.2.0-rc.1`
2. Mark as pre-release in GitHub
3. Test thoroughly before creating stable release

## Updating CHANGELOG.md

Before each release, update `CHANGELOG.md`:

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New features

### Changed
- Modifications to existing features

### Fixed
- Bug fixes

### Removed
- Removed features

### Security
- Security improvements
```

## Installation Instructions for Users

With stable releases, users can now install specific versions:

### Using pip
```bash
# Latest stable release
pip install git+https://github.com/perplexityai/pplx-kernels.git

# Specific version
pip install git+https://github.com/perplexityai/pplx-kernels.git@v0.1.0
```

### Using requirements.txt
```
# Pin to specific version
git+https://github.com/perplexityai/pplx-kernels.git@v0.1.0

# Use version range (when published to PyPI)
pplx-kernels>=0.1.0,<0.2.0
```

### Using Poetry
```toml
[tool.poetry.dependencies]
pplx-kernels = {git = "https://github.com/perplexityai/pplx-kernels.git", tag = "v0.1.0"}
```

## Migration Guide

### For Existing Users

If you're currently pinning to commit SHAs, migrate to version tags:

**Before:**
```bash
pip install git+https://github.com/perplexityai/pplx-kernels.git@2bd6e30fab358829bd01d1c0907be8088ff9e5e1
```

**After:**
```bash
pip install git+https://github.com/perplexityai/pplx-kernels.git@v0.1.0
```

## Hotfix Process

For critical fixes that need immediate release:

1. Create hotfix branch from the release tag
2. Apply minimal fix
3. Increment patch version (e.g., v0.1.0 → v0.1.1)
4. Follow normal release process
5. Merge back to main branch

## Release Checklist

- [ ] Update version in `pyproject.toml`
- [ ] Update `CHANGELOG.md` with new version
- [ ] Test build and installation locally
- [ ] Run full test suite
- [ ] Create and push Git tag
- [ ] Create GitHub release with release notes
- [ ] Verify release artifacts
- [ ] Update documentation if needed
- [ ] Announce release (if applicable)

## Troubleshooting

### Version Conflicts
If you encounter version conflicts, ensure `pyproject.toml` version matches the Git tag.

### Failed Builds
Check that all dependencies are properly specified and the build environment is correct.

### Missing Tags
Ensure tags are pushed to the remote repository: `git push origin --tags`

## Future Improvements

- PyPI publishing for easier installation
- Automated changelog generation from commit messages
- Integration with package managers
- Automated testing for each release
