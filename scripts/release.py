#!/usr/bin/env python3
"""
Release management script for pplx-kernels.

This script helps automate version bumping and release preparation.
"""

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_current_version():
    """Get current version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found")
    
    content = pyproject_path.read_text()
    match = re.search(r'version = "([^"]+)"', content)
    if not match:
        raise ValueError("Version not found in pyproject.toml")
    
    return match.group(1)


def update_version(new_version):
    """Update version in pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()
    
    updated_content = re.sub(
        r'version = "[^"]+"',
        f'version = "{new_version}"',
        content
    )
    
    pyproject_path.write_text(updated_content)
    print(f"Updated pyproject.toml version to {new_version}")


def update_changelog(version, changes=None):
    """Update CHANGELOG.md with new version."""
    changelog_path = Path("CHANGELOG.md")
    if not changelog_path.exists():
        print("CHANGELOG.md not found, skipping changelog update")
        return
    
    content = changelog_path.read_text()
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Find the [Unreleased] section and add new version
    unreleased_pattern = r'## \[Unreleased\]'
    if re.search(unreleased_pattern, content):
        new_section = f"## [Unreleased]\n\n## [{version}] - {today}"
        content = re.sub(unreleased_pattern, new_section, content)
    else:
        # Add new section after the header
        header_end = content.find('\n\n')
        if header_end != -1:
            insertion_point = header_end + 2
            new_section = f"## [{version}] - {today}\n\n"
            content = content[:insertion_point] + new_section + content[insertion_point:]
    
    changelog_path.write_text(content)
    print(f"Updated CHANGELOG.md with version {version}")


def bump_version(current_version, bump_type):
    """Bump version based on type (major, minor, patch)."""
    parts = current_version.split('.')
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {current_version}")
    
    major, minor, patch = map(int, parts)
    
    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")


def create_git_tag(version, push=False):
    """Create and optionally push git tag."""
    tag = f"v{version}"
    
    # Check if tag already exists
    result = subprocess.run(["git", "tag", "-l", tag], capture_output=True, text=True)
    if result.stdout.strip():
        print(f"Tag {tag} already exists")
        return False
    
    # Create tag
    subprocess.run(["git", "tag", "-a", tag, "-m", f"Release {tag}"], check=True)
    print(f"Created tag {tag}")
    
    if push:
        subprocess.run(["git", "push", "origin", tag], check=True)
        print(f"Pushed tag {tag} to origin")
    
    return True


def commit_changes(version):
    """Commit version changes."""
    files = ["pyproject.toml", "CHANGELOG.md"]
    existing_files = [f for f in files if Path(f).exists()]
    
    if not existing_files:
        print("No files to commit")
        return
    
    subprocess.run(["git", "add"] + existing_files, check=True)
    subprocess.run(["git", "commit", "-m", f"Prepare release v{version}"], check=True)
    print(f"Committed changes for version {version}")


def main():
    parser = argparse.ArgumentParser(description="Release management for pplx-kernels")
    parser.add_argument("--version", help="Specific version to set (e.g., 1.2.3)")
    parser.add_argument("--bump", choices=["major", "minor", "patch"], help="Bump version")
    parser.add_argument("--tag", action="store_true", help="Create git tag")
    parser.add_argument("--push", action="store_true", help="Push tag to origin")
    parser.add_argument("--commit", action="store_true", help="Commit changes")
    parser.add_argument("--current", action="store_true", help="Show current version")
    
    args = parser.parse_args()
    
    if args.current:
        print(f"Current version: {get_current_version()}")
        return
    
    if not any([args.version, args.bump]):
        parser.error("Must specify either --version or --bump")
    
    current_version = get_current_version()
    print(f"Current version: {current_version}")
    
    if args.version:
        new_version = args.version
    else:
        new_version = bump_version(current_version, args.bump)
    
    print(f"New version: {new_version}")
    
    # Update files
    update_version(new_version)
    update_changelog(new_version)
    
    # Git operations
    if args.commit:
        commit_changes(new_version)
    
    if args.tag:
        create_git_tag(new_version, args.push)
    
    print("\nRelease preparation complete!")
    print(f"Next steps:")
    if not args.commit:
        print(f"  1. Review changes and commit: git add . && git commit -m 'Prepare release v{new_version}'")
    if not args.tag:
        print(f"  2. Create tag: git tag -a v{new_version} -m 'Release v{new_version}'")
    if not args.push:
        print(f"  3. Push tag: git push origin v{new_version}")
    print(f"  4. The GitHub Actions workflow will create the release automatically")


if __name__ == "__main__":
    main()
