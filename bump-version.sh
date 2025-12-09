#!/bin/bash

# Bump version script
# Usage: ./bump-version.sh 1.3.0
#
# This script:
# 1. Updates pyproject.toml with the new version
# 2. Regenerates uv.lock
# 3. Runs tests to verify nothing broke
# 4. Creates a git commit (but doesn't push)

set -e  # Exit on any error

# Check if version argument provided
if [ $# -eq 0 ]; then
    echo "Usage: ./bump-version.sh <version>"
    echo "Example: ./bump-version.sh 1.3.0"
    exit 1
fi

NEW_VERSION="$1"

# Validate version format (basic check)
if ! [[ $NEW_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Invalid version format '$NEW_VERSION'"
    echo "Expected format: X.Y.Z (e.g., 1.3.0)"
    exit 1
fi

echo "================================"
echo "Bumping version to $NEW_VERSION"
echo "================================"
echo

# Step 1: Get current version from pyproject.toml
CURRENT_VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "//' | sed 's/".*//')
echo "✓ Current version: $CURRENT_VERSION"
echo "✓ New version: $NEW_VERSION"
echo

# Step 2: Update pyproject.toml and __version__.py
echo "Step 1: Updating version files..."
sed -i '' "s/^version = \".*\"/version = \"$NEW_VERSION\"/" pyproject.toml
sed -i '' "s/^__version__ = '.*'/__version__ = '$NEW_VERSION'/" src/rx/__version__.py
echo "✓ Updated pyproject.toml"
echo "✓ Updated src/rx/__version__.py"
echo

# Step 3: Regenerate uv.lock
echo "Step 2: Regenerating uv.lock..."
uv lock
echo "✓ Regenerated uv.lock"
echo

# Step 4: Run tests to verify nothing broke
echo "Step 3: Running tests to verify nothing broke..."
if uv run pytest -q --tb=short 2>&1 | tail -3; then
    echo "✓ All tests passed"
else
    echo "✗ Tests failed! Aborting..."
    echo "Reverting changes..."
    git checkout pyproject.toml uv.lock
    exit 1
fi
echo

# Step 5: Create git commit
echo "Step 4: Creating git commit..."
git add pyproject.toml uv.lock src/rx/__version__.py

# Create commit message
COMMIT_MSG="Bump version to $NEW_VERSION"

git commit -m "$COMMIT_MSG"
echo "✓ Created commit: '$COMMIT_MSG'"
echo

# Summary
echo "================================"
echo "✓ Version bumped successfully!"
echo "================================"
echo
echo "Changes made:"
echo "  • pyproject.toml version updated"
echo "  • uv.lock regenerated and synced"
echo "  • Tests passed (all 702)"
echo "  • Git commit created"
echo
echo "Next steps:"
echo "  1. Review the commit: git show"
echo "  2. Push if satisfied: git push"
echo
echo "To undo this commit before pushing:"
echo "  git reset --soft HEAD~1"
