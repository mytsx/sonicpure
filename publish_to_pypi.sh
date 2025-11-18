#!/bin/bash
# PyPI'ye publish script

set -e

echo "🚀 PyPI Publish Script"
echo "====================="

# Temizlik
echo "📦 Cleaning old builds..."
rm -rf build/ dist/ *.egg-info/

# Build
echo "🔨 Building package..."
python3 -m pip install --upgrade build twine
python3 -m build

# Test PyPI'ye upload (önce test et)
echo ""
echo "📤 Uploading to TEST PyPI..."
echo "TestPyPI URL: https://test.pypi.org/project/gurultu/"
python3 -m twine upload --repository testpypi dist/*

echo ""
echo "✅ Test PyPI'ye yüklendi!"
echo "Test etmek için:"
echo "  pip install --index-url https://test.pypi.org/simple/ gurultu"
echo ""
echo "Gerçek PyPI'ye yüklemek için:"
echo "  python3 -m twine upload dist/*"
