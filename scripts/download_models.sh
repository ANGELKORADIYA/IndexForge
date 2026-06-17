#!/bin/bash

# Exit on error
set -e

# Create directories
mkdir -p models data/symspell-dict

echo "--- MemorySearch Model Downloader ---"

# Note: In a real scenario, we would use curl/wget to download specific files.
# For now, this script documents what needs to be downloaded as per PROJECT.md.

echo "Step 1: Downloading all-MiniLM-L6-v2 (embedding model)..."
# In a real setup, you might use:
# git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 models/all-MiniLM-L6-v2
echo "[SKIP] Embedding model download (logic to be implemented or handled by fastembed-rs)"

echo "Step 2: Downloading ms-marco cross-encoder (re-ranker)..."
echo "[SKIP] Re-ranker model download (logic to be implemented)"

echo "Step 3: Downloading SymSpell frequency dictionary..."
if [ ! -f data/symspell-dict/frequency.txt ]; then
    curl -L https://raw.githubusercontent.com/wolfgarbe/SymSpell/master/SymSpell/frequency_dictionary_en_82_765.txt \
        -o data/symspell-dict/frequency.txt
    echo "Done."
else
    echo "SymSpell dictionary already exists."
fi

echo "--- All models handled ---"
echo "Note: Some models are downloaded automatically by the Rust crates on first run."
