#!/usr/bin/env python
"""
Regenerate tokenizer from hamlet.txt as JSON (no Keras dependency)
"""

import json
import os

# Read hamlet.txt
if not os.path.exists('hamlet.txt'):
    print("❌ hamlet.txt not found!")
    exit(1)

with open('hamlet.txt', 'r') as f:
    text = f.read().lower()

# Create word index
words = set()
for line in text.split('\n'):
    words.update(line.split())

# Create word_index dictionary
word_index = {word: idx + 1 for idx, word in enumerate(sorted(words))}
word_index[''] = 0  # Add padding

# Create reverse index
reverse_word_index = {v: k for k, v in word_index.items()}

# Save as JSON
tokenizer_data = {
    'word_index': word_index,
    'reverse_word_index': reverse_word_index,
    'vocab_size': len(word_index)
}

with open('tokenizer.json', 'w') as f:
    json.dump(tokenizer_data, f)

print(f"✅ Tokenizer created: {len(word_index)} words")
print(f"📁 Saved as: tokenizer.json")
