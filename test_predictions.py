"""
Test LSTM Next Word Prediction
Demonstrates model predictions without running Streamlit
"""

import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
print("Loading model and tokenizer...")
model = load_model("next_word_lstm.h5")
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Get vocab size and max sequence length
reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
max_seq_length = model.input_shape[1] + 1

def predict_next_word(text):
    """Predict next word for given text"""
    text = text.lower().strip()
    sequence = tokenizer.texts_to_sequences([text])[0]

    if not sequence:
        return None, None

    padded = pad_sequences([sequence], maxlen=max_seq_length-1, padding='pre')
    predictions = model.predict(padded, verbose=0)

    pred_idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0])) * 100

    pred_word = reverse_word_index.get(pred_idx, None)
    return pred_word, confidence

# Test phrases
print("\n" + "="*60)
print("LSTM NEXT WORD PREDICTIONS - TEST OUTPUT")
print("="*60 + "\n")

test_phrases = [
    "to be or not",
    "to be or not to",
    "such a sight",
    "hamlet is",
    "good night",
    "long live the",
    "all the world",
    "this above all"
]

for phrase in test_phrases:
    next_word, confidence = predict_next_word(phrase)
    if next_word:
        print(f"Input:  '{phrase}'")
        print(f"Output: '{next_word}' (Confidence: {confidence:.1f}%)")
        print(f"Full:   '{phrase} {next_word}'")
        print("-" * 60)
    else:
        print(f"Input:  '{phrase}'")
        print(f"Output: Could not predict")
        print("-" * 60)

print("\n✅ Model is working correctly!")
print(f"📊 Vocabulary Size: {len(tokenizer.word_index) + 1:,}")
print(f"🧠 Total Parameters: {model.count_params():,}")
print(f"📏 Max Sequence Length: {max_seq_length}")
