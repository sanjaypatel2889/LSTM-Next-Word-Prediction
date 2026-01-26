# app.py
# -----------------------------
# Next Word Prediction using LSTM + Streamlit
# Shakespeare's Hamlet Text Generation
# -----------------------------

import streamlit as st
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set page config
st.set_page_config(page_title="LSTM Next Word Prediction", layout="centered")

# Check if files exist
@st.cache_resource
def load_resources():
    """Load model and tokenizer with error handling"""
    model_path = "next_word_lstm.h5"
    tokenizer_path = "tokenizer.pickle"
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()
    
    # Check if tokenizer exists
    if not os.path.exists(tokenizer_path):
        st.error(f"❌ Tokenizer file not found: {tokenizer_path}")
        st.stop()
    
    # Load model
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()
    
    # Load tokenizer
    try:
        with open(tokenizer_path, "rb") as handle:
            tokenizer = pickle.load(handle)
    except Exception as e:
        st.error(f"❌ Failed to load tokenizer: {e}")
        st.stop()
    
    return model, tokenizer

# Load resources
model, tokenizer = load_resources()

# -----------------------------
# Function: predict next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    """Predict next word with error handling"""
    try:
        # Convert text to sequence of integers
        sequence = tokenizer.texts_to_sequences([text])[0]
        
        # Validate sequence
        if not sequence:
            return None, "⚠️ No recognized words in input. Please use Hamlet-related text."
        
        # Pad sequence to required length
        sequence = pad_sequences(
            [sequence],
            maxlen=max_sequence_len - 1,
            padding="pre"
        )

        # Predict probabilities for next word
        predictions = model.predict(sequence, verbose=0)

        # Get index of highest probability
        predicted_index = np.argmax(predictions, axis=1)[0]

        # Map index back to word
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                return word, None

        return None, "Could not map prediction to word"
    
    except Exception as e:
        return None, f"Error during prediction: {str(e)}"

# -----------------------------
# Streamlit UI
st.title("🎭 Next Word Prediction using LSTM")
st.markdown("**Shakespeare's Hamlet Text Generation**")
st.markdown("---")

# Display info
st.info("""
This LSTM model was trained on Shakespeare's Hamlet to predict the next word in a sequence.
Try entering phrases from Hamlet or related text!

**Examples:** "to be or not", "Such a sight", "Hamlet is", "good night"
""")

# Text input
input_text = st.text_input("Enter a sequence of words", "To be or not", key="input")

# Button
if st.button("🔮 Predict Next Word"):
    if not input_text.strip():
        st.warning("⚠️ Please enter some text!")
    else:
        # Get max sequence length from model
        max_sequence_len = model.input_shape[1] + 1

        # Predict next word
        next_word, error = predict_next_word(
            model,
            tokenizer,
            input_text,
            max_sequence_len
        )

        # Display result
        if error:
            st.error(f"❌ {error}"
        elif next_word:
            st.success(f"**Next word:** `{next_word}`")
            st.markdown(f"**Full phrase:** {input_text} **{next_word}**")
        else:
            st.warning("⚠️ Could not predict next word")
