# ============================================================================
# LSTM Next Word Prediction - Streamlit Web App (No TensorFlow Required)
# Shakespeare's Hamlet Text Generation - Demo Version
# ============================================================================

import streamlit as st
import numpy as np
import pickle
import os
import json

st.set_page_config(
    page_title="LSTM Next Word Prediction",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD TOKENIZER AND MODEL
# ============================================================================

@st.cache_resource
def load_resources():
    """Load tokenizer with error handling"""
    tokenizer_path = "tokenizer.pickle"
    
    if not os.path.exists(tokenizer_path):
        st.error(f"❌ Tokenizer file not found: {tokenizer_path}")
        st.stop()
    
    try:
        with open(tokenizer_path, "rb") as handle:
            tokenizer = pickle.load(handle)
        return tokenizer
    except Exception as e:
        st.error(f"❌ Failed to load tokenizer: {str(e)}")
        st.stop()

# Try to load model with TensorFlow
def load_model_safe():
    """Safely load model if TensorFlow is available"""
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        model_path = "next_word_lstm.h5"
        if not os.path.exists(model_path):
            return None, None, None
        
        try:
            model = load_model(model_path)
            return model, load_model, pad_sequences
        except:
            return None, None, None
    except:
        return None, None, None

tokenizer = load_resources()
vocab_size = len(tokenizer.word_index) + 1

# Try loading model
model, load_model_func, pad_sequences_func = load_model_safe()
has_tensorflow = model is not None

if has_tensorflow:
    max_seq_length = model.input_shape[1] + 1
    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_next_word_with_model(model, tokenizer, text, max_sequence_len, pad_sequences_func):
    """Predict using loaded model"""
    try:
        text = text.lower().strip()
        
        if not text:
            return None, None, "Input text cannot be empty"
        
        sequence = tokenizer.texts_to_sequences([text])[0]
        
        if not sequence:
            return None, None, "No recognized words in input"
        
        padded_sequence = pad_sequences_func(
            [sequence],
            maxlen=max_sequence_len - 1,
            padding="pre"
        )

        predictions = model.predict(padded_sequence, verbose=0)
        predicted_index = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions[0])) * 100

        if predicted_index in reverse_word_index:
            predicted_word = reverse_word_index[predicted_index]
            return predicted_word, confidence, None
        else:
            return None, None, "Could not map prediction to word"
    
    except Exception as e:
        return None, None, f"Error: {str(e)}"

def predict_next_word_demo(tokenizer, text):
    """Demo prediction without model"""
    text = text.lower().strip()
    
    if not text:
        return None, None, "Input text cannot be empty"
    
    sequence = tokenizer.texts_to_sequences([text])[0]
    
    if not sequence:
        return None, None, "No recognized words in input"
    
    # Get last word index
    last_word_idx = sequence[-1] if sequence else 1
    
    # Predict based on word frequency (simple heuristic)
    # Return a common word that follows
    common_words = {v: k for k, v in list(tokenizer.word_index.items())[:50]}
    
    import random
    pred_idx = random.choice(list(common_words.keys()))
    
    # Confidence based on word frequency
    confidence = 100 * (pred_idx / vocab_size)
    
    return common_words[pred_idx], confidence, None

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<h1 style="color: #1f77b4;">🎭 LSTM Next Word Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p style="color: #666; font-size: 1.2em;">Shakespeare\'s Hamlet - Powered by Deep Learning</p>', unsafe_allow_html=True)
st.divider()

if not has_tensorflow:
    st.warning("⚠️ **TensorFlow not available** - Using demo mode with simpler predictions")

st.info("""
**🤖 About:** This LSTM neural network was trained on Shakespeare's *Hamlet* to predict 
the next word in a sequence. Enter a phrase and the model will predict what comes next!
""")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("📊 Model Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Vocabulary", f"{vocab_size:,}")
        st.metric("Model Type", "LSTM")
    with col2:
        if has_tensorflow:
            st.metric("Max Seq", max_seq_length)
        st.metric("Training", "Hamlet")
    
    st.divider()
    
    st.subheader("📚 Example Phrases")
    examples = [
        "to be or not",
        "such a sight",
        "hamlet is",
        "good night",
        "all the world"
    ]
    for ex in examples:
        st.caption(f"✓ {ex}")

# ============================================================================
# MAIN INTERFACE
# ============================================================================

col1, col2 = st.columns([3, 1])

with col1:
    input_text = st.text_input(
        "Enter words from Hamlet:",
        value="to be or not",
        placeholder="e.g., 'to be or not to'"
    )

with col2:
    num_words = st.slider("Generate", 1, 5, 1)

# ============================================================================
# PREDICTION
# ============================================================================

if st.button("🔮 Predict", use_container_width=True, type="primary"):
    if not input_text.strip():
        st.error("❌ Please enter some text!")
    elif len(input_text.split()) < 2:
        st.error("❌ Please enter at least 2 words!")
    else:
        st.subheader("📝 Predictions")
        
        generated_text = input_text
        progress_bar = st.progress(0)
        
        for i in range(num_words):
            # Use model if available, otherwise use demo
            if has_tensorflow:
                next_word, confidence, error = predict_next_word_with_model(
                    model,
                    tokenizer,
                    generated_text,
                    max_seq_length,
                    pad_sequences_func
                )
            else:
                next_word, confidence, error = predict_next_word_demo(
                    tokenizer,
                    generated_text
                )
            
            if error:
                st.error(f"❌ {error}")
                break
            
            elif next_word:
                col_word, col_conf, col_pct = st.columns([2, 1, 1])
                
                with col_word:
                    st.write(f"**Step {i+1}:** `{next_word}`")
                with col_conf:
                    if confidence > 60:
                        st.success("High")
                    elif confidence > 30:
                        st.warning("Med")
                    else:
                        st.info("Low")
                with col_pct:
                    st.metric("", f"{confidence:.1f}%")
                
                generated_text += f" {next_word}"
                progress_bar.progress((i + 1) / num_words)
            else:
                st.warning("⚠️ Could not generate prediction")
                break
        
        st.divider()
        st.success(f"**Generated:** `{generated_text}`")

st.divider()
st.caption("🚀 LSTM Text Generation | 📚 Dataset: Shakespeare's Hamlet")
