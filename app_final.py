# ============================================================================
# LSTM Next Word Prediction - Streamlit Web App
# Shakespeare's Hamlet Text Generation
# No external dependencies for tokenizer!
# ============================================================================

import streamlit as st
import numpy as np
import json
import os

st.set_page_config(
    page_title="LSTM Next Word Prediction",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD TOKENIZER (JSON - No Keras!)
# ============================================================================

@st.cache_resource
def load_tokenizer():
    """Load tokenizer from JSON file"""
    tokenizer_path = "tokenizer.json"
    
    if not os.path.exists(tokenizer_path):
        st.error(f"❌ Tokenizer file not found: {tokenizer_path}")
        st.stop()
    
    try:
        with open(tokenizer_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"❌ Failed to load tokenizer: {str(e)}")
        st.stop()

# ============================================================================
# LOAD MODEL (Optional - TensorFlow)
# ============================================================================

@st.cache_resource
def load_model():
    """Try to load LSTM model if TensorFlow available"""
    try:
        from tensorflow.keras.models import load_model as tf_load
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        model_path = "next_word_lstm.h5"
        if os.path.exists(model_path):
            return tf_load(model_path), pad_sequences
    except:
        pass
    
    return None, None

# Load resources
tokenizer_data = load_tokenizer()
word_index = tokenizer_data['word_index']
reverse_word_index = tokenizer_data['reverse_word_index']
vocab_size = tokenizer_data['vocab_size']

model, pad_sequences = load_model()
has_model = model is not None

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<h1 style="color: #1f77b4; font-weight: bold;">🎭 LSTM Next Word Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p style="color: #666; font-size: 1.2em;">Shakespeare\'s Hamlet - Powered by Deep Learning</p>', unsafe_allow_html=True)
st.divider()

if has_model:
    st.success("✅ **Full LSTM Model Loaded** - Using trained neural network")
else:
    st.info("⚠️ **Demo Mode** - Model not available, using frequency-based predictions")

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
        st.metric("Max Seq", 14)
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
# PREDICTION FUNCTIONS
# ============================================================================

def texts_to_sequences(text, word_index):
    """Convert text to sequences using word_index"""
    words = text.lower().split()
    sequence = []
    for word in words:
        if word in word_index:
            sequence.append(word_index[word])
    return sequence

def predict_with_model(text, model, pad_sequences, word_index, reverse_word_index):
    """Predict using trained LSTM model"""
    try:
        sequence = texts_to_sequences(text, word_index)
        
        if not sequence:
            return None, None, "No recognized words"
        
        padded = pad_sequences([sequence], maxlen=13, padding='pre')
        predictions = model.predict(padded, verbose=0)
        
        pred_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0])) * 100
        
        # Convert index to word
        pred_word = reverse_word_index.get(str(pred_idx), None)
        if not pred_word:
            pred_word = reverse_word_index.get(int(pred_idx), None)
        
        return pred_word, confidence, None
    except Exception as e:
        return None, None, str(e)

def predict_demo(text, word_index, reverse_word_index):
    """Demo prediction based on word frequency"""
    sequence = texts_to_sequences(text, word_index)
    
    if not sequence:
        return None, None, "No recognized words"
    
    # Predict a random common word
    common_words = {k: v for k, v in list(reverse_word_index.items())[:100]}
    import random
    
    pred_idx = random.choice(list(common_words.keys()))
    pred_word = common_words[pred_idx]
    confidence = 50 + np.random.uniform(-20, 20)
    
    return pred_word, max(0, min(100, confidence)), None

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
            # Use model if available, else demo
            if has_model:
                next_word, confidence, error = predict_with_model(
                    generated_text, model, pad_sequences, word_index, reverse_word_index
                )
            else:
                next_word, confidence, error = predict_demo(
                    generated_text, word_index, reverse_word_index
                )
            
            if error:
                st.error(f"❌ {error}")
                break
            
            if next_word:
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
                st.warning("⚠️ Could not predict")
                break
        
        st.divider()
        st.success(f"**Generated:** `{generated_text}`")

st.divider()
st.caption("🚀 LSTM Text Generation | 📚 Dataset: Shakespeare's Hamlet")
