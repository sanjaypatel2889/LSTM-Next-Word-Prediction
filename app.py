# ============================================================================
# LSTM Next Word Prediction - Streamlit Web App
# Shakespeare's Hamlet Text Generation
# ============================================================================

import streamlit as st
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD MODEL AND TOKENIZER
# ============================================================================

@st.cache_resource
def load_resources():
    """Load model and tokenizer with proper error handling"""
    try:
        # Try TensorFlow first
        from tensorflow.keras.models import load_model
    except:
        try:
            # Fallback to Keras
            from keras.models import load_model
        except:
            st.error("❌ TensorFlow/Keras not installed! Run: `pip install tensorflow`")
            st.stop()
    
    model_path = "next_word_lstm.h5"
    tokenizer_path = "tokenizer.pickle"
    
    # Load model
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()
    
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.stop()
    
    # Load tokenizer
    if not os.path.exists(tokenizer_path):
        st.error(f"❌ Tokenizer file not found: {tokenizer_path}")
        st.stop()
    
    try:
        with open(tokenizer_path, "rb") as handle:
            tokenizer = pickle.load(handle)
    except Exception as e:
        st.error(f"❌ Failed to load tokenizer: {str(e)}")
        st.stop()
    
    return model, tokenizer

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="LSTM Next Word Prediction",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main-title { font-size: 2.5em; color: #1f77b4; font-weight: bold; }
        .subtitle { font-size: 1.2em; color: #666; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD RESOURCES
# ============================================================================

try:
    model, tokenizer = load_resources()
    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
    vocab_size = len(tokenizer.word_index) + 1
    max_seq_length = model.input_shape[1] + 1
except Exception as e:
    st.error(f"Failed to initialize: {str(e)}")
    st.stop()

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_next_word(model, tokenizer, text, max_sequence_len):
    """Predict next word with confidence score"""
    try:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
    except:
        from keras.preprocessing.sequence import pad_sequences
    
    try:
        text = text.lower().strip()
        
        if not text:
            return None, None, "Input text cannot be empty"
        
        sequence = tokenizer.texts_to_sequences([text])[0]
        
        if not sequence:
            return None, None, "No recognized words in input"
        
        padded_sequence = pad_sequences(
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
# MAIN CONTENT
# ============================================================================

st.markdown('<h1 class="main-title">🎭 LSTM Next Word Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Shakespeare\'s Hamlet - Powered by Deep Learning</p>', unsafe_allow_html=True)
st.divider()

st.info("""
**🤖 About:** This LSTM neural network was trained on Shakespeare's *Hamlet* to predict 
the next word in a sequence. Enter a phrase and the model will predict what comes next!
""")

# Input section
col1, col2 = st.columns([3, 1])

with col1:
    input_text = st.text_input(
        "Enter words from Hamlet:",
        value="to be or not",
        placeholder="e.g., 'to be or not to'"
    )

with col2:
    num_words = st.slider("Generate", 1, 5, 1)

# Predict button
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
            next_word, confidence, error = predict_next_word(
                model,
                tokenizer,
                generated_text,
                max_seq_length
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

