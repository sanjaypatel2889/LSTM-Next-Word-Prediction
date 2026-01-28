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

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="LSTM Next Word Prediction",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .main-title { font-size: 2.5em; color: #1f77b4; }
        .subtitle { font-size: 1.2em; color: #666; margin-bottom: 20px; }
        .prediction-box { 
            background-color: #f0f2f6; 
            padding: 15px; 
            border-radius: 10px; 
            margin: 10px 0;
        }
        .confidence-high { color: #28a745; font-weight: bold; }
        .confidence-medium { color: #ffc107; font-weight: bold; }
        .confidence-low { color: #dc3545; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# RESOURCE LOADING
# ============================================================================
@st.cache_resource
def load_resources():
    """Load model and tokenizer with comprehensive error handling"""
    model_path = "next_word_lstm.h5"
    tokenizer_path = "tokenizer.pickle"
    
    errors = []
    
    # Check if files exist
    if not os.path.exists(model_path):
        errors.append(f"❌ Model file not found: {model_path}")
    
    if not os.path.exists(tokenizer_path):
        errors.append(f"❌ Tokenizer file not found: {tokenizer_path}")
    
    if errors:
        st.error("\n".join(errors))
        st.stop()
    
    # Load model
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.stop()
    
    # Load tokenizer
    try:
        with open(tokenizer_path, "rb") as handle:
            tokenizer = pickle.load(handle)
    except Exception as e:
        st.error(f"❌ Failed to load tokenizer: {str(e)}")
        st.stop()
    
    return model, tokenizer

# Load resources
try:
    model, tokenizer = load_resources()
    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
    vocab_size = len(tokenizer.word_index) + 1
    max_seq_length = model.input_shape[1] + 1
except Exception as e:
    st.error(f"Failed to initialize app: {str(e)}")
    st.stop()

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def predict_next_word(model, tokenizer, text, max_sequence_len):
    """
    Predict the next word with error handling and confidence score.
    
    Args:
        model: Trained LSTM model
        tokenizer: Keras tokenizer
        text: Input text sequence
        max_sequence_len: Maximum sequence length
    
    Returns:
        tuple: (predicted_word, confidence_score, error_message)
    """
    try:
        # Convert text to lowercase (must match training data)
        text = text.lower().strip()
        
        if not text:
            return None, None, "Input text cannot be empty"
        
        # Convert text to sequence of integers
        sequence = tokenizer.texts_to_sequences([text])[0]
        
        # Check if any words were recognized
        if not sequence:
            return None, None, "⚠️ No recognized words in input. Please use words from Hamlet."
        
        # Pad sequence to required length
        padded_sequence = pad_sequences(
            [sequence],
            maxlen=max_sequence_len - 1,
            padding="pre"
        )

        # Predict probabilities for next word
        predictions = model.predict(padded_sequence, verbose=0)

        # Get index and confidence of highest probability
        predicted_index = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions[0])) * 100

        # Map index back to word
        if predicted_index in reverse_word_index:
            predicted_word = reverse_word_index[predicted_index]
            return predicted_word, confidence, None
        else:
            return None, None, "Could not map prediction to word"
    
    except Exception as e:
        return None, None, f"Prediction error: {str(e)}"

# ============================================================================
# SIDEBAR - PROJECT INFORMATION
# ============================================================================
with st.sidebar:
    st.header("📊 Model Information")
    
    # Model statistics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Vocabulary", f"{vocab_size:,}")
        st.metric("Model Type", "LSTM")
    with col2:
        st.metric("Max Seq Length", max_seq_length)
        st.metric("Training Data", "Hamlet")
    
    st.divider()
    
    # How it works
    st.subheader("🎓 How It Works")
    st.markdown("""
    1. **Input**: Enter a phrase from Hamlet
    2. **Processing**: Model analyzes word patterns
    3. **Prediction**: Generates most likely next word
    4. **Confidence**: Shows prediction certainty (0-100%)
    5. **Generation**: Can generate multiple words in sequence
    """)
    
    st.divider()
    
    # Example phrases
    st.subheader("📚 Example Phrases")
    example_phrases = [
        "to be or not",
        "to be or not to",
        "such a sight",
        "hamlet is",
        "good night",
        "this above all",
        "all the world's"
    ]
    for phrase in example_phrases:
        st.caption(f"✓ \"{phrase}\"")
    
    st.divider()
    
    # Model details
    st.subheader("⚙️ Model Architecture")
    st.markdown("""
    - **Embedding Layer**: 100 dimensions
    - **LSTM Layer 1**: 150 units, return_sequences=True
    - **Dropout**: 0.2 regularization
    - **LSTM Layer 2**: 100 units
    - **Output Layer**: Softmax (vocab size)
    - **Optimizer**: Adam
    - **Loss**: Categorical Crossentropy
    """)

# ============================================================================
# MAIN CONTENT
# ============================================================================
# Header
st.markdown('<h1 class="main-title">🎭 Next Word Prediction using LSTM</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Shakespeare\'s Hamlet Text Generation with Deep Learning</p>', unsafe_allow_html=True)
st.divider()

# Information box
st.info("""
**🤖 About This Model:**

This LSTM (Long Short-Term Memory) neural network was trained on the complete text of Shakespeare's *Hamlet*. 
It learns the statistical patterns and linguistic structures of the play, allowing it to predict contextually 
appropriate next words given a sequence of input words.

**💡 Try it out:** Enter any phrase from Hamlet and let the model predict what comes next!
""")

st.divider()

# ============================================================================
# USER INPUT SECTION
# ============================================================================
st.subheader("🔤 Text Input")

# Text input
input_text = st.text_input(
    "Enter a sequence of words from Hamlet:",
    value="To be or not",
    placeholder="e.g., 'to be or not to'",
    help="Type 2+ words for best results"
)

# Control settings
col1, col2, col3 = st.columns(3)

with col1:
    num_predictions = st.slider(
        "Words to generate:",
        min_value=1,
        max_value=5,
        value=1,
        help="How many words to predict in sequence"
    )

with col2:
    model_choice = st.selectbox(
        "Model variant:",
        ["LSTM v1", "LSTM v2"],
        help="LSTM v2 has better architecture"
    )

with col3:
    show_top_k = st.checkbox("Show Top 3 Predictions", value=False)

st.divider()

# ============================================================================
# PREDICTION BUTTON & LOGIC
# ============================================================================
predict_button = st.button("🔮 Generate Predictions", use_container_width=True, type="primary")

if predict_button:
    # Input validation
    if not input_text.strip():
        st.error("❌ Please enter some text!")
    elif len(input_text.split()) < 2:
        st.error("❌ Please enter at least 2 words for better context!")
    else:
        st.subheader("📝 Results")
        
        # Create a placeholder for results
        results_container = st.container()
        progress_bar = st.progress(0)
        
        generated_text = input_text
        all_predictions = []
        
        # Generate predictions
        with results_container:
            for step in range(num_predictions):
                # Get prediction
                next_word, confidence, error = predict_next_word(
                    model,
                    tokenizer,
                    generated_text,
                    max_seq_length
                )
                
                # Handle results
                if error:
                    st.error(f"❌ {error}")
                    break
                
                elif next_word:
                    # Store prediction
                    all_predictions.append({
                        'word': next_word,
                        'confidence': confidence,
                        'step': step + 1
                    })
                    
                    # Display prediction card
                    with st.container():
                        col_word, col_conf, col_pct = st.columns([2, 1, 1])
                        
                        with col_word:
                            st.write(f"**Step {step + 1}:** `{next_word}`")
                        
                        with col_conf:
                            # Color code confidence
                            if confidence > 60:
                                st.markdown(f'<span class="confidence-high">High</span>', unsafe_allow_html=True)
                            elif confidence > 30:
                                st.markdown(f'<span class="confidence-medium">Medium</span>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<span class="confidence-low">Low</span>', unsafe_allow_html=True)
                        
                        with col_pct:
                            st.metric("Confidence", f"{confidence:.1f}%")
                    
                    # Add to generated text
                    generated_text += f" {next_word}"
                    
                    # Update progress
                    progress_bar.progress((step + 1) / num_predictions)
                
                else:
                    st.warning("⚠️ Could not generate prediction for this sequence.")
                    break
        
        st.divider()
        
        # Final result
        if all_predictions:
            st.success(f"✅ **Generated Text:** `{generated_text}`")
            
            # Summary statistics
            avg_confidence = np.mean([p['confidence'] for p in all_predictions])
            st.metric("Average Confidence", f"{avg_confidence:.1f}%")

st.divider()

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
---
**📌 Notes:**
- Model predicts based on patterns from *Hamlet* training text
- Predictions improve with more context words
- Confidence score indicates prediction certainty
- Use Hamlet-specific vocabulary for best results

**🔗 Project:** LSTM Text Generation | **📚 Dataset:** Shakespeare's Hamlet
""")
