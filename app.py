import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="LSTM Next Word Prediction",
    page_icon="🎭",
    layout="wide"
)

st.markdown("""
<style>
.main-title {font-size: 2.5em; color: #1f77b4; font-weight: bold; text-align: center;}
.subtitle {font-size: 1.2em; color: #666; text-align: center; margin-bottom: 30px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        model = load_model("next_word_lstm.h5", compile=False)
        with open("tokenizer.pickle", "rb") as f:
            tokenizer = pickle.load(f)

        return model, tokenizer, pad_sequences
    except Exception as e:
        st.error(f"Error loading: {e}")
        st.stop()

model, tokenizer, pad_sequences = load_resources()
reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
vocab_size = len(tokenizer.word_index) + 1
max_seq_length = model.input_shape[1] + 1

def predict_next_word(text):
    text = text.lower().strip()
    if not text:
        return None, 0, "Empty input"

    sequence = tokenizer.texts_to_sequences([text])[0]
    if not sequence:
        return None, 0, "No recognized words"

    padded = pad_sequences([sequence], maxlen=max_seq_length-1, padding='pre')
    predictions = model.predict(padded, verbose=0)

    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0])) * 100
    predicted_word = reverse_word_index.get(predicted_index)

    return predicted_word, confidence, None if predicted_word else "No prediction"

st.markdown('<h1 class="main-title">🎭 LSTM Next Word Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Shakespeare\'s Hamlet - Deep Learning</p>', unsafe_allow_html=True)
st.divider()

st.info("**🤖 About:** LSTM neural network predicts the next word in Shakespeare's Hamlet. Enter a phrase!")

with st.sidebar:
    st.header("📊 Model Info")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Vocabulary", f"{vocab_size:,}")
        st.metric("Type", "LSTM")
    with col2:
        st.metric("Max Seq", max_seq_length)
        st.metric("Data", "Hamlet")

    st.divider()
    st.subheader("📚 Examples")
    for ex in ["to be or not", "to be or not to", "such a sight", "hamlet is", "good night", "long live the"]:
        st.caption(f"✓ {ex}")

col1, col2 = st.columns([3, 1])
with col1:
    input_text = st.text_input("Enter words from Hamlet:", value="to be or not", placeholder="e.g., 'to be or not to'")
with col2:
    num_words = st.slider("Generate", 1, 5, 1)

if st.button("🔮 Predict", use_container_width=True, type="primary"):
    if not input_text.strip():
        st.error("❌ Please enter text!")
    elif len(input_text.split()) < 2:
        st.warning("⚠️ Enter at least 2 words")
    else:
        st.subheader("📝 Predictions")
        generated_text = input_text
        progress_bar = st.progress(0)

        for i in range(num_words):
            next_word, confidence, error = predict_next_word(generated_text)

            if error:
                st.error(f"❌ {error}")
                break

            if next_word:
                col_word, col_conf, col_pct = st.columns([2, 1, 1])
                with col_word:
                    st.write(f"**Step {i+1}:** `{next_word}`")
                with col_conf:
                    st.success("High") if confidence > 60 else st.warning("Med") if confidence > 30 else st.info("Low")
                with col_pct:
                    st.metric("", f"{confidence:.1f}%")

                generated_text += f" {next_word}"
                progress_bar.progress((i + 1) / num_words)
            else:
                st.warning("⚠️ No prediction")
                break

        st.divider()
        st.success(f"✅ **Generated:** `{generated_text}`")

st.divider()
st.caption("🚀 Built with Streamlit | 📚 Shakespeare's Hamlet")
