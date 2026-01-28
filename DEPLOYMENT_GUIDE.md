# 🚀 Streamlit Deployment Guide - LSTM Next Word Prediction

## Overview
This is a complete guide to deploy the LSTM Next Word Prediction app on Streamlit Cloud (free hosting) or local deployment.

---

## **Option 1: Run Locally (Development)**

### Prerequisites
- Python 3.8+
- pip package manager

### Steps:

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify Files Exist:**
   - `app.py` ✓
   - `next_word_lstm.h5` ✓
   - `tokenizer.pickle` ✓
   - `hamlet.txt` ✓

3. **Run the App:**
```bash
streamlit run app.py
```

4. **Access the App:**
   - Open your browser to: `http://localhost:8501`
   - The app will automatically reload on code changes

---

## **Option 2: Deploy on Streamlit Cloud (Free)**

### Prerequisites
- GitHub account
- Streamlit Cloud account (free signup at streamlit.io)

### Step-by-Step Deployment:

#### **Step 1: Push to GitHub**

1. Create a new GitHub repository: `lstm-next-word-prediction`

2. Clone the repo locally:
```bash
git clone https://github.com/YOUR_USERNAME/lstm-next-word-prediction.git
cd lstm-next-word-prediction
```

3. Copy all project files to this directory:
   - `app.py`
   - `experiment.ipynb`
   - `next_word_lstm.h5`
   - `tokenizer.pickle`
   - `hamlet.txt`
   - `requirements.txt`
   - `README.md`

4. Commit and push to GitHub:
```bash
git add .
git commit -m "Initial LSTM deployment"
git push origin main
```

#### **Step 2: Create Streamlit Account**

1. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
2. Sign up with your GitHub account
3. Authorize Streamlit to access your repositories

#### **Step 3: Deploy on Streamlit Cloud**

1. Click **"New app"** on your Streamlit Cloud dashboard
2. Select:
   - **Repository:** `YOUR_USERNAME/lstm-next-word-prediction`
   - **Branch:** `main`
   - **Main file path:** `app.py`

3. Click **Deploy!**

4. Streamlit will install dependencies and launch your app
5. Your app URL will be: `https://your-app-name-xxxxx.streamlit.app`

#### **Step 4: Share Your App**

- Your app is now live and shareable!
- Share the URL with anyone

---

## **Project File Structure**

```
lstm-next-word-prediction/
├── app.py                      # Streamlit application
├── experiment.ipynb            # Jupyter notebook with model training
├── next_word_lstm.h5          # Trained LSTM model
├── tokenizer.pickle            # Keras tokenizer (4,818 vocabulary)
├── hamlet.txt                  # Training data
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── DEPLOYMENT_GUIDE.md        # This file
```

---

## **App Features**

### Main Interface
- ✅ Text input for phrases from Hamlet
- ✅ Adjustable prediction count (1-5 words)
- ✅ Model variant selection (LSTM v1/v2)
- ✅ Confidence scores for predictions
- ✅ Real-time progress bar
- ✅ Generated text output

### Sidebar Information
- 📊 Model architecture details
- 📚 Example phrases
- 🎓 How it works guide
- ⚙️ Technical specifications

---

## **Model Information**

**Architecture:**
- Embedding Layer: 100 dimensions
- LSTM Layer 1: 150 units (return_sequences=True)
- Dropout: 0.2 regularization
- LSTM Layer 2: 100 units
- Dense Output: Softmax (4,818 vocabulary size)

**Training:**
- Dataset: Shakespeare's Hamlet (~30K words)
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Epochs: 50-200 (with EarlyStopping)

**Performance:**
- Total Parameters: 1,157,418
- Trainable Parameters: 1,157,418
- Input Shape: (batch_size, 13)

---

## **Troubleshooting**

### Issue: "Model file not found"
**Solution:** Ensure `next_word_lstm.h5` is in the same directory as `app.py`

### Issue: "Tokenizer file not found"
**Solution:** Ensure `tokenizer.pickle` is in the same directory as `app.py`

### Issue: "No module named tensorflow"
**Solution:** Run `pip install -r requirements.txt`

### Issue: App takes long to start
**Solution:** First run caches model and tokenizer. Subsequent runs will be faster.

### Issue: Streamlit Cloud deployment fails
**Solution:**
- Check file sizes (max 100MB per file)
- Ensure all required files are in GitHub repo
- Check `requirements.txt` has correct versions

---

## **Performance Optimization**

The app uses Streamlit's `@st.cache_resource` decorator to cache the model and tokenizer, ensuring:
- ⚡ Fast initial load after first run
- ⚡ No reloading of large files on interaction
- ⚡ Better user experience

---

## **Example Phrases to Try**

```
✓ "to be or not"
✓ "to be or not to"
✓ "such a sight"
✓ "hamlet is"
✓ "good night"
✓ "this above all"
✓ "all the world's"
✓ "long live the"
```

---

## **Customization**

### Change Default Input:
Edit `app.py` line 227:
```python
input_text = st.text_input(
    "Enter a sequence of words from Hamlet:",
    value="To be or not",  # ← Change this
    ...
)
```

### Change Model Colors:
Edit CSS in `app.py` lines 22-32:
```python
st.markdown("""
    <style>
        .main-title { color: #1f77b4; }  # ← Change color
        ...
    </style>
""", unsafe_allow_html=True)
```

---

## **Support**

For issues or questions:
1. Check the troubleshooting section above
2. Review app.py code comments
3. Check Streamlit documentation: https://docs.streamlit.io
4. Check TensorFlow documentation: https://www.tensorflow.org/api_docs

---

## **License**

This project uses Shakespeare's Hamlet from the NLTK Gutenberg corpus (public domain).

---

**Happy Deploying! 🚀**
