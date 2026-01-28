# 🚀 QUICK COMMAND REFERENCE

## Local Deployment (Windows PowerShell)

### 1️⃣ First Time Setup
```powershell
cd "c:\Users\sanjay\Downloads\LSTM PROJECT"
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2️⃣ Run the App
```powershell
streamlit run app.py
```

### 3️⃣ Access in Browser
```
http://localhost:8501
```

---

## Git Commands (GitHub Deployment)

### Initialize Repository
```powershell
cd "c:\Users\sanjay\Downloads\LSTM PROJECT"
git init
git add .
git commit -m "Initial LSTM project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/lstm-next-word-prediction.git
git push -u origin main
```

### Update Repository
```powershell
git add .
git commit -m "Description of changes"
git push origin main
```

---

## Streamlit Cloud Deployment

1. Go to: https://streamlit.io/cloud
2. Click: "New app"
3. Select:
   - Repository: `your-username/lstm-next-word-prediction`
   - Branch: `main`
   - File: `app.py`
4. Click: "Deploy"

**Your app URL:** `https://your-app-xxxxx.streamlit.app`

---

## Testing & Verification

### Test Imports
```powershell
python -c "import streamlit, tensorflow, numpy; print('✓ All imports OK')"
```

### Check File Existence
```powershell
# Check required files
Test-Path "next_word_lstm.h5"
Test-Path "tokenizer.pickle"
Test-Path "app.py"
Test-Path "requirements.txt"
```

### View File Sizes
```powershell
Get-Item next_word_lstm.h5 | Select-Object Length
Get-Item tokenizer.pickle | Select-Object Length
```

---

## Troubleshooting Commands

### Clear Cache
```powershell
rm -r ~/.streamlit/cache
```

### Reinstall Packages
```powershell
pip install -r requirements.txt --force-reinstall
```

### Check Python Version
```powershell
python --version  # Should be 3.8+
```

### Check TensorFlow
```powershell
python -c "import tensorflow; print(tensorflow.__version__)"
```

---

## Useful Git Commands

### View Status
```powershell
git status
```

### View Commit History
```powershell
git log --oneline
```

### Undo Last Commit
```powershell
git reset --soft HEAD~1
```

### Check Remote
```powershell
git remote -v
```

---

## Streamlit Specific Commands

### Run in Debug Mode
```powershell
streamlit run app.py --logger.level=debug
```

### Clear Cache
```powershell
streamlit cache clear
```

### Config Location
```powershell
# Windows:
~\.streamlit\config.toml
```

---

## Deployment Checklist

- [ ] All files present in project directory
- [ ] `requirements.txt` has correct versions
- [ ] `app.py` has no syntax errors
- [ ] Model files (.h5, .pickle) are not corrupted
- [ ] Project pushed to GitHub
- [ ] Streamlit Cloud app created
- [ ] App URL is accessible
- [ ] Test app with example phrases

---

## File Locations Reference

| File | Size | Purpose |
|------|------|---------|
| `app.py` | ~15KB | Main web app |
| `next_word_lstm.h5` | ~38MB | Trained model |
| `tokenizer.pickle` | ~280KB | Word tokenizer |
| `hamlet.txt` | ~160KB | Training data |
| `requirements.txt` | <1KB | Dependencies |
| `experiment.ipynb` | ~150KB | Training notebook |

---

## Port Information

**Default Streamlit Port:** 8501
- Local: `http://localhost:8501`
- Network: `http://YOUR_IP:8501`

---

## Environment Variables (Optional)

```powershell
# To set port number
$env:STREAMLIT_SERVER_PORT=8505

# To run in headless mode
$env:STREAMLIT_SERVER_HEADLESS=true
```

---

## Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| Port already in use | Change port: `streamlit run app.py --server.port 8502` |
| Model not found | Move `.h5` file to same directory as `app.py` |
| Slow startup | Wait, first run caches model (~2-3 sec) |
| Out of memory | Reduce `num_predictions` slider or restart Python |
| Import error | Run `pip install -r requirements.txt` |

---

## Windows vs Mac/Linux Commands

### Navigate Directory
```
Windows: cd "c:\path\to\project"
Mac/Linux: cd /path/to/project
```

### Activate Virtual Env
```
Windows: venv\Scripts\Activate.ps1
Mac/Linux: source venv/bin/activate
```

### Clear Screen
```
Windows: cls
Mac/Linux: clear
```

---

## Useful Links

- 🔗 Streamlit: https://streamlit.io
- 🔗 TensorFlow: https://tensorflow.org
- 🔗 GitHub: https://github.com
- 🔗 Python Docs: https://python.org/docs
- 🔗 NLTK: https://www.nltk.org

---

**📌 Keep this file handy for quick reference!**

*Last Updated: January 26, 2025*
