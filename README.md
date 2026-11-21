# Sri Lankan Used Vehicle Price – Ensemble Learning Dashboard
    pip install -r requirements.txt
    python train.py          # trains RF, XGBoost, ANN, hybrid -> artifacts/
    streamlit run app.py
Prices are in units of LKR 100,000 (lakhs). Hybrid = weighted average of RF and ANN, weight tuned on validation split.
