# MoodLens — Emotion Detector from Text

A 2nd-year B.Tech AI/ML project by Vikranth Reddy (Woxsen University).

TF-IDF (word unigrams, sublinear TF, L2) + multinomial logistic regression (C=1.0) trained on 300 hand-written sentences (50 per emotion: joy, sadness, anger, fear, surprise, love; 796 vocabulary words, seed 42).

- 5-fold CV accuracy: 72.0% (majority-class baseline: 16.7%)
- Live demo: https://moodlens-gilt.vercel.app (runs fully in the browser — no backend)
- `train.py` trains the model and writes `model.json`; `index.html` applies the same weights in JavaScript.
