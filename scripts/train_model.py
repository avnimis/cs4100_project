import numpy as np
import pickle
from scipy.sparse import load_npz
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ── load features ─────────────────────────────────────────────────────────────

def load_artifacts(feat_dir="data/features"):
    X  = load_npz(f"{feat_dir}/X_features.npz")
    y  = np.load(f"{feat_dir}/y_labels.npy")
    with open(f"{feat_dir}/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return X, y, le

# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(name, model, X, y, le, cv):
    """Cross-validated evaluation — avoids train/test split on small datasets."""
    y_pred = cross_val_predict(model, X, y, cv=cv)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(classification_report(y, y_pred, target_names=le.classes_, zero_division=0))

    # confusion matrix
    cm = confusion_matrix(y, y_pred)
    print("Confusion matrix (rows=actual, cols=predicted):")
    header = "          " + "  ".join(f"{c[:5]:>5}" for c in le.classes_)
    print(header)
    for i, row in enumerate(cm):
        label = f"{le.classes_[i][:10]:<10}"
        print(label + "  ".join(f"{v:>5}" for v in row))

    return y_pred

# ── train final model ─────────────────────────────────────────────────────────

def train_final(model, X, y):
    model.fit(X, y)
    return model

# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X, y, le = load_artifacts()
    print(f"Loaded: {X.shape[0]} emails, {X.shape[1]} features, {len(le.classes_)} classes")
    print(f"Classes: {list(le.classes_)}")

    # use stratified k-fold — keeps class ratios in each fold
    # with only 25 samples, 5-fold gives ~5 test emails per fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Complement Naive Bayes": ComplementNB(alpha=0.5),
        "Logistic Regression":    LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    }

    results = {}
    for name, model in models.items():
        y_pred = evaluate(name, model, X, y, le, cv)
        results[name] = y_pred

    # train final model on all data and save
    print("\nTraining final Logistic Regression on all data...")
    final_model = train_final(LogisticRegression(max_iter=1000, C=1.0, random_state=42), X, y)
    with open("data/features/final_model.pkl", "wb") as f:
        pickle.dump(final_model, f)
    print("Saved → data/features/final_model.pkl")

    # quick sanity check — predict on a few examples
    print("\n--- Sanity check: predicting on training data ---")
    with open("data/emails/all_emails_clean.json") as f:
        import json
        emails = [e for e in json.load(f) if e.get("label") not in (None, "unknown")]

    with open("data/features/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from extract_features import build_features
    X_all, _, _, _ = build_features(emails, vectorizer=vectorizer, fit=False)
    preds = final_model.predict(X_all)
    pred_labels = le.inverse_transform(preds)

    print(f"{'Subject':<45} {'Actual':<15} {'Predicted':<15}")
    print("-" * 75)
    for e, pred in zip(emails, pred_labels):
        subj   = e.get("clean_subject", "")[:44]
        actual = e.get("label", "?")
        match  = "✓" if actual == pred else "✗"
        print(f"{subj:<45} {actual:<15} {pred:<15} {match}")