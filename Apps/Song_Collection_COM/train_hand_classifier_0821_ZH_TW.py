"""Train and evaluate hand classification models on HoloGrip Song 2 Ground-Truth dataset.

Features are extracted from the calibrated +-80ms IMU window on the shared song_time_ms timeline.
Uses 5-Fold Stratified Cross-Validation across multiple classifiers:
- Random Forest
- Support Vector Machine (RBF / Linear)
- Multi-Layer Perceptron (MLP)
- Gradient Boosting
- Logistic Regression
"""

import os
import csv
import json
import numpy as np
from datetime import datetime, timezone
import joblib

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# Paths
CLEANED_DIR = r'C:\Users\Typelin_Station\Desktop\HoloGrip\Data\Derived\Song_Collection_COM\S20260812_P01_song02_T1127_cleaned'
GT_CSV_PATH = os.path.join(CLEANED_DIR, 'Song2_ground_truth_labels_final_0821.csv')
RAW_CSV_PATH = r'C:\Users\Typelin_Station\Desktop\HoloGrip\Data\Raw\Song_Collection_COM\S20260812_P01_song02_raw_100hz_20260812_112101.csv'
OUT_DIR = CLEANED_DIR
WINDOW_HALF_MS = 80.0  # +-80ms

def load_data():
    print(f"Loading Ground-Truth labels from {GT_CSV_PATH}...")
    with open(GT_CSV_PATH, 'r', encoding='utf-8') as f:
        gt_rows = list(csv.DictReader(f))

    print(f"Loading Raw 100Hz IMU CSV from {RAW_CSV_PATH}...")
    samples_by_hand = {'L': [], 'R': []}
    with open(RAW_CSV_PATH, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = row['hand']
            if h in samples_by_hand:
                samples_by_hand[h].append({
                    'song_time_ms': float(row['song_time_ms']),
                    'ax': float(row['ax_g']),
                    'ay': float(row['ay_g']),
                    'az': float(row['az_g']),
                    'mag': float(row['accel_magnitude_g']),
                    'yaw': float(row['cal_yaw_deg']),
                    'pitch': float(row['cal_pitch_deg']),
                    'roll': float(row['roll_deg']),
                })

    for h in ['L', 'R']:
        samples_by_hand[h].sort(key=lambda x: x['song_time_ms'])

    return gt_rows, samples_by_hand

def extract_features(gt_rows, samples_by_hand):
    X = []
    y = []
    metadata = []

    feature_names = [
        # Left Hand features
        'L_max_mag', 'L_mean_mag', 'L_std_mag', 'L_energy', 'L_max_ax', 'L_max_ay', 'L_max_az', 'L_jerk_max',
        # Right Hand features
        'R_max_mag', 'R_mean_mag', 'R_std_mag', 'R_energy', 'R_max_ax', 'R_max_ay', 'R_max_az', 'R_jerk_max',
        # Differential / Interaction features
        'energy_diff', 'energy_ratio_asym', 'max_mag_diff', 'mean_mag_diff',
        'velocity', 'drum_zone_id'
    ]

    for row in gt_rows:
        final_h = row['final_hand']
        if final_h not in ['L', 'R']:
            # Skip dual hands for binary single-hand classifier baseline
            continue

        label = 0 if final_h == 'L' else 1  # 0: Left, 1: Right
        center_t = float(row['csv_center_ms'])
        start_t = center_t - WINDOW_HALF_MS
        end_t = center_t + WINDOW_HALF_MS

        l_slice = [s for s in samples_by_hand['L'] if start_t <= s['song_time_ms'] <= end_t]
        r_slice = [s for s in samples_by_hand['R'] if start_t <= s['song_time_ms'] <= end_t]

        # Extract features for Left
        if l_slice:
            l_mags = np.array([s['mag'] for s in l_slice])
            l_axs = np.array([s['ax'] for s in l_slice])
            l_ays = np.array([s['ay'] for s in l_slice])
            l_azs = np.array([s['az'] for s in l_slice])
            
            L_max_mag = float(np.max(l_mags))
            L_mean_mag = float(np.mean(l_mags))
            L_std_mag = float(np.std(l_mags))
            L_energy = float(np.sum(np.abs(l_mags - 1.0)))
            L_max_ax = float(np.max(np.abs(l_axs)))
            L_max_ay = float(np.max(np.abs(l_ays)))
            L_max_az = float(np.max(np.abs(l_azs)))
            L_jerk_max = float(np.max(np.abs(np.diff(l_mags)))) if len(l_mags) > 1 else 0.0
        else:
            L_max_mag, L_mean_mag, L_std_mag, L_energy, L_max_ax, L_max_ay, L_max_az, L_jerk_max = 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Extract features for Right
        if r_slice:
            r_mags = np.array([s['mag'] for s in r_slice])
            r_axs = np.array([s['ax'] for s in r_slice])
            r_ays = np.array([s['ay'] for s in r_slice])
            r_azs = np.array([s['az'] for s in r_slice])
            
            R_max_mag = float(np.max(r_mags))
            R_mean_mag = float(np.mean(r_mags))
            R_std_mag = float(np.std(r_mags))
            R_energy = float(np.sum(np.abs(r_mags - 1.0)))
            R_max_ax = float(np.max(np.abs(r_axs)))
            R_max_ay = float(np.max(np.abs(r_ays)))
            R_max_az = float(np.max(np.abs(r_azs)))
            R_jerk_max = float(np.max(np.abs(np.diff(r_mags)))) if len(r_mags) > 1 else 0.0
        else:
            R_max_mag, R_mean_mag, R_std_mag, R_energy, R_max_ax, R_max_ay, R_max_az, R_jerk_max = 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Asymmetry and difference
        total_energy = L_energy + R_energy
        energy_diff = R_energy - L_energy
        energy_ratio_asym = (R_energy - L_energy) / (total_energy + 1e-6)
        max_mag_diff = R_max_mag - L_max_mag
        mean_mag_diff = R_mean_mag - L_mean_mag

        vel = float(row.get('velocity', 64))
        
        # Zone map
        zone_map = {"小鼓": 0, "高音 Tom": 1, "中音 Tom": 2, "落地 Tom": 3, "Hi-Hat": 4, "Crash": 5, "Ride": 6}
        drum_zone = zone_map.get(row['drum'], 0)

        feat_vector = [
            L_max_mag, L_mean_mag, L_std_mag, L_energy, L_max_ax, L_max_ay, L_max_az, L_jerk_max,
            R_max_mag, R_mean_mag, R_std_mag, R_energy, R_max_ax, R_max_ay, R_max_az, R_jerk_max,
            energy_diff, energy_ratio_asym, max_mag_diff, mean_mag_diff,
            vel, drum_zone
        ]

        X.append(feat_vector)
        y.append(label)
        metadata.append(row)

    return np.array(X), np.array(y), feature_names, metadata

def run_training():
    gt_rows, samples_by_hand = load_data()
    X, y, feature_names, metadata = extract_features(gt_rows, samples_by_hand)

    print(f"\nFeature matrix X shape: {X.shape}, labels y shape: {y.shape}")
    print(f"Class distribution: Left (0) = {np.sum(y == 0)} ({np.mean(y == 0)*100:.1f}%), Right (1) = {np.sum(y == 1)} ({np.mean(y == 1)*100:.1f}%)")

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "SVM (RBF)": Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))]),
        "SVM (Linear)": Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='linear', C=1.0, probability=True, random_state=42))]),
        "MLP Classifier": Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))]),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        "Logistic Regression": Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(random_state=42))]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("\n=======================================================")
    print("      5-Fold Stratified Cross-Validation Results       ")
    print("=======================================================")

    best_model_name = None
    best_f1 = 0.0

    for name, model in models.items():
        scores = cross_validate(
            model, X, y, cv=cv,
            scoring=['accuracy', 'precision', 'recall', 'f1']
        )
        acc_mean = float(np.mean(scores['test_accuracy']))
        acc_std = float(np.std(scores['test_accuracy']))
        prec_mean = float(np.mean(scores['test_precision']))
        rec_mean = float(np.mean(scores['test_recall']))
        f1_mean = float(np.mean(scores['test_f1']))
        f1_std = float(np.std(scores['test_f1']))

        results[name] = {
            'accuracy_mean': acc_mean,
            'accuracy_std': acc_std,
            'precision_mean': prec_mean,
            'recall_mean': rec_mean,
            'f1_mean': f1_mean,
            'f1_std': f1_std
        }

        print(f"[{name:20s}] Acc: {acc_mean*100:5.2f}% (+/- {acc_std*100:4.2f}%) | F1: {f1_mean*100:5.2f}% | Prec: {prec_mean*100:5.2f}% | Rec: {rec_mean*100:5.2f}%")

        if f1_mean > best_f1:
            best_f1 = f1_mean
            best_model_name = name

    # Train best model on all data and save
    print(f"\nBest Model: {best_model_name} (F1 = {best_f1*100:.2f}%)")
    best_clf = models[best_model_name]
    best_clf.fit(X, y)
    
    # Save model artifact
    model_save_path = os.path.join(OUT_DIR, 'hologrip_song2_best_hand_classifier.joblib')
    joblib.dump(best_clf, model_save_path)
    print(f"Saved trained model to: {model_save_path}")

    # Random forest feature importances
    rf = models["Random Forest"]
    rf.fit(X, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\n--- Top 10 Feature Importances (Random Forest) ---")
    feat_ranks = []
    for f in range(min(10, len(feature_names))):
        idx = indices[f]
        print(f"  {f+1:2d}. {feature_names[idx]:20s} ({importances[idx]*100:5.2f}%)")
        feat_ranks.append({"rank": f+1, "feature": feature_names[idx], "importance": float(importances[idx])})

    # Save benchmark report json
    report_json_path = os.path.join(OUT_DIR, 'model_benchmark_report_0821.json')
    report_data = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Song2 (307 binary single-hand samples + 4 dual)",
        "window_half_ms": WINDOW_HALF_MS,
        "sample_count": len(X),
        "class_counts": {"Left (0)": int(np.sum(y == 0)), "Right (1)": int(np.sum(y == 1))},
        "cv_folds": 5,
        "benchmark_results": results,
        "best_model": best_model_name,
        "top_features": feat_ranks,
        "saved_model_path": model_save_path
    }
    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved benchmark report to: {report_json_path}")
    return report_data

if __name__ == "__main__":
    run_training()
