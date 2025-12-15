import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns  # Advanced plotting
import os
import sys
from tqdm import tqdm
import logging
from tools.tool_global import extract_ground_truth

# ==============================================================================
# Visualization Style Configuration (Publication Quality)
# ==============================================================================
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['font.family'] = 'sans-serif'  # or 'serif' for LaTeX style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['grid.alpha'] = 0.3

# ==============================================================================
# Import XGBoost
# ==============================================================================
try:
    from xgboost import XGBClassifier
except ImportError:
    print("❌ Error: XGBoost not found. Please run `pip install xgboost`.")
    sys.exit(1)

# ==============================================================================
# Helper Modules Imports (Relative import compatibility)
# ==============================================================================
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from analysis_token_logprobs import AttributeConfidenceAnalyzer
except ImportError:
    pass


class PerformanceEvaluator:
    """Helper class to extract ground truth and evaluate correctness."""

    def __init__(self):
        pass

    def extract_ground_truth(self, data: Dict) -> Dict[str, str]:
        labels = {}
        # Uses the external tool to extract ground truth
        labels = extract_ground_truth(data)

        # Clean up
        for k, v in labels.items():
            if v == "": labels[k] = None

        return labels

    def fuzzy_match(self, extracted: str, ground_truth: str) -> Tuple[bool, float]:
        if not extracted and not ground_truth: return True, 1.0
        if not extracted or not ground_truth: return False, 0.0

        ex = str(extracted).lower().strip()
        gt = str(ground_truth).lower().strip()

        if ex == gt: return True, 1.0
        if ex in gt or gt in ex: return True, 0.9  # Substring match

        return False, 0.0


class ConfidenceDiscriminator:
    """
    XGBoost-based Confidence Discriminator.
    Predicts whether an LLM-extracted attribute is reliable (True Positive) or hallucinated (False Positive).
    """

    def __init__(self, model_type: str = 'xgboost'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.models = {}
        self.thresholds = {}
        self.feature_importance = {}
        self.feature_names = []

        # Store validation results for plotting
        self.validation_results = {}  # {attr: {'y_true': [], 'y_probs': []}}

        self.evaluator_helper = PerformanceEvaluator()

    def _get_empty_features(self) -> Dict[str, float]:
        """Returns a default feature dictionary with safe initial values."""
        return {
            # 1. Log Probability Features
            'avg_logprob': -10.0,
            'min_logprob': -10.0,
            'max_logprob': 0.0,
            'std_logprob': 0.0,
            'perplexity': 100.0,
            'certainty_score': 0.0,
            'low_conf_token_ratio': 1.0,
            'logprob_range': 10.0,

            # 2. Reasoning Quality Features
            'reasoning_length': 0,
            'reasoning_token_count': 0,
            'explicit_keywords': 0,  # "explicitly", "clearly"
            'uncertain_keywords': 0,  # "might", "could", "unclear"
            'negative_keywords': 0,  # "not found", "unknown"

            # 3. Structural/Content Features
            'has_value': 0.0,
            'value_length': 0,
            'value_has_numbers': 0.0,
            'value_has_special': 0.0,
            'is_generic': 0.0,  # "camera", "device", "unknown"

            # 4. Consistency Features
            'claimed_confidence': 0.0,
            'recommended_confidence': 0.0,
            'confidence_gap': 0.0  # abs(claimed - recommended)
        }

    def extract_features(self, confidence_analysis: Dict, attribute: str) -> Dict[str, float]:
        """
        Extracts a numerical feature vector from the LLM's confidence analysis output.
        """
        features = self._get_empty_features()

        if not confidence_analysis or attribute not in confidence_analysis:
            return features

        attr_data = confidence_analysis.get(attribute)
        if not attr_data or not isinstance(attr_data, dict):
            return features

        # --- 1. Log Probability Stats ---
        reasoning_conf = attr_data.get('reasoning_confidence', {})

        # Safely extract values with defaults
        features['avg_logprob'] = float(reasoning_conf.get('avg_logprob', -10.0))
        features['min_logprob'] = float(reasoning_conf.get('min_logprob', -10.0))
        features['max_logprob'] = float(reasoning_conf.get('max_logprob', 0.0))
        features['std_logprob'] = float(reasoning_conf.get('std_logprob', 0.0))
        features['perplexity'] = float(reasoning_conf.get('perplexity', 100.0) or 100.0)
        features['certainty_score'] = float(reasoning_conf.get('certainty_score', 0.0))
        features['low_conf_token_ratio'] = float(reasoning_conf.get('low_confidence_token_ratio', 0.0))
        features['logprob_range'] = features['max_logprob'] - features['min_logprob']

        # --- 2. Reasoning Text Analysis ---
        reasoning_text = str(attr_data.get('reasoning_text', ''))
        features['reasoning_length'] = len(reasoning_text)
        features['reasoning_token_count'] = int(attr_data.get('token_count', 0))

        reasoning_lower = reasoning_text.lower()
        features['explicit_keywords'] = sum(
            1 for word in ['explicitly', 'clearly', 'states', 'verbatim', 'pattern'] if word in reasoning_lower)
        features['uncertain_keywords'] = sum(
            1 for word in ['might', 'could', 'possibly', 'unclear', 'ambiguous', 'likely', 'infer'] if
            word in reasoning_lower)
        features['negative_keywords'] = sum(
            1 for word in ['not found', 'no mention', 'cannot', 'unable', 'missing', 'failed'] if
            word in reasoning_lower)

        # --- 3. Extracted Value Analysis ---
        extracted_value = attr_data.get('extracted_value')
        features['has_value'] = 1.0 if extracted_value and str(extracted_value).lower() not in ['none', 'null',
                                                                                                ''] else 0.0

        if features['has_value']:
            val_str = str(extracted_value)
            features['value_length'] = len(val_str)
            features['value_has_numbers'] = 1.0 if any(c.isdigit() for c in val_str) else 0.0
            features['value_has_special'] = 1.0 if any(c in '.-_/' for c in val_str) else 0.0

            generic_words = ['unknown', 'device', 'generic', 'recording', 'router', 'camera', 'server', 'n/a']
            features['is_generic'] = 1.0 if any(g == val_str.lower() for g in generic_words) else 0.0

        # --- 4. Confidence Self-Report ---
        features['claimed_confidence'] = float(attr_data.get('claimed_confidence', 0.0))
        features['recommended_confidence'] = float(attr_data.get('recommended_confidence', 0.0))
        features['confidence_gap'] = abs(features['claimed_confidence'] - features['recommended_confidence'])

        return features

    def prepare_training_data(self, jsonl_file: str, confidence_analyzer: AttributeConfidenceAnalyzer,
                              debug: bool = True, index_list: list = []) -> Dict[str, pd.DataFrame]:
        """
        Reads LLM results, extracts features, and creates labeled datasets for training.
        Label 1 (Positive): LLM extraction matches Ground Truth.
        Label 0 (Negative): LLM extraction does NOT match Ground Truth (Hallucination).
        """
        print(f"Preparing training data from: {jsonl_file}")

        data_by_attr = {
            'brand': [],
            'model': [],
            'firmware_version': []
        }

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        valid_count = 0

        for line in tqdm(lines, desc="Processing samples"):
            try:
                data = json.loads(line)
                index = data.get("index", data.get("sample_id"))

                # Filter by index_list if provided
                if index_list and index not in index_list:
                    continue

                # 1. Get Ground Truth
                ground_truth = self.evaluator_helper.extract_ground_truth(data)

                # 2. Get LLM Output & Raw Logprobs
                # If missing logprobs, try to re-analyze or skip
                # Here we assume we can call the analyzer
                raw_response = data.get('raw_outputs', {})

                # confidence_analysis = confidence_analyzer.analyze_full_output(raw_response, logprobs)
                confidence_analysis_l = []
                pred_l = []
                
                # Try First Pass
                try:
                    confidence_analysis = raw_response.get('first_pass').get('parsed').get('confidence_analysis')
                    preds = raw_response.get('first_pass').get('parsed').get('parsed_json', {})
                    confidence_analysis_l.append(confidence_analysis)
                    pred_l.append(preds)
                except Exception as e:
                    # Note: Failed to get confidence_analysis, will rely only on existing data in file
                    print(f"Note: Failed to get confidence_analysis ({e}), relying on existing file data only")
                    pass
                
                # Try Second Pass
                try:
                    confidence_analysis = raw_response.get('second_pass').get('parsed').get('confidence_analysis')
                    preds = raw_response.get('second_pass').get('parsed').get('parsed_json', {})
                    confidence_analysis_l.append(confidence_analysis)
                    pred_l.append(preds)
                except Exception as e:
                    # Note: Failed to get confidence_analysis, will rely only on existing data in file
                    print(f"Note: Failed to get confidence_analysis ({e}), relying on existing file data only")
                    pass
                    
                # 3. Process each attribute
                if not confidence_analysis_l:
                    continue
                for confidence_analysis, preds in zip(confidence_analysis_l, pred_l):
                    for attr in ['brand', 'model', 'firmware_version']:
                        gt_val = ground_truth.get(attr)

                        # Get prediction for this attribute
                        # Handle different JSON structures safely
                        pred_val = None
                        if attr in preds:
                            if isinstance(preds[attr], dict):
                                pred_val = preds[attr].get('value')
                            else:
                                pred_val = preds[attr]  # assuming string

                        # Extract Features
                        feats = self.extract_features(confidence_analysis, attr)

                        # Determine Label
                        # We only train on samples where GT exists.
                        # Case A: GT exists.
                        #    - Pred matches GT -> Label 1 (Reliable)
                        #    - Pred mismatches GT -> Label 0 (Hallucination)
                        # Case B: GT does not exist (None).
                        #    - Pred is None -> Correct (but no info to learn from features usually, or learn to predict None?)
                        #    - Pred exists -> Label 0 (Hallucination)

                        # Simplification: Only train on samples where LLM produced a value (Pred is not None).
                        # Because if LLM produced nothing, we don't need a discriminator to reject it.
                        if pred_val:
                            is_match, score = self.evaluator_helper.fuzzy_match(pred_val, gt_val)

                            # Strict logic:
                            # If GT is None, and Pred exists -> Hallucination (Label 0)
                            # If GT exists, and Pred matches -> Reliable (Label 1)
                            # If GT exists, and Pred mismatches -> Hallucination (Label 0)

                            label = 1 if (gt_val and is_match) else 0

                            # Add label to features
                            feats['label'] = label
                            data_by_attr[attr].append(feats)

                    valid_count += 1

            except Exception as e:
                if debug and valid_count < 5:
                    print(f"Error processing line: {e}")
                continue

        # Convert to DataFrames
        datasets = {}
        for attr, rows in data_by_attr.items():
            if rows:
                df = pd.DataFrame(rows)
                datasets[attr] = df
                print(f"  {attr}: {len(df)} samples (Pos: {df['label'].sum()}, Neg: {len(df) - df['label'].sum()})")
            else:
                print(f"  {attr}: No valid training samples found.")

        return datasets

    def train(self, training_data: Dict[str, pd.DataFrame]):
        """
        Trains XGBoost classifiers for each attribute and stores validation results.
        """
        print("\n" + "=" * 80)
        print("TRAINING DISCRIMINATORS (XGBoost)")
        print("=" * 80)

        for attr, df in training_data.items():
            print(f"\nTraining Model for Attribute: {attr.upper()}")

            if 'label' not in df.columns or len(df) < 10:
                print(f"Skipping {attr}: Not enough data.")
                continue

            # Prepare X and y
            y = df['label']
            X_raw = df.drop(columns=['label'])

            # Save feature names once
            if not self.feature_names:
                self.feature_names = list(X_raw.columns)

            # Handle NaNs and Infinity
            X_cleaned = np.nan_to_num(X_raw.astype(np.float64), nan=0.0, posinf=1e6, neginf=-1e6)

            # Scale features
            X_scaled = self.scaler.fit_transform(X_cleaned)

            # Split Data (Stratified)
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )

            # Store validation data for plotting later
            self.validation_results[attr] = {
                'y_true': y_val,
                'y_probs': None  # To be filled
            }

            # Handle Class Imbalance
            num_pos = y_train.sum()
            num_neg = len(y_train) - num_pos
            scale_weight = num_neg / num_pos if num_pos > 0 else 1.0
            print(f"  Class Balance: Pos={num_pos}, Neg={num_neg}, Scale Weight={scale_weight:.2f}")

            # Initialize XGBoost
            model = XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_weight,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )

            # Train
            model.fit(X_train, y_train)

            # Evaluate
            train_acc = model.score(X_train, y_train)
            val_acc = model.score(X_val, y_val)
            print(f"  Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            # Get Probabilities for Validation Set
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            self.validation_results[attr]['y_probs'] = y_pred_proba

            # Find Optimal Threshold (Maximizing F1)
            optimal_thresh = self._optimize_threshold(y_val, y_pred_proba)
            print(f"  Optimal Threshold: {optimal_thresh:.4f}")

            # Store Model
            self.models[attr] = model
            self.thresholds[attr] = optimal_thresh

            # Store Feature Importance
            if hasattr(model, 'feature_importances_'):
                importances = dict(zip(self.feature_names, model.feature_importances_))
                self.feature_importance[attr] = importances

    def _optimize_threshold(self, y_true, y_pred_proba):
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.linspace(0.1, 0.95, 50):
            y_pred = (y_pred_proba >= thresh).astype(int)
            # Calculate F1
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / y_true.sum() if y_true.sum() > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        return best_thresh

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'thresholds': self.thresholds,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance
            }, f)
        print(f"Discriminator saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        instance = cls()
        instance.models = data['models']
        instance.thresholds = data['thresholds']
        instance.scaler = data['scaler']
        instance.feature_names = data.get('feature_names', [])
        instance.feature_importance = data.get('feature_importance', {})
        return instance

    def predict(self, confidence_analysis: Dict, attribute: str) -> Dict:
        """
        Predicts whether the extraction for 'attribute' is reliable.
        Returns: {'decision': 'ACCEPT'/'REJECT', 'confidence_score': float}
        """
        if attribute not in self.models:
            # Fallback if no model for this attribute (e.g. unknown attr)
            return {'decision': 'ACCEPT', 'confidence_score': 0.5, 'is_reliable': True}

        # 1. Extract Features
        feats_dict = self.extract_features(confidence_analysis, attribute)

        # 2. Prepare Vector
        # Ensure order matches training
        if not self.feature_names:
            # Should be loaded
            feature_vector = np.array(list(feats_dict.values())).reshape(1, -1)
        else:
            feature_vector = np.array([feats_dict[f] for f in self.feature_names]).reshape(1, -1)

        # 3. Scale
        X_input = np.nan_to_num(feature_vector.astype(np.float64), nan=0.0)
        X_scaled = self.scaler.transform(X_input)

        # 4. Predict
        model = self.models[attribute]
        prob = model.predict_proba(X_scaled)[0, 1]
        threshold = self.thresholds.get(attribute, 0.5)

        decision = "ACCEPT" if prob >= threshold else "REJECT"

        return {
            'decision': decision,
            'confidence_score': float(prob),
            'is_reliable': bool(prob >= threshold),
            'threshold_used': float(threshold)
        }

    # ==========================================================================
    # VISUALIZATION METHODS (Publication Quality)
    # ==========================================================================

    def plot_all_analysis(self, output_dir: str):
        """Generates all analysis plots."""
        os.makedirs(output_dir, exist_ok=True)
        print("\nGenerating Analysis Plots...")
        self.plot_feature_importance(output_dir)
        self.plot_threshold_impact(output_dir)
        self.plot_score_distribution(output_dir)

    def plot_feature_importance(self, output_dir: str):
        """Top-10 Feature Importance Bar Plot."""
        for attr, importance_dict in self.feature_importance.items():
            if not importance_dict: continue

            df_imp = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance'])
            df_imp = df_imp.sort_values(by='Importance', ascending=False).head(10)

            plt.figure(figsize=(8, 6))
            ax = sns.barplot(x='Importance', y='Feature', data=df_imp, palette='viridis', edgecolor='black')

            for i, v in enumerate(df_imp['Importance']):
                ax.text(v + 0.002, i, f"{v:.3f}", color='black', va='center', fontsize=10)

            plt.title(f'Feature Importance: {attr.title()}', fontweight='bold')
            plt.xlabel('Importance Score (Gain)')
            plt.ylabel('')
            sns.despine()
            plt.tight_layout()

            plt.savefig(os.path.join(output_dir, f"feature_importance_{attr}.pdf"))
            plt.close()
            print(f"  - Saved feature_importance_{attr}.pdf")

    def plot_threshold_impact(self, output_dir: str):
        """Plot TP, TN, FP, FN curves vs Threshold."""
        
        for attr, res in self.validation_results.items():
            y_true = np.array(res['y_true'])
            y_probs = np.array(res['y_probs'])
            if len(y_true) == 0: continue

            thresholds = np.linspace(0.01, 0.99, 100)
            tps, tns, fps, fns = [], [], [], []

            for t in thresholds:
                y_pred = (y_probs >= t).astype(int)
                tn, fp, fn, tp_val = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                tps.append(tp_val)
                tns.append(tn)
                fps.append(fp)
                fns.append(fn)

            plt.figure(figsize=(8, 6))
            plt.plot(thresholds, tps, label='TP (Correct Accept)', color='#2ecc71', linewidth=2.5)
            plt.plot(thresholds, fns, label='FN (Missed Good)', color='#f39c12', linewidth=2.5, linestyle='--')
            plt.plot(thresholds, tns, label='TN (Correct Reject)', color='#3498db', linewidth=2.5)
            plt.plot(thresholds, fps, label='FP (Hallucination)', color='#e74c3c', linewidth=2.5, linestyle='--')

            opt_thresh = self.thresholds.get(attr, 0.5)
            plt.axvline(x=opt_thresh, color='gray', linestyle=':', label=f'Optimal ({opt_thresh:.2f})')

            plt.title(f'Threshold Impact: {attr.title()}', fontweight='bold')
            plt.xlabel('Confidence Threshold')
            plt.ylabel('Count')
            plt.legend()
            sns.despine()
            plt.tight_layout()

            plt.savefig(os.path.join(output_dir, f"threshold_impact_{attr}.pdf"))
            plt.close()
            print(f"  - Saved threshold_impact_{attr}.pdf")

    def plot_score_distribution(self, output_dir: str):
        """KDE Plot of scores for Positive vs Negative samples."""
        for attr, res in self.validation_results.items():
            y_true = np.array(res['y_true'])
            y_probs = np.array(res['y_probs'])

            df_plot = pd.DataFrame({
                'Score': y_probs,
                'Label': ['Positive' if y == 1 else 'Negative' for y in y_true]
            })

            plt.figure(figsize=(8, 5))
            sns.kdeplot(data=df_plot, x='Score', hue='Label', fill=True,
                        common_norm=False, palette=['#e74c3c', '#2ecc71'], alpha=0.4)

            plt.title(f'Score Distribution: {attr.title()}', fontweight='bold')
            plt.xlim(0, 1)
            sns.despine()
            plt.tight_layout()

            plt.savefig(os.path.join(output_dir, f"score_distribution_{attr}.pdf"))
            plt.close()
            print(f"  - Saved score_distribution_{attr}.pdf")


# ==============================================================================
# Main Execution Entry Point
# ==============================================================================
def main():
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    INPUT_FILE = r"../agent/results/relabel_merged_cydar_data_multi_agent_results_sft_Qwen2___5_7B_Instruct_1130.jsonl"
    OUTPUT_DIR = r"./discriminator_results_plots_1204_xgboost"
    MODEL_PATH = os.path.join(OUTPUT_DIR, "discriminator_model_optimized_xgb_1204.pkl")

    # Optional: Filter specific IDs if needed (e.g. from a separate file)
    index_list = []
    # with open('some_id_list.json') as f: index_list = json.load(f)

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    print(">>> Initializing Discriminator Pipeline...")
    # Initialize Analyzer (Assumes analysis_token_logprobs.py is available)
    try:
        analyzer = AttributeConfidenceAnalyzer()
    except NameError:
        # Fallback if import failed
        print("Warning: AttributeConfidenceAnalyzer not found. Using Mock.")

        class MockAnalyzer:
            def analyze_full_output(self, r, l): return {}

        analyzer = MockAnalyzer()

    discriminator = ConfidenceDiscriminator(model_type='xgboost')

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    print(">>> Preparing Data...")
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    training_data = discriminator.prepare_training_data(
        INPUT_FILE, analyzer, debug=True, index_list=index_list
    )

    if not training_data:
        print("❌ No training data produced.")
        return

    print(">>> Training Models...")
    discriminator.train(training_data)

    # -------------------------------------------------------------------------
    # Saving
    # -------------------------------------------------------------------------
    discriminator.save(MODEL_PATH)

    # -------------------------------------------------------------------------
    # Plotting (The New Feature)
    # -------------------------------------------------------------------------
    print(">>> Generating Plots...")
    discriminator.plot_all_analysis(OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("✅ Pipeline Completed Successfully.")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Plots: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
