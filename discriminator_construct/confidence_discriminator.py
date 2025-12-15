import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import json
import pickle
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

# ==============================================================================
# New: Import XGBoost
# ==============================================================================
try:
    from xgboost import XGBClassifier
except ImportError:
    print("❌ Error: xgboost library not found. Please run `pip install xgboost` to install.")
    sys.exit(1)

# ==============================================================================
# Import auxiliary modules (Attempt compatibility with different directory structures)
# ==============================================================================
try:
    # Try direct import
    from performance_analysis.performance_analysis_new_1114 import PerformanceEvaluator
    from analysis_token_logprobs import AttributeConfidenceAnalyzer
except ImportError:
    try:
        # Try importing from current directory (if user places files in the same level)
        sys.path.append(os.getcwd())
        from performance_analysis.performance_analysis_new_1114 import PerformanceEvaluator
        # Assuming AttributeConfidenceAnalyzer is also at the same level or needs similar handling
        from analysis_token_logprobs import AttributeConfidenceAnalyzer
    except ImportError:
        print("❌ Error: Unable to import performance_analysis_new_1114 or analysis_token_logprobs.")
        print("Please ensure related files are in the correct directory.")


        # To prevent IDE static check errors, define empty placeholder classes
        class PerformanceEvaluator:
            def extract_ground_truth(self, data): return {}

            def fuzzy_match(self, a, b): return False, 0.0

            def evaluate(self, f): return {}, pd.DataFrame()

            def generate_report(self, m): return ""


        class AttributeConfidenceAnalyzer:
            def analyze_full_output(self, r, t): return {}


class ConfidenceDiscriminator:
    """
    Confidence Discriminator: Predicts the reliability of extracted information
    based on multi-dimensional features from the LLM's self-analysis.

    Trains separate models for: Brand, Model, Firmware Version.
    """

    def __init__(self, model_type: str = 'xgboost'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        # Storage for separate models per attribute
        self.models = {}
        self.thresholds = {}
        self.feature_importance = {}
        self.feature_names = []

        # Helper from the new evaluation script
        self.evaluator_helper = PerformanceEvaluator()

    def extract_features(self,
                         confidence_analysis: Dict,
                         attribute: str) -> Dict[str, float]:
        """
        Extracts features based on the specific JSON structure provided.
        (Logic kept exactly as in uploaded file)
        """
        # Initialize default features
        features = self._get_empty_features()

        if not confidence_analysis or attribute not in confidence_analysis:
            return features

        attr_data = confidence_analysis.get(attribute)
        if not attr_data or not isinstance(attr_data, dict):
            return features

        # --- 1. Logprob & Perplexity Features (from reasoning_confidence) ---
        reasoning_conf = attr_data.get('reasoning_confidence', {})
        if not isinstance(reasoning_conf, dict):
            reasoning_conf = {}

        features['avg_logprob'] = float(reasoning_conf.get('avg_logprob', -10.0))
        features['min_logprob'] = float(reasoning_conf.get('min_logprob', -10.0))
        features['max_logprob'] = float(reasoning_conf.get('max_logprob', 0.0))
        features['std_logprob'] = float(reasoning_conf.get('std_logprob', 0.0))
        features['perplexity'] = float(reasoning_conf.get('perplexity', 100.0) or 100.0)  # Handle None
        features['certainty_score'] = float(reasoning_conf.get('certainty_score', 0.0))
        features['low_conf_token_ratio'] = float(reasoning_conf.get('low_confidence_token_ratio', 0.0))
        features['logprob_range'] = features['max_logprob'] - features['min_logprob']

        # --- 2. Reasoning Text Features ---
        reasoning_text = str(attr_data.get('reasoning_text', ''))
        features['reasoning_length'] = len(reasoning_text)
        features['reasoning_token_count'] = int(attr_data.get('token_count', 0))

        reasoning_lower = reasoning_text.lower()
        features['explicit_keywords'] = sum(
            1 for word in ['explicitly', 'clearly', 'states', 'verbatim'] if word in reasoning_lower)
        features['uncertain_keywords'] = sum(
            1 for word in ['might', 'could', 'possibly', 'unclear', 'ambiguous', 'likely'] if word in reasoning_lower)
        features['negative_keywords'] = sum(
            1 for word in ['not found', 'no mention', 'cannot', 'unable', 'missing'] if word in reasoning_lower)

        # --- 3. Extracted Value Features ---
        extracted_value = attr_data.get('extracted_value')
        features['has_value'] = 1.0 if extracted_value else 0.0

        if extracted_value:
            val_str = str(extracted_value)
            features['value_length'] = len(val_str)
            features['value_has_numbers'] = 1.0 if any(c.isdigit() for c in val_str) else 0.0
            features['value_has_special'] = 1.0 if any(c in '.-_/' for c in val_str) else 0.0

            # Generic word detection (often indicates hallucinations)
            generic_words = ['unknown', 'device', 'generic', 'recording', 'router', 'camera', 'server', 'n/a']
            features['is_generic'] = 1.0 if any(g == val_str.lower() for g in generic_words) else 0.0
        else:
            # If value is null, these features remain 0
            pass

        # --- 4. Confidence Consistency Features ---
        features['claimed_confidence'] = float(attr_data.get('claimed_confidence', 0.0))
        features['recommended_confidence'] = float(attr_data.get('recommended_confidence', 0.0))

        # Gap between what the model claims and what the logprobs suggest
        features['confidence_gap'] = abs(features['claimed_confidence'] - features['recommended_confidence'])

        return features

    def _get_empty_features(self) -> Dict[str, float]:
        """Returns a default feature vector with safe initial values."""
        return {
            'avg_logprob': -10.0, 'min_logprob': -10.0, 'max_logprob': 0.0,
            'std_logprob': 0.0, 'perplexity': 100.0,
            'certainty_score': 0.0, 'low_conf_token_ratio': 1.0, 'logprob_range': 10.0,
            'reasoning_length': 0, 'reasoning_token_count': 0,
            'explicit_keywords': 0, 'uncertain_keywords': 0, 'negative_keywords': 0,
            'has_value': 0.0, 'value_length': 0,
            'value_has_numbers': 0.0, 'value_has_special': 0.0,
            'is_generic': 0.0,
            'claimed_confidence': 0.0, 'recommended_confidence': 0.0,
            'confidence_gap': 0.0
        }

    def prepare_training_data(self, jsonl_file: str, confidence_analyzer: AttributeConfidenceAnalyzer,
                              debug: bool = True, index_list: list = []) -> Dict[str, pd.DataFrame]:
        """Parses the JSONL file, extracts Ground Truth, matches Predictions, and builds feature sets."""
        print(f"Preparing training data from: {jsonl_file}")

        data_containers = {
            'brand': [],
            'model': [],
            'firmware_version': []
        }

        if not os.path.exists(jsonl_file):
            print(f"Error: File not found {jsonl_file}")
            return {}

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Processing Samples"):
            try:
                data = json.loads(line)
                index = data.get('index', data.get('sample_id'))
                # Exclude data in the list (e.g., SFT training data)
                if index not in index_list:
                    continue

                # 1. Extract Ground Truth using the new logic
                gt_labels = self.evaluator_helper.extract_ground_truth(data)

                # Correct brand labels
                if gt_labels.get('brand') and gt_labels.get('brand').lower() not in data.get('banner', '').lower():
                    gt_labels['brand'] = ''

                # 2. Get Confidence Analysis Data from Multiple Passes
                analysis_sources = []

                # Check for raw_outputs structure (First Pass & Second Pass)
                raw_outputs = data.get('raw_outputs', {})
                if raw_outputs:
                    # First Pass
                    if 'first_pass' in raw_outputs:
                        parsed = raw_outputs['first_pass'].get('parsed', {})
                        if parsed and 'confidence_analysis' in parsed:
                            if parsed['confidence_analysis']["brand"]["token_count"] == 0:
                                print("Skipping sample with brand token count 0")
                            else:
                                analysis_sources.append(parsed['confidence_analysis'])

                    # Second Pass
                    if 'second_pass' in raw_outputs:
                        parsed = raw_outputs['second_pass'].get('parsed', {})
                        if parsed and 'confidence_analysis' in parsed:
                            analysis_sources.append(parsed['confidence_analysis'])

                # Fallback: Check root level confidence_analysis (Old format)
                if not analysis_sources:
                    # conf_analysis = data.get('confidence_analysis', {})
                    conf_analysis = {}
                    if confidence_analyzer:
                        # Try to compute if missing
                        conf_analysis = confidence_analyzer.analyze_full_output(
                            data.get('original_data').get('parsed_output', {}).get('raw_response', ''),
                            data.get('original_data').get('token_logprobs', [])
                        )
                    if conf_analysis:
                        analysis_sources.append(conf_analysis)

                # 3. Process each pass as a separate data point
                for conf_analysis in analysis_sources:
                    if not conf_analysis: continue

                    for attr in ['brand', 'model', 'firmware_version']:
                        attr_data = conf_analysis.get(attr, {})
                        extracted_val = attr_data.get('extracted_value')

                        # Skip if nothing extracted (optional, depending on whether we want to train on Nulls)
                        if extracted_val is None:
                            continue

                        # 4. Extract Features
                        features = self.extract_features(conf_analysis, attr)

                        # 5. Determine Label
                        gt_val = gt_labels.get(attr)
                        is_correct = False

                        if gt_val:
                            is_match, _ = self.evaluator_helper.fuzzy_match(extracted_val, gt_val)
                            is_correct = True if is_match else False
                        else:
                            is_correct = False  # Hallucination (Extracted something but GT is null)

                        features['label'] = 1 if is_correct else 0
                        data_containers[attr].append(features)

            except Exception as e:
                # print(f"Warning: Error processing line: {e}")
                continue

        # Convert to DataFrames
        result_dfs = {}
        for attr, rows in data_containers.items():
            if not rows:
                print(f"Warning: No training data found for {attr}")
                continue
            df = pd.DataFrame(rows)
            result_dfs[attr] = df
            if debug:
                print(
                    f"[{attr}] Total: {len(df)} | Positive: {df['label'].sum()} | Negative: {len(df) - df['label'].sum()}")

        return result_dfs

    def train(self, training_data: Dict[str, pd.DataFrame]):
        """Trains a separate XGBoost model for each attribute."""
        print("\n" + "=" * 80)
        print("TRAINING DISCRIMINATORS (XGBoost)")
        print("=" * 80)

        for attr, df in training_data.items():
            print(f"\nTraining Model for Attribute: {attr.upper()}")

            if 'label' not in df.columns:
                continue

            y = df['label']
            X_raw = df.drop(columns=['label'])

            if not self.feature_names:
                self.feature_names = list(X_raw.columns)

            # Data cleaning: Replace NaN with 0, inf with max/min values
            X_cleaned = np.nan_to_num(
                X_raw.astype(np.float64),
                nan=0.0,
                posinf=np.finfo(np.float64).max,
                neginf=np.finfo(np.float64).min
            )

            # Normalize features
            X_scaled = self.scaler.fit_transform(X_cleaned)

            # Split training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )

            # --- Use XGBClassifier ---

            # Calculate scale_pos_weight (negative count / positive count) to handle imbalance
            num_pos = y_train.sum()
            num_neg = len(y_train) - num_pos
            scale_weight = num_neg / num_pos if num_pos > 0 else 1.0

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

            model.fit(X_train, y_train)

            train_acc = model.score(X_train, y_train)
            val_acc = model.score(X_val, y_val)
            print(f"  Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            # Get predicted probabilities
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            # Optimize threshold
            optimal_thresh = self._optimize_threshold(y_val, y_pred_proba)
            print(f"  Optimal Threshold: {optimal_thresh:.4f}")

            self.models[attr] = model
            self.thresholds[attr] = optimal_thresh

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importances = dict(zip(self.feature_names, model.feature_importances_))
                self.feature_importance[attr] = importances
                top5 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"  Top Features: {top5}")

    def _optimize_threshold(self, y_true, y_pred_proba, target_fpr=0.15):
        """Finds threshold that maximizes F1 while keeping FPR low."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.linspace(0.1, 0.95, 50):
            y_pred = (y_pred_proba >= thresh).astype(int)
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (y_true.sum()) if y_true.sum() > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            if fpr <= target_fpr and f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        return best_thresh

    def predict(self, confidence_analysis: Dict, attribute: str) -> Dict:
        """Predicts if a specific attribute extraction is reliable."""
        if attribute not in self.models:
            return {'is_reliable': True, 'score': 1.0, 'decision': 'NO_MODEL'}

        features = self.extract_features(confidence_analysis, attribute)
        feature_values = [features.get(name, 0) for name in self.feature_names]

        X_cleaned = np.nan_to_num(
            [feature_values],
            nan=0.0,
            posinf=np.finfo(np.float64).max,
            neginf=np.finfo(np.float64).min
        )
        X_scaled = self.scaler.transform(X_cleaned)

        model = self.models[attribute]
        prob = model.predict_proba(X_scaled)[0, 1]
        thresh = self.thresholds[attribute]
        is_reliable = prob >= thresh

        return {
            'is_reliable': bool(is_reliable),
            'confidence_score': float(prob),
            'threshold': float(thresh),
            'decision': 'ACCEPT' if is_reliable else 'REJECT'
        }

    def batch_filter(self, input_file: str, output_file: str):
        """Applies the trained discriminators to a file."""
        print(f"\nFiltering {input_file}...")
        stats = {k: {'total': 0, 'filtered': 0} for k in self.models.keys()}

        with open(input_file, 'r', encoding='utf-8') as fin, \
                open(output_file, 'w', encoding='utf-8') as fout:

            for line in fin:
                data = json.loads(line)

                # Check for confidence analysis in raw_outputs (priority) or root
                conf_analysis = {}
                # Logic: Use second pass analysis if available (more refined), else first pass, else root
                raw_outputs = data.get('raw_outputs', {})
                if 'second_pass' in raw_outputs:
                    conf_analysis = raw_outputs['second_pass'].get('parsed', {}).get('confidence_analysis', {})
                elif 'first_pass' in raw_outputs:
                    conf_analysis = raw_outputs['first_pass'].get('parsed', {}).get('confidence_analysis', {})
                else:
                    conf_analysis = data.get('confidence_analysis', {})

                discriminator_results = {}

                for attr in self.models.keys():
                    if attr in conf_analysis:
                        if conf_analysis[attr].get('extracted_value') is not None:
                            stats[attr]['total'] += 1
                            res = self.predict(conf_analysis, attr)
                            discriminator_results[attr] = res
                            if res['decision'] == 'REJECT':
                                # Note: This only marks it. The downstream evaluator needs to respect this flag.
                                # Or we can modify the extracted value here (e.g. set to None in a copy)
                                # For safety, we just record the decision.
                                stats[attr]['filtered'] += 1

                data['discriminator_results'] = discriminator_results
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')

        print("Filter Stats:")
        for attr, s in stats.items():
            rate = (s['filtered'] / s['total'] * 100) if s['total'] > 0 else 0
            print(f"  {attr}: Filtered {s['filtered']}/{s['total']} ({rate:.1f}%)")

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
        obj = cls()
        obj.models = data['models']
        obj.thresholds = data['thresholds']
        obj.scaler = data['scaler']
        obj.feature_names = data['feature_names']
        obj.feature_importance = data.get('feature_importance', {})
        return obj

    def plot_feature_importance(self, output_dir: str):
        """Generates and saves feature importance plots for each attribute."""
        print(f"\nGenerating Feature Importance Plots in {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)

        for attr, importance_dict in self.feature_importance.items():
            if not importance_dict:
                continue

            # Sort by importance
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            # Take top 15 for readability
            top_features = sorted_features[:15]

            names = [x[0] for x in top_features]
            values = [x[1] for x in top_features]

            plt.figure(figsize=(10, 6))
            plt.barh(names[::-1], values[::-1], color='skyblue')
            plt.xlabel('Importance Score')
            plt.title(f'Feature Importance: {attr.upper()}')
            plt.tight_layout()

            save_path = os.path.join(output_dir, f"feature_importance_{attr}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"  Saved: {save_path}")


# ==================== Main Execution ====================

def main():
    # ------------------------------------------------------------------
    # [Configuration Area]
    # ------------------------------------------------------------------

    # 1. Set working mode: True = Retrain model; False = Load existing model and run filtering
    TRAIN_MODE = True

    # 2. Set input file path (Must contain raw_outputs or confidence_analysis)
    INPUT_FILE = r"../agent/results/relabel_merged_cydar_data_multi_agent_results_sft_Qwen2___5_7B_Instruct_1130.jsonl"

    # 3. Set output directory
    OUTPUT_DIR = r"./discriminator_results_1130_xgboost"

    # 4. Model save/load name
    MODEL_NAME = "discriminator_model_1130_xgb.pkl"

    # 5. ID Exclusion List Path (Optional)
    ID_LIST_PATH = '../sft_sample/sft_dataset_CORRECT_only_1130_id.json'

    # ------------------------------------------------------------------
    # [Execution Logic]
    # ------------------------------------------------------------------

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Creating output directory: {OUTPUT_DIR}")

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Input file not found: {INPUT_FILE}")
        return

    # Load Exclusion List
    index_list = []
    try:
        if os.path.exists(ID_LIST_PATH):
            with open(ID_LIST_PATH, 'r') as f:
                index_list = json.load(f)
            print(f"Loaded exclusion list with {len(index_list)} IDs.")
    except Exception as e:
        print(f"Note: Failed to load exclusion list ({e}), using all data.")

    # Initialize Confidence Analyzer (Optional, used if analysis missing in file)
    try:
        analyzer = AttributeConfidenceAnalyzer()
    except Exception as e:
        print(f"Note: Failed to initialize AttributeConfidenceAnalyzer ({e}), relying only on data in file")
        analyzer = None

    model_path = os.path.join(OUTPUT_DIR, MODEL_NAME)
    discriminator = ConfidenceDiscriminator(model_type='xgboost')

    if TRAIN_MODE:
        print(">>> Entering training mode...")
        # 1. Prepare data (pass index_list)
        training_data = discriminator.prepare_training_data(INPUT_FILE, analyzer, index_list=index_list)

        if not training_data:
            print("❌ Unable to generate training data, please check input file format.")
            return

        # 2. Train
        discriminator.train(training_data)

        # 3. Save
        discriminator.save(model_path)

        # 4. Generate feature importance plots
        discriminator.plot_feature_importance(OUTPUT_DIR)

    else:
        print(">>> Entering inference/filtering mode...")
        if os.path.exists(model_path):
            discriminator = ConfidenceDiscriminator.load(model_path)
        else:
            print(f"❌ Model file not found: {model_path}")
            return

    # 5. Execute batch filtering (Optional, for verification only)
    # filtered_output_path = os.path.join(OUTPUT_DIR, "filtered_results_check.jsonl")
    # discriminator.batch_filter(INPUT_FILE, filtered_output_path)


if __name__ == "__main__":
    main()
