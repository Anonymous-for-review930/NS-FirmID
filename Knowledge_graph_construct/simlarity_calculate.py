import re
import difflib
from typing import List, Tuple, Set
from collections import Counter
import math


class ModelSimilarityCalculator:
    def __init__(self):
        # Common brand prefixes
        self.brand_prefixes = {
            'hp', 'hewlett-packard', 'canon', 'epson', 'brother', 'samsung',
            'dell', 'xerox', 'lexmark', 'ricoh', 'kyocera', 'sharp', 'toshiba',
            'cisco', 'huawei', 'h3c', 'juniper', 'arista', 'netgear', 'tp-link',
            'dlink', 'd-link', 'linksys', 'asus', 'zyxel'
        }

        # Common model suffixes and descriptors
        self.common_suffixes = {
            'series', 'plus', 'pro', 'lite', 'mini', 'max', 'ultra', 'premium',
            'standard', 'basic', 'advanced', 'enterprise', 'business', 'home',
            'office', 'connect', 'wireless', 'wired', 'ethernet', 'switch',
            'router', 'printer', 'scanner', 'multifunction', 'mfp', 'aio',
            'color', 'mono', 'monochrome', 'laser', 'inkjet', 'thermal'
        }

        # Common stop words and prepositions
        self.stop_words = {
            'and', 'or', 'with', 'for', 'the', 'a', 'an', 'of', 'in', 'on', 'at'
        }

    def normalize_model(self, model: str) -> str:
        """Normalize the model string"""
        # Convert to lowercase
        model = model.lower().strip()

        # Remove punctuation, keep alphanumeric characters and spaces
        model = re.sub(r'[^\w\s]', ' ', model)

        # Handle multiple spaces
        model = re.sub(r'\s+', ' ', model).strip()

        return model

    def extract_core_model(self, model: str) -> str:
        """Extract the core part of the model, removing brand prefixes and common suffixes"""
        normalized = self.normalize_model(model)
        words = normalized.split()

        # Remove brand prefixes
        filtered_words = []
        for word in words:
            if word not in self.brand_prefixes:
                filtered_words.append(word)

        # If empty after removal, keep original words
        if not filtered_words:
            filtered_words = words

        # Remove common suffixes (but keep core model identifiers)
        core_words = []
        for word in filtered_words:
            if word not in self.common_suffixes and word not in self.stop_words:
                core_words.append(word)

        # If too few words remain after removal, keep some important suffixes
        if len(core_words) < 2 and len(filtered_words) > len(core_words):
            # Keep some potentially important descriptors
            important_suffixes = {'series', 'plus', 'pro', 'switch', 'router', 'printer'}
            for word in filtered_words:
                if word in important_suffixes and word not in core_words:
                    core_words.append(word)

        return ' '.join(core_words) if core_words else normalized

    def extract_model_numbers(self, model: str) -> List[str]:
        """Extract numeric sequences from the model"""
        # Match consecutive alphanumeric combinations (model identifiers)
        patterns = [
            r'[a-z]*\d+[a-z]*',  # e.g.: hp1020, c4200, j9979a
            r'\d+[a-z]+\d*',  # e.g.: 1820g, 2900dn
            r'[a-z]+\d+[a-z]*\d*'  # e.g.: laserjet1020
        ]

        model_nums = set()
        normalized = self.normalize_model(model)

        for pattern in patterns:
            matches = re.findall(pattern, normalized)
            model_nums.update(matches)

        return list(model_nums)

    def calculate_numeric_similarity(self, model1: str, model2: str) -> float:
        """Calculate similarity of the model's numeric part"""
        nums1 = self.extract_model_numbers(model1)
        nums2 = self.extract_model_numbers(model2)

        if not nums1 or not nums2:
            return 0.0

        max_sim = 0.0
        for num1 in nums1:
            for num2 in nums2:
                # Exact match
                if num1 == num2:
                    max_sim = max(max_sim, 1.0)
                # One contains the other
                elif num1 in num2 or num2 in num1:
                    max_sim = max(max_sim, 0.8)
                else:
                    # String similarity
                    seq_sim = difflib.SequenceMatcher(None, num1, num2).ratio()
                    max_sim = max(max_sim, seq_sim * 0.6)

        return max_sim

    def calculate_token_similarity(self, model1: str, model2: str) -> float:
        """Calculate similarity based on token overlap"""
        core1 = self.extract_core_model(model1)
        core2 = self.extract_core_model(model2)

        tokens1 = set(core1.split())
        tokens2 = set(core2.split())

        if not tokens1 or not tokens2:
            return 0.0

        # Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))

        jaccard = intersection / union if union > 0 else 0.0

        # Consider sequence similarity (order of words)
        sequence_sim = difflib.SequenceMatcher(None, core1, core2).ratio()

        return (jaccard * 0.7 + sequence_sim * 0.3)

    def calculate_series_similarity(self, model1: str, model2: str) -> float:
        """Calculate series similarity (e.g., 1820 series)"""
        # Extract possible series identifiers
        series_pattern = r'(\d+)[a-z]*(?:\s|$)'

        series1 = re.findall(series_pattern, self.normalize_model(model1))
        series2 = re.findall(series_pattern, self.normalize_model(model2))

        if not series1 or not series2:
            return 0.0

        # Check if there are identical series numbers
        common_series = set(series1).intersection(set(series2))
        if common_series:
            return 0.9  # High similarity for same series

        # Check for close series (e.g., 1820 and 1850)
        for s1 in series1:
            for s2 in series2:
                try:
                    num1, num2 = int(s1), int(s2)
                    diff = abs(num1 - num2)
                    if diff <= 50:  # Series numbers are close
                        return 0.7
                except ValueError:
                    continue

        return 0.0

    def calculate_semantic_similarity(self, model1: str, model2: str) -> float:
        """Calculate semantic similarity between two models"""
        if not model1 or not model2:
            return 0.0

        # Exactly the same
        if model1.lower().strip() == model2.lower().strip():
            return 1.0

        # Calculate various similarities
        numeric_sim = self.calculate_numeric_similarity(model1, model2)
        token_sim = self.calculate_token_similarity(model1, model2)
        series_sim = self.calculate_series_similarity(model1, model2)

        # Basic string similarity
        basic_sim = difflib.SequenceMatcher(None,
                                            self.normalize_model(model1),
                                            self.normalize_model(model2)).ratio()

        # Check for containment relationship and give extra bonus
        norm1, norm2 = self.normalize_model(model1), self.normalize_model(model2)
        containment_bonus = 0.0

        if norm1 in norm2 or norm2 in norm1:
            containment_bonus = 0.2
        elif any(word in norm2 for word in norm1.split() if len(word) > 2):
            containment_bonus = 0.1

        # Comprehensive calculation (weights adjustable)
        final_score = (
                numeric_sim * 0.35 +  # Numeric similarity is most important
                token_sim * 0.25 +  # Token similarity
                series_sim * 0.20 +  # Series similarity
                basic_sim * 0.20 +  # Basic similarity
                containment_bonus  # Containment bonus
        )

        # Ensure result is between 0 and 1
        return min(1.0, max(0.0, final_score))


# Create global instance and convenience function
_calculator = ModelSimilarityCalculator()


def calculate_model_similarity(model1: str, model2: str) -> float:
    """Convenience function: Calculate semantic similarity between two models

    Args:
        model1: First model
        model2: Second model

    Returns:
        Similarity score (0.0-1.0)
    """
    return _calculator.calculate_semantic_similarity(model1, model2)


# Test cases
if __name__ == "__main__":
    test_cases = [
        # Different formats but same model
        ("HP LaserJet 1020", "laserjet1020"),
        ("Canon LBP-2900", "canon lbp2900 plus"),
        ("Brother HL-2140", "brother hl 2140 series"),

        # Same series but different models
        ("HP 1820-8G Switch", "hp officeconnect 1820-24g switch"),
        ("Canon LBP-2900", "canon lbp-3000"),

        # Similar but not exactly the same
        ("Epson Stylus C88", "epson stylus c88 plus"),
        ("Samsung ML-1640", "samsung ml 1640 series"),

        # Different series
        ("HP LaserJet 1020", "Canon LBP-2900"),
        ("Cisco WS-C2960", "Huawei S5700"),

        # Exactly the same
        ("HP LaserJet 1020", "HP LaserJet 1020"),

        # Containment relationship
        ("1820-8g switch", "hp officeconnect 1820-8g switch j9979a"),
    ]

    print("Model similarity test results:")
    print("=" * 80)

    for model1, model2 in test_cases:
        similarity = calculate_model_similarity(model1, model2)
        print(f"{model1:<30} vs {model2:<30} => {similarity:.3f}")

    print("\n" + "=" * 80)
    print("Core model extraction test:")
    test_models = [
        "HP OfficeConnect 1820-8G Switch J9979A",
        "Canon ImageCLASS LBP-2900 Laser Printer",
        "Brother HL-2140 Monochrome Laser Printer Series"
    ]

    for model in test_models:
        core = _calculator.extract_core_model(model)
        nums = _calculator.extract_model_numbers(model)
        print(f"Original: {model}")
        print(f"Core: {core}")
        print(f"Model Nums: {nums}")
        print("-" * 50)
