import re
import difflib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import unicodedata


@dataclass
class MatchResult:
    """Match result class"""
    model: str
    brand: str
    versions: List[str]
    confidence: float
    match_type: str


class NestedModelMatcher:
    """Intelligent model matcher - Supports nested knowledge base format"""

    def __init__(self, knowledge_base: Dict[str, Dict[str, List[str]]]):
        """
        Initialize the matcher

        Args:
            knowledge_base: Nested knowledge base, format is {Model: {Brand: [Version List]}}
        """
        self.knowledge_base = knowledge_base
        self.flat_index = self._build_flat_index()
        self.normalized_index = self._build_normalized_index()

    def _build_flat_index(self) -> List[Tuple[str, str, List[str]]]:
        """Build flat index"""
        flat_list = []
        for model, brands_dict in self.knowledge_base.items():
            for brand, versions in brands_dict.items():
                flat_list.append((model, brand, versions))
        return flat_list

    def _normalize_text(self, text: str) -> str:
        """Normalize text processing"""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower().strip()

        # Remove special characters, keep alphanumeric and common separators
        text = re.sub(r'[^\w\-\.\s]', '', text)

        # Unify separators
        text = re.sub(r'[\s\-\_\.]+', '-', text)

        # Remove leading and trailing separators
        text = text.strip('-')

        return text

    def _extract_core_parts(self, text: str) -> List[str]:
        """Extract core parts (alphanumeric combinations)"""
        # Extract consecutive alphanumeric combinations
        parts = re.findall(r'[a-zA-Z]*\d+[a-zA-Z]*|\d+[a-zA-Z]+|[a-zA-Z]+\d*', text)
        return [p.lower() for p in parts if len(p) > 1]

    def _remove_brand_prefix(self, text: str, brand: str) -> str:
        """Remove brand prefix"""
        norm_text = self._normalize_text(text)
        norm_brand = self._normalize_text(brand)

        if norm_text.startswith(norm_brand + '-'):
            return norm_text[len(norm_brand + '-'):]
        elif norm_text.startswith(norm_brand):
            return norm_text[len(norm_brand):]

        return norm_text

    def _build_normalized_index(self) -> Dict[str, List[Tuple[str, str, List[str]]]]:
        """Build normalized index"""
        normalized_index = {}

        for model, brand, versions in self.flat_index:
            # Generate multiple normalized variations
            variations = []

            # 1. Full model normalization
            norm_full = self._normalize_text(model)
            variations.append(norm_full)

            # 2. Model with brand prefix removed
            norm_no_brand = self._remove_brand_prefix(model, brand)
            if norm_no_brand != norm_full:
                variations.append(norm_no_brand)

            # 3. Core parts extraction
            core_parts = self._extract_core_parts(model)
            variations.extend(core_parts)

            # 4. Brand + Model combination
            brand_model = self._normalize_text(f"{brand} {model}")
            variations.append(brand_model)

            # 5. Original format (lowercase)
            original_lower = model.lower().strip()
            variations.append(original_lower)

            # Store all variations
            for variation in set(variations):  # Deduplicate
                if variation and len(variation) > 0:
                    if variation not in normalized_index:
                        normalized_index[variation] = []

                    # Avoid duplicate entries
                    entry = (model, brand, versions)
                    if entry not in normalized_index[variation]:
                        normalized_index[variation].append(entry)

        return normalized_index

    def _exact_match(self, query: str) -> List[MatchResult]:
        """Exact match"""
        results = []
        norm_query = self._normalize_text(query)

        if norm_query in self.normalized_index:
            for model, brand, versions in self.normalized_index[norm_query]:
                results.append(MatchResult(
                    model=model,
                    brand=brand,
                    versions=versions,
                    confidence=1.0,
                    match_type="exact"
                ))

        return results

    def _fuzzy_match(self, query: str, threshold: float = 0.6) -> List[MatchResult]:
        """Fuzzy match"""
        results = []
        norm_query = self._normalize_text(query)

        for indexed_key, entries in self.normalized_index.items():
            # Calculate similarity using difflib
            similarity = difflib.SequenceMatcher(None, norm_query, indexed_key).ratio()

            if similarity >= threshold:
                for model, brand, versions in entries:
                    results.append(MatchResult(
                        model=model,
                        brand=brand,
                        versions=versions,
                        confidence=similarity,
                        match_type="fuzzy"
                    ))

        return results

    def _partial_match(self, query: str) -> List[MatchResult]:
        """Partial match"""
        results = []
        norm_query = self._normalize_text(query)
        query_parts = self._extract_core_parts(query)

        for indexed_key, entries in self.normalized_index.items():
            indexed_parts = self._extract_core_parts(indexed_key)

            # Calculate overlap of core parts
            if query_parts and indexed_parts:
                common_parts = set(query_parts) & set(indexed_parts)
                if common_parts:
                    confidence = len(common_parts) / max(len(query_parts), len(indexed_parts))
                    if confidence > 0.5:
                        for model, brand, versions in entries:
                            results.append(MatchResult(
                                model=model,
                                brand=brand,
                                versions=versions,
                                confidence=confidence * 0.8,  # Lower confidence for partial matches
                                match_type="partial"
                            ))

        return results

    def _substring_match(self, query: str) -> List[MatchResult]:
        """Substring match"""
        results = []
        norm_query = self._normalize_text(query)

        for indexed_key, entries in self.normalized_index.items():
            # Bidirectional substring check
            if len(norm_query) >= 3 and len(indexed_key) >= 3:
                if norm_query in indexed_key or indexed_key in norm_query:
                    confidence = min(len(norm_query), len(indexed_key)) / max(len(norm_query), len(indexed_key))
                    if confidence > 0.6:
                        for model, brand, versions in entries:
                            results.append(MatchResult(
                                model=model,
                                brand=brand,
                                versions=versions,
                                confidence=confidence * 0.7,  # Lower confidence for substring matches
                                match_type="substring"
                            ))

        return results

    def _brand_aware_match(self, query: str, hint_brand: str = None) -> List[MatchResult]:
        """Brand-aware match"""
        if not hint_brand:
            return []

        results = []
        norm_query = self._normalize_text(query)
        norm_brand = self._normalize_text(hint_brand)

        # Find all models under this brand
        brand_models = []
        for model, brand, versions in self.flat_index:
            if self._normalize_text(brand) == norm_brand:
                brand_models.append((model, brand, versions))

        # Match within the brand scope
        for model, brand, versions in brand_models:
            norm_model = self._normalize_text(model)
            norm_no_brand = self._remove_brand_prefix(model, brand)

            # Match model directly (remove brand prefix)
            if norm_query == norm_no_brand or norm_query in norm_no_brand or norm_no_brand in norm_query:
                confidence = 0.9 if norm_query == norm_no_brand else 0.8
                results.append(MatchResult(
                    model=model,
                    brand=brand,
                    versions=versions,
                    confidence=confidence,
                    match_type="brand_aware"
                ))

        return results

    def search(self, query: str, hint_brand: str = None, max_results: int = 5) -> List[MatchResult]:
        """
        Search for matching models

        Args:
            query: Model to query
            hint_brand: Brand hint (if known)
            max_results: Maximum number of results to return

        Returns:
            List of match results, sorted by confidence in descending order
        """
        all_results = []

        # 1. If there is a brand hint, prioritize brand-aware matching
        if hint_brand:
            brand_results = self._brand_aware_match(query, hint_brand)
            all_results.extend(brand_results)

            # If brand-aware matching yields high confidence results, return them first
            high_confidence_brand = [r for r in brand_results if r.confidence > 0.85]
            if high_confidence_brand:
                return sorted(high_confidence_brand, key=lambda x: x.confidence, reverse=True)[:max_results]

        # 2. Exact match
        exact_results = self._exact_match(query)
        all_results.extend(exact_results)

        # If there is an exact match, return it first
        if exact_results:
            return sorted(exact_results, key=lambda x: x.confidence, reverse=True)[:max_results]

        # 3. Fuzzy match
        fuzzy_results = self._fuzzy_match(query, threshold=0.7)
        all_results.extend(fuzzy_results)

        # 4. Partial match
        partial_results = self._partial_match(query)
        all_results.extend(partial_results)

        # 5. Substring match
        substring_results = self._substring_match(query)
        all_results.extend(substring_results)

        # Deduplicate and sort
        seen = set()
        unique_results = []
        for result in all_results:
            key = (result.model, result.brand)
            if key not in seen:
                seen.add(key)
                unique_results.append(result)

        # Sort by confidence in descending order
        unique_results.sort(key=lambda x: x.confidence, reverse=True)

        return unique_results[:max_results]

    def get_all_brands(self) -> List[str]:
        """Get list of all brands"""
        brands = set()
        for model, brand, versions in self.flat_index:
            brands.add(brand)
        return sorted(list(brands))

    def get_models_by_brand(self, brand: str) -> List[Tuple[str, List[str]]]:
        """Get all models by brand"""
        norm_brand = self._normalize_text(brand)
        models = []
        for model, model_brand, versions in self.flat_index:
            if self._normalize_text(model_brand) == norm_brand:
                models.append((model, versions))
        return models

    def batch_search(self, queries: List[str], hint_brands: List[str] = None) -> Dict[str, List[MatchResult]]:
        """Batch search"""
        results = {}
        hint_brands = hint_brands or [None] * len(queries)

        for i, query in enumerate(queries):
            brand_hint = hint_brands[i] if i < len(hint_brands) else None
            results[query] = self.search(query, hint_brand=brand_hint)

        return results


# Example usage
if __name__ == "__main__":
    # Example nested knowledge base: {Model: {Brand: [Version List]}}
    knowledge_base = {
        "iPhone-14-Pro": {
            "Apple": ["iOS-16.0", "iOS-16.1", "iOS-16.2"]
        },
        "iPhone-14-Pro-Max": {
            "Apple": ["iOS-16.0", "iOS-16.1"]
        },
        "Galaxy-S23": {
            "Samsung": ["Android-13", "One-UI-5.1"]
        },
        "Galaxy-S23-Ultra": {
            "Samsung": ["Android-13", "One-UI-5.1"]
        },
        "MacBook-Pro-M2": {
            "Apple": ["macOS-13.0", "macOS-13.1"]
        },
        "MacBook-Air-M2": {
            "Apple": ["macOS-13.0"]
        },
        "ThinkPad-X1-Carbon": {
            "Lenovo": ["Windows-11", "Ubuntu-22.04"]
        },
        "Surface-Pro-9": {
            "Microsoft": ["Windows-11"]
        },
        "iPad-Pro-11": {
            "Apple": ["iPadOS-16.0", "iPadOS-16.1"]
        },
        "Pixel-7-Pro": {
            "Google": ["Android-13", "Android-14"]
        },
        "OnePlus-10-Pro": {
            "OnePlus": ["Android-12", "OxygenOS-12"]
        }
    }

    # Create matcher
    matcher = NestedModelMatcher(knowledge_base)

    # Test queries
    test_cases = [
        ("iphone14pro", None),  # No separators, lowercase
        ("iPhone 14Pro", "Apple"),  # With brand hint
        ("Galaxy S23", "Samsung"),  # Brand hint
        ("MacBookPro M2", None),  # Different separators
        ("X1 Carbon", "Lenovo"),  # Partial match + brand hint
        ("Surface Pro9", None),  # Fewer separators
        ("ipadpro11", "Apple"),  # All lowercase + brand hint
        ("Pixel7Pro", None),  # Combined variations
        ("ThinkPad X1", None),  # Partial match
        ("14 Pro Max", "Apple")  # Suffix only + brand hint
    ]

    print("=== Nested Knowledge Base Intelligent Model Matching Test ===\n")

    for query, brand_hint in test_cases:
        print(f"Query: '{query}'" + (f" (Brand Hint: {brand_hint})" if brand_hint else ""))
        results = matcher.search(query, hint_brand=brand_hint, max_results=3)

        if results:
            for i, result in enumerate(results, 1):
                print(f"  {i}. Model: {result.model}")
                print(f"     Brand: {result.brand}")
                print(f"     Versions: {', '.join(result.versions)}")
                print(f"     Confidence: {result.confidence:.3f}")
                print(f"     Match Type: {result.match_type}")
                print()
        else:
            print("  No matching results found\n")

        print("-" * 60)

    # Show brand information
    print(f"\nAll brands in knowledge base: {matcher.get_all_brands()}")
    print(f"\nAll models for brand Apple: {matcher.get_models_by_brand('Apple')}")
