import json
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
from Knowledge_graph_construct.extract_info_from_database import *
from tools.tool_global import clean_keyword, sort_by_confidence_desc

@dataclass
class ValidationResult:
    """Validation result class"""
    brand: str
    model: str
    version: str
    confidence: float
    match_details: Dict[str, str]  # Records matching details for each field


@dataclass
class MatchInfo:
    """Match info class"""
    original: str
    matched: str
    confidence: float
    match_type: str

@dataclass
class MatchInfoModel:
    """Match info class -- Model"""
    original: str
    matched: str
    confidence: float
    match_type: str
    brand: str
    version_list: str

@dataclass
class MatchInfoVersion:
    """Match info class -- Version"""
    original: str
    matched: str
    confidence: float
    match_type: str
    brand: str
    model: str


class TripletValidator:
    """Smart Brand-Model-Version Triplet Validator"""

    def __init__(self, database_brand: Dict[str, Dict[str, List[str]]],
                 database_model: Dict[str, Dict[str, List[str]]]):
        """
        Initialize validator

        Args:
            database_brand: Knowledge base in {brand: {model: [versions]}} format
            database_model: Knowledge base in {model: {brand: [versions]}} format
        """
        self.database_brand = database_brand
        self.database_model = database_model
        self.matcher = self._create_model_matcher()

        # Pre-process knowledge base index
        self.brand_index = self._build_brand_index()
        self.model_index = self._build_model_index()
        self.version_index = self._build_version_index()
        self.confidence_weight = [0.2, 0.3, 0.5]  # Weight distribution

    def _create_model_matcher(self):
        """Create model matcher"""
        try:
            from Knowledge_graph_construct.extract_info_from_database import NestedModelMatcher  # Replace with actual import
            return NestedModelMatcher(self.database_model)
        except ImportError:
            # If import fails, use simplified version
            print("no matcher module")
            # return SimpleMatcher(self.database_model)
            return None

    def _normalize_text(self, text: str) -> str:
        """Normalize text"""
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(r'[^\w\-\.\s]', '', text)
        text = re.sub(r'[\s\-\_\.]+', '-', text)
        text = clean_keyword(text)
        return text.strip()

    def _build_brand_index(self) -> Dict[str, str]:
        """Build brand normalization index"""
        index = {}
        for brand in self.database_brand.keys():
            norm_brand = self._normalize_text(brand)
            index[norm_brand] = brand
            # Add common variations
            if ' ' in brand:
                index[brand.replace(' ', '').lower()] = brand
        return index

    def _build_model_index(self) -> Dict[str, str]:
        """Build model normalization index"""
        index = {}
        for model in self.database_model.keys():
            norm_model = self._normalize_text(model)
            index[norm_model] = model
        return index

    def _build_version_index(self) -> Dict[str, Set[str]]:
        """Build version index - Record all possible original formats corresponding to each version"""
        index = defaultdict(set)
        for model_data in self.database_model.values():
            for brand_data in model_data.values():
                for version in brand_data:
                    norm_version = self._normalize_text(version)
                    index[norm_version].add(version)
        return index

    def _fuzzy_match_brand(self, candidate: str, threshold: float = 0.8) -> list[MatchInfo]:
        """Fuzzy match brand"""
        norm_candidate = self._normalize_text(candidate)
        matches = []
        # First attempt exact match
        if norm_candidate in self.brand_index:
            matches.append(MatchInfo(
                original=candidate,
                matched=self.brand_index[norm_candidate],
                confidence=1.0,
                match_type="exact"
            ))
            return matches

        # # Fuzzy match
        # best_match = []
        # best_score = 0
        #
        # for norm_brand, original_brand in self.brand_index.items():
        #     score = difflib.SequenceMatcher(None, norm_candidate, norm_brand).ratio()
        #     if score > best_score and score >= threshold:
        #         best_score = score
        #         best_match.append(MatchInfo(
        #             original=candidate,
        #             matched=original_brand,
        #             confidence=score,
        #             match_type="fuzzy"
        #         ))
        return matches
        # return best_match

    def _extract_model_components(self, model: str) -> Dict[str, str]:
        """Extract key components of the model"""
        model_lower = model.lower()
        components = {}

        # Extract brand
        brands = ['iphone', 'galaxy', 'pixel', 'macbook', 'ipad', 'surface', 'thinkpad']
        for brand in brands:
            if brand in model_lower:
                components['product_line'] = brand
                break

        # Extract numbers (generation)
        numbers = re.findall(r'\d+', model)
        if numbers:
            components['generation'] = numbers[0]

        # Extract spec identifiers
        specs = ['pro', 'max', 'mini', 'plus', 'ultra', 'air']
        found_specs = []
        for spec in specs:
            if spec in model_lower:
                found_specs.append(spec)
        components['specs'] = found_specs

        return components

    def _calculate_semantic_similarity(self, candidate: str, target: str) -> float:

        """Calculate semantic similarity"""
        candidate_comp = self._extract_model_components(candidate)
        target_comp = self._extract_model_components(target)

        score = 0.0
        total_weight = 0.0

        # Product line match (weight: 0.4)
        if candidate_comp.get('product_line') == target_comp.get('product_line'):
            score += 0.4
        total_weight += 0.4

        # Generation similarity (weight: 0.3)
        cand_gen = candidate_comp.get('generation')
        targ_gen = target_comp.get('generation')
        if cand_gen and targ_gen:
            try:
                gen_diff = abs(int(cand_gen) - int(targ_gen))
                # The smaller the generation difference, the higher the similarity
                gen_similarity = max(0.0, 1 - gen_diff * 0.2)  # Deduct 0.2 points for each generation difference
                score += gen_similarity * 0.3
            except ValueError:
                pass
        total_weight += 0.3

        # Spec match (weight: 0.3)
        cand_specs = set(candidate_comp.get('specs', []))
        targ_specs = set(target_comp.get('specs', []))
        if cand_specs or targ_specs:
            spec_similarity = len(cand_specs & targ_specs) / max(len(cand_specs | targ_specs), 1)
            score += spec_similarity * 0.3
        total_weight += 0.3

        return score / total_weight if total_weight > 0 else 0.0


    def _calculate_model_similarity_all(self, candidate: str, target: str) -> float:
        ###@@@@@@@@@@###
        from Knowledge_graph_construct.simlarity_calculate import calculate_model_similarity
        ###@@@@@@@@@@###
        return calculate_model_similarity(candidate, target)

    def _fuzzy_match_model(self, candidate: str, hint_brand: str = "", threshold: float = 0.7) -> list[MatchInfoModel]:
        """Enhanced fuzzy match model"""
        # 1. Use existing matcher
        match_list = []
        # try:
        #     results = self.matcher.search(candidate, hint_brand=hint_brand, max_results=3)
        #     if results:
        #         for item in results:
        #             if item.confidence >= threshold:
        #                 match_list.append(MatchInfoModel(
        #                                             original=candidate,
        #                                             matched=results[0].model,
        #                                             confidence=results[0].confidence,
        #                                             match_type=results[0].match_type,
        #                                             brand=hint_brand
        #                                         ))
        #         return match_list
        # except:

        # 2. Exact match
        norm_candidate = self._normalize_text(candidate)
        # if hint_brand and hint_brand in self.database_brand.keys():
        #     model_dict = self.database_brand[hint_brand]
        #
        # if norm_candidate in self.model_index:
        #     matched_model = self.model_index[norm_candidate]
        #     match_list.append(MatchInfoModel(
        #         original=candidate,
        #         matched=matched_model,
        #         confidence=1.0,
        #         match_type="exact",
        #         brand=json.dumps(list(self.database_model[matched_model].keys())),
        #         version_list=
        #     ))
        #     return match_list

        # 3. Semantic similarity match (new)
        best_semantic_score = 0.0

        # If there is a brand hint, prioritize search within that brand
        search_models = []
        if hint_brand and hint_brand in self.database_brand:
            search_models = list(self.database_brand[hint_brand].keys())
        else:
            search_models = list(self.database_model.keys())

        for model in search_models:
            model_ = self._normalize_text(model)
            norm_candidate_ = clean_keyword(norm_candidate)
            if len(model_) < 3:
                continue
            # if model not in ['laserjet pro mfp 4101', 'm4100-d12g']:
            #     continue
            # if '1820' not in model:
            #     continue
            # print(model)
            # if "3845" not in model:
            #     continue
            try:
                # First calculate traditional string similarity
                string_sim = difflib.SequenceMatcher(None, norm_candidate_, model_).ratio()

                # Then calculate semantic similarity
                semantic_sim = self._calculate_model_similarity_all(norm_candidate_, model_)
                # Add bonus score
                if model_.startswith(norm_candidate_) or norm_candidate_.startswith(model_) or model_.endswith(
                        norm_candidate_) or norm_candidate_.endswith(model_):
                    string_sim += 0.3
                elif model_ in norm_candidate_ or norm_candidate_ in model_:
                    string_sim += 0.2
                # if string_sim > 1:
                #     string_sim = 1
                combined_score = string_sim * 0.2 + semantic_sim * 0.8
            except:
                print("Error calculating similarity!")
                print(model)
                continue
            # Combined similarity (String similarity 60%, Semantic similarity 40%)
            else:
                try:
                    if combined_score >= threshold:  # Lower threshold combined_score > best_semantic_score and
                        best_semantic_score = combined_score
                        match_type = "semantic" if semantic_sim > string_sim else "fuzzy"
                        if hint_brand in self.database_model[model].keys():
                            match_item = MatchInfoModel(
                            original=candidate,
                            matched=model,
                            confidence=combined_score,
                            match_type=match_type,
                            brand=hint_brand,
                            version_list=json.dumps(self.database_model[model][hint_brand])
                        )
                        else:
                            if len(self.database_model[model].keys()) == 1:

                                match_item = MatchInfoModel(
                                    original=candidate,
                                    matched=model,
                                    confidence=combined_score,
                                    match_type=match_type,
                                    brand=list(self.database_model[model].keys())[0],
                                    version_list=json.dumps(list(self.database_model[model].values())[0])
                                )
                            else:
                                match_item = MatchInfoModel(
                                    original=candidate,
                                    matched=model,
                                    confidence=combined_score,
                                    match_type=match_type,
                                    brand=json.dumps(list(self.database_model[model].keys())),
                                    version_list=None
                                )
                        if match_item not in match_list:
                            match_list.append(match_item)

                    else:
                        continue
                except Exception as e:
                    print("Error finding similar models:", e)
                    continue

        return match_list

    def _fuzzy_match_version(self, candidate: str, brand: str = "", model: str = "", threshold: float = 0.8) -> list[MatchInfoVersion]:
        """Fuzzy match version"""
        norm_candidate = self._normalize_text(candidate)
        version_match_result = []
        # Exact match
        if self.database_model[model][brand] == []:

            if norm_candidate in self.version_index:
                prob_model = []
                prob_brand = []
                # Select most similar original version
                # best_original = min(self.version_index[norm_candidate],
                #                     key=lambda x: abs(len(x) - len(candidate)))
                best_matched = self.version_index[norm_candidate]
                # Brand and model corresponding to index
                for item in best_matched:
                    if brand in self.database_brand.keys():
                        for model_item, version_list in self.database_brand[brand].items():
                            if item in version_list:
                                version_match_result.append(MatchInfoVersion(
                                                                original=candidate,
                                                                matched=item,
                                                                confidence=1.0,
                                                                match_type="exact",
                                                                brand=brand,
                                                                model=model_item
                                                            ))

                    else:
                        for b, m_v in self.database_brand.items():
                            for m, vv in m_v.items():
                                if item in vv:
                                    version_match_result.append(MatchInfoVersion(
                                        original=candidate,
                                        matched=item,
                                        confidence=1.0,
                                        match_type="exact",
                                        brand=b,
                                        model=m
                                    ))


            else:
                return []
            return version_match_result
        else:
            version_list = self.database_model[model][brand]
            # Fuzzy match
            best_match = []
            best_score = 0
            match_list = []
            for version in version_list:
                candidate_ver = self._normalize_text(version)
                score = difflib.SequenceMatcher(None, norm_candidate, candidate_ver).ratio()
            # for norm_version, original_versions in self.version_index.items():
            #     score = difflib.SequenceMatcher(None, norm_candidate, norm_version).ratio()
            #     if score > best_score and score >= threshold:
            #         best_score = score
            #         # best_original = min(original_versions,
            #         #                     key=lambda x: abs(len(x) - len(candidate)))
            #         best_original = version
            #         best_match = MatchInfo(
            #             original=candidate,
            #             matched=best_original,
            #             confidence=score,
            #             match_type="fuzzy"
            #         )
                if score >= threshold:
                    if version.lower() in match_list:
                        continue
                    match_list.append(version.lower())
                    best_match.append(MatchInfoVersion(
                        original=candidate,
                        matched=version,
                        confidence=score,
                        match_type="fuzzy",
                        brand=brand,
                        model=model
                    ))

            return best_match


    def validate_triplets(self, candidate_brands: List[str],
                          candidate_models: List[str],
                          candidate_versions: List[str],
                          confidence_threshold: float = 0.7, max_model_results: int = 5) -> (List[ValidationResult], Dict[str, Dict[str, List[str]]], Dict[str, List[MatchInfoVersion]]):
        """
        Validate brand-model-version triplets

        Args:
            candidate_brands: List of candidate brands
            candidate_models: List of candidate models
            candidate_versions: List of candidate versions
            confidence_threshold: Minimum confidence threshold

        Returns:
            List of validated triplets, sorted by confidence in descending order
        """
        valid_results = []
        processed_combinations = set()  # Avoid duplicate processing
        version_match_dict = defaultdict(list)
        # Phase 1: Brand-based validation
        brand_match_all = []
        model_match_dict = defaultdict(list)
        for brand_candidate in candidate_brands:   # Temporarily select only the one brand with highest similarity
            # model_match_dict[brand_candidate] = []
            brand_match = self._fuzzy_match_brand(brand_candidate)
            brand_match_all+=brand_match
            if not brand_match or brand_match[0].confidence < confidence_threshold:
                # Brand might be incorrect
                continue

            matched_brand = brand_match[0].matched
            if matched_brand not in self.database_brand:
                continue
            if matched_brand not in model_match_dict:
                model_match_dict[matched_brand] = []
            else:
                continue
            # Get all models under this brand
            available_models = list(self.database_brand[matched_brand].keys())
            model_match_all = []
            # version_match_dict = defaultdict(list)
            for model_candidate in candidate_models:
                # Direct match
                if model_candidate not in version_match_dict:
                    version_match_dict[model_candidate] = []
                model_match = []
                if model_candidate in available_models and self.database_brand[matched_brand][model_candidate] != []:
                    model_tmp = MatchInfoModel(
                        original=model_candidate,
                        matched=model_candidate,
                        confidence=1.0,
                        match_type="exact",
                        brand=matched_brand,
                        version_list=json.dumps(self.database_brand[matched_brand][model_candidate])
                    )
                    if model_tmp not in model_match:
                        model_match.append(model_tmp)
                    if self.database_brand[matched_brand][model_candidate] == []:
                        model_match_ = self._fuzzy_match_model(model_candidate, hint_brand=matched_brand)
                        if model_match_ == []:
                            continue
                        for itemm in model_match_:
                            if itemm not in model_match:
                                model_match.append(itemm)
                else:
                    # Fuzzy match (within that brand scope)
                    model_match_ = self._fuzzy_match_model(model_candidate, hint_brand=matched_brand)
                    if model_match_ == []:
                        continue
                    for itemm in model_match_:
                        if itemm not in model_match:
                            model_match.append(itemm)

                # for model_tmp in model_match:
                #     if model_tmp not in model_match_all:
                #         model_match_all.append(model_tmp)
                # todo: Sort and filter for highest model confidence
                model_match = sort_by_confidence_desc(model_match)
                # todo: Keep top 5 that have version_list
                count = 0
                for i, model_match_item in enumerate(model_match):
                    version_list = model_match_item.version_list
                    if version_list == None:
                        continue  # todo: None indicates no corresponding definite brand for this model, so version was not extracted
                    try:
                        version_list_json = json.loads(version_list)
                        if version_list_json == []:
                            continue
                        else:
                            count += 1
                    except json.decoder.JSONDecodeError:
                        continue
                    if count == max_model_results:
                        break
                if count == 0:
                    count = min(max_model_results, len(model_match))
                model_match = model_match[:count]
                max_model_confidence = model_match[0].confidence
                for model_tmp in model_match:
                    if model_tmp not in model_match_all:
                        model_match_all.append(model_tmp)
                version_match = []
                model_version_list = []
                for version_candidate in candidate_versions:
                    # Direct match version
                    # match_flag = False
                    for model in model_match:
                        model_matched = model.matched
                        try:
                            model_version_list = json.loads(model.version_list)
                            if version_candidate in model_version_list:
                                version_tmp = MatchInfoVersion(
                                    original=version_candidate,
                                    matched=version_candidate,
                                    confidence=1.0,
                                    match_type="exact",
                                    brand=matched_brand,
                                    model=model_matched
                                )
                                if version_tmp not in version_match:
                                    version_match.append(version_tmp)
                            else:
                                # Fuzzy match version
                                version_tmp = self._fuzzy_match_version(version_candidate, brand=matched_brand,
                                                                        model=model_matched)
                                if version_tmp == []:
                                    continue
                                for tmpp in version_tmp:
                                    if tmpp not in version_match:
                                        version_match.append(tmpp)
                        except json.decoder.JSONDecodeError:  # This situation occurs because the version is matched based on the model, and the corresponding brand-versions dictionary was not saved ---- Incorrect brand and library happens to have incorrect brand might enter this code
                            if model.match_type == "fuzzy":
                                continue
                            else:
                                for brand, item in self.database_model[model_matched].items():
                                    norm_version_candidate = self._normalize_text(version_candidate)
                                    if norm_version_candidate in self.version_index.keys():
                                        version_kb = self.version_index[norm_version_candidate]
                                        for version_kb_item in version_kb:
                                            if version_kb_item in item:
                                                version_tmp = MatchInfoVersion(
                                                    original=version_candidate,
                                                    matched=version_candidate,
                                                    confidence=1,
                                                    match_type="exact",
                                                    brand=brand,
                                                    model=model_matched
                                                )
                                                if version_tmp not in version_match:
                                                    version_match.append(version_tmp)

                            # match_flag = True
                    # if not match_flag:


                ## Organize matched combinations -- Single candidate model under candidate brand loop
                if not version_match:  # Indicates version might be wrong
                    # continue
                    if not model_match: # Model also not matched
                        overall_confidence = self.confidence_weight[0]*1.0
                        version_item = ValidationResult(
                            brand=matched_brand,
                            model="",
                            version="",
                            confidence=overall_confidence,
                            match_details={
                                'brand': f"exact(1.0)",
                                'model': f"exact(0)",
                                'version': f"exact(0)"
                            }
                        )
                        if version_item not in valid_results:
                            valid_results.append(version_item)
                    else:
                        overall_confidence = self.confidence_weight[0] * 1.0 + self.confidence_weight[1] * max_model_confidence
                        version_item = ValidationResult(
                            brand=matched_brand,
                            model=model_candidate,
                            version="",
                            confidence=overall_confidence,
                            match_details={
                                'brand': f"exact(1.0)",
                                'model': f"fuzz({max_model_confidence})",
                                'version': f"exact(0)"
                            }
                        )
                        if version_item not in valid_results:
                            valid_results.append(version_item)
                else:

                    version_confidence = 0.0
                    index = 0
                    index_list = []
                    version_flag = False
                    version_match_filtered = []
                    matched_model_list = []
                    # Get matched models
                    for model_itemm in model_match:
                        model_ = model_itemm.matched
                        if model_ not in matched_model_list:
                            matched_model_list.append(model_)
                    # Filter versions that match well with candidate model similarity

                    for k, item in enumerate(version_match):  # Take highest confidence
                        confidence_threshold = item.confidence
                        if confidence_threshold > version_confidence:
                            version_confidence = confidence_threshold
                            index = k
                            version_flag = True
                    if not version_flag:
                        continue
                    for k, item in enumerate(version_match):
                        if item.confidence == version_confidence:  # Same confidence
                            index_list.append(k)
                    # Use highest similarity with library version as confidence
                    model_confidence = 0.0
                    brand_confidence = 0.0
                    triplet = ()
                    for index in index_list:
                        ## Case 1: Corresponding brand/model found directly in library and both are in candidate list, indicating this combination is correct
                        if version_match[index].model in candidate_models:  # If model exists in library, confidence is highest
                            model_confidence = 1.0
                            if version_match[index].brand in candidate_brands:  # If brand exists in library, confidence is highest
                                brand_confidence = 1.0
                                # overall_confidence = self.confidence_weight[0] * brand_confidence + self.confidence_weight[
                                #     1] * model_confidence + self.confidence_weight[2] * version_confidence
                                # valid_results.append([(version_match[index].brand, version_match[index].model,
                                #                        version_match[index].original)])
                                triplet = (version_match[index].brand, version_match[index].model, version_match[index].original)
                                overall_confidence = self.confidence_weight[0] * brand_confidence + self.confidence_weight[
                                    1] * model_confidence + self.confidence_weight[2] * version_confidence
                                if triplet not in processed_combinations:
                                    processed_combinations.add(triplet)
                                    valid_results.append(ValidationResult(
                                        brand=version_match[index].brand,
                                        model=version_match[index].model,
                                        version=version_match[index].original,
                                        confidence=overall_confidence,
                                        match_details={
                                            'brand': f"exact({brand_confidence})",
                                            'model': f"exact({model_confidence})",
                                            'version': f"{version_match[index].match_type}({version_confidence})"
                                        }
                                    ))
                            else:  # Case 2: No direct brand found but direct model found
                                brand_confidence = 0.0
                                triplet = (
                                "", version_match[index].model, version_match[index].original)
                                overall_confidence = self.confidence_weight[0] * brand_confidence + self.confidence_weight[
                                    1] * model_confidence + self.confidence_weight[2] * version_confidence
                                if triplet not in processed_combinations:
                                    processed_combinations.add(triplet)
                                    valid_results.append(ValidationResult(
                                        brand=triplet[0],
                                        model=triplet[1],
                                        version=triplet[2],
                                        confidence=overall_confidence,
                                        match_details={
                                            'brand': f"implicit({brand_confidence})({version_match[index].brand})",
                                            'model': f"exact({model_confidence})",
                                            'version': f"{version_match[index].match_type}({version_confidence})"
                                        }
                                    ))
                        else:  # Case 3: No direct model found but direct brand found
                            if version_match[index].brand in candidate_brands:
                                brand_confidence = 1.0
                                if model_match:
                                    for item in model_match:
                                        if item.confidence > model_confidence:
                                            model_confidence = item.confidence
                                    triplet = (
                                    version_match[index].brand, model_candidate, version_match[index].original)
                                else:
                                    model_confidence=0.0
                                    triplet = (version_match[index].brand, "", version_match[index].original)
                                overall_confidence = self.confidence_weight[0] * brand_confidence + self.confidence_weight[
                                    1] * model_confidence + self.confidence_weight[2] * version_confidence
                                if triplet not in processed_combinations:
                                    processed_combinations.add(triplet)
                                    valid_results.append(ValidationResult(
                                        brand=triplet[0],
                                        model=triplet[1],
                                        version=triplet[2],
                                        confidence=overall_confidence,
                                        match_details={
                                            'brand': f"exact({brand_confidence})",
                                            'model': f"implicit({model_confidence})({version_match[index].model})",
                                            'version': f"{version_match[index].match_type}({version_confidence})"
                                        }
                                    ))
                            else:  # Case 4: Neither direct model nor brand found

                                for model_item in model_match:
                                    if version_match[index].model == model_item.matched:
                                        model_confidence = model_item.confidence
                                        if model_item.brand in candidate_brands:
                                            brand_confidence = 1.0
                                            # valid_results.append([(model_item.brand, model_item.original,
                                            #                        version_match[index].original), brand_confidence*self.confidence_weight[0] + model_confidence*self.confidence_weight[1] + version_confidence*self.confidence_weight[2]])
                                            triplet = (model_item.brand, version_match[index].model, version_match[index].original)
                                            overall_confidence = self.confidence_weight[0] * brand_confidence + \
                                                                 self.confidence_weight[
                                                                     1] * model_confidence + self.confidence_weight[
                                                                     2] * version_confidence
                                            if triplet not in processed_combinations:
                                                processed_combinations.add(triplet)
                                                valid_results.append(ValidationResult(
                                                    brand=triplet[0],
                                                    model=triplet[1],
                                                    version=triplet[2],
                                                    confidence=overall_confidence,
                                                    match_details={
                                                        'brand': f"implicit({brand_confidence})",
                                                        'model': f"fuzzy({model_confidence}({model_item.matched}))",
                                                        'version': f"{version_match[index].match_type}({version_confidence})"
                                                    }
                                                ))
                                        else:
                                            brand_confidence = 0.0
                                            for brand_item in brand_match:
                                                if model_item.brand == brand_item.matched:
                                                    brand_confidence = brand_item.confidence
                                            # valid_results.append([(model_item.brand, model_item.original, version_match[index].original), brand_confidence*self.confidence_weight[0] + model_confidence*self.confidence_weight[1] + version_confidence*self.confidence_weight[2]])
                                                    triplet = (model_item.brand, model_item.original, version_match[index].original)
                                                else:
                                                    triplet = ("", model_item.original, version_match[index].original)
                                            overall_confidence = self.confidence_weight[0] * brand_confidence + \
                                                                 self.confidence_weight[
                                                                     1] * model_confidence + self.confidence_weight[
                                                                     2] * version_confidence
                                            if triplet not in processed_combinations:
                                                processed_combinations.add(triplet)
                                                valid_results.append(ValidationResult(
                                                    brand=triplet[0],
                                                    model=triplet[1],
                                                    version=triplet[2],
                                                    confidence=overall_confidence,
                                                    match_details={
                                                        'brand': f"none({brand_confidence})",
                                                        'model': f"fuzzy({model_confidence})",
                                                        'version': f"{version_match[index].match_type}({version_confidence})"
                                                    }
                                                ))
                for ver_item in version_match:
                    if ver_item not in version_match_dict[model_candidate]:
                        version_match_dict[model_candidate].append(version_match)

            if model_match_all == []:
                version_match_dict[""] = []
                # If no models match, version matching is skipped; need to handle this here
                for version_candidate in candidate_versions:
                    norm_version_candidate = self._normalize_text(version_candidate)
                    if norm_version_candidate in self.version_index.keys():
                        matched_versions = self.version_index[norm_version_candidate]
                    else:
                        continue
                    if matched_brand in self.database_brand.keys():
                        model_version_dict = self.database_brand[matched_brand]
                        for model, versions in model_version_dict.items():
                            for item_version in matched_versions:
                                if item_version in versions:
                                    version_tmp = MatchInfoVersion(
                                        original=version_candidate,
                                        matched=item_version,
                                        confidence=1,
                                        match_type="exact",
                                        brand=matched_brand,
                                        model=model
                                    )
                                    if version_tmp not in version_match_dict[""]:
                                        version_match_dict[""].append(version_tmp)
                                    triplet = ("", "", version_candidate)
                                    overall_confidence = self.confidence_weight[0] * 0.0 + \
                                                         self.confidence_weight[
                                                             1] * 0.0 + self.confidence_weight[
                                                             2] * 1.0
                                    if triplet not in processed_combinations:
                                        processed_combinations.add(triplet)
                                        valid_results.append(ValidationResult(
                                            brand=triplet[0],
                                            model=triplet[1],
                                            version=triplet[2],
                                            confidence=overall_confidence,
                                            match_details={
                                                'brand': f"{matched_brand}",
                                                'model': f"none()",
                                                'version': f"exact(1.0)"
                                            }
                                        ))


            # Summarize by brand
            model_match_dict[matched_brand] = model_match_all


        if all(v == [] for b, v in model_match_dict.items()):
            model_match_dict = {"": []}
            # Indicates potential brand issue
            for model_candidate in candidate_models:
                model_match_ = self._fuzzy_match_model(model_candidate)
                if model_match_ == []:
                    continue
                model_match_ = sort_by_confidence_desc(model_match_)
                # todo: Keep top 5 that have version_list
                count = 0
                for i, model_match_item in enumerate(model_match_):
                    version_list = model_match_item.version_list
                    if version_list == None:
                        continue  # todo: None indicates no corresponding definite brand for this model, so version was not extracted
                    try:
                        version_list_json = json.loads(version_list)
                        if version_list_json == []:
                            continue
                        else:
                            count += 1
                    except json.decoder.JSONDecodeError:
                        continue
                    if count == max_model_results:
                        break
                if count == 0:
                    count = min(max_model_results, len(model_match_))
                model_match_ = model_match_[:count]
                max_model_confidence = model_match_[0].confidence

                for model_item in model_match_:
                    version_list = self.database_model[model_item.matched]  # Is brand-version list dictionary
                    model_item.version_list = json.dumps(version_list)
                    if model_item not in model_match_dict[""]:
                        model_match_dict[""].append(model_item)
                        for version_candidate in candidate_versions:
                            version_match = []
                            for brand, versions in version_list.items():
                                for version in versions:
                                    if self._normalize_text(version) == self._normalize_text(version_candidate):
                                        version_tmp = MatchInfoVersion(
                                            original=version_candidate,
                                            matched=version,
                                            confidence=1,
                                            match_type="exact",
                                            brand=brand,
                                            model=model_item.matched
                                        )
                                        if version_tmp not in version_match:
                                            version_match.append(version_tmp)
                                        triplet = ("", model_item.original, version_candidate)
                                        overall_confidence = self.confidence_weight[0] * 0.0 + \
                                                             self.confidence_weight[
                                                                 1] * model_item.confidence + self.confidence_weight[
                                                                 2] * 1.0
                                        if triplet not in processed_combinations:
                                            processed_combinations.add(triplet)
                                            valid_results.append(ValidationResult(
                                                brand=triplet[0],
                                                model=triplet[1],
                                                version=triplet[2],
                                                confidence=overall_confidence,
                                                match_details={
                                                    'brand': f"none()",
                                                    'model': f"fuzzy({model_item.confidence})",
                                                    'version': f"exact(1.0)"
                                                }
                                            ))
                            if version_match:
                                for ver_item in version_match:
                                    if ver_item not in version_match_dict[model_candidate]:
                                        version_match_dict[model_candidate].append(version_match)
            if model_match_dict == {"": []}:  # Model incorrect too, look at version directly
                version_match_dict[""] = []
                for version_candidate in candidate_versions:
                    if self._normalize_text(version_candidate) in self.version_index.keys():
                        matched_versions = self.version_index[self._normalize_text(version_candidate)]
                        for version_item in matched_versions:
                            for b, model_dict in self.database_brand.items():
                                for model, version_list in model_dict.items():
                                    if version_item in version_list:
                                        version_tmp = MatchInfoVersion(
                                            original=version_candidate,
                                            matched=version_item,
                                            confidence=1,
                                            match_type="exact",
                                            brand=b,
                                            model=model
                                        )
                                        if version_tmp not in version_match_dict[""]:
                                            version_match_dict[""].append(version_tmp)
                                        triplet = ("", "", version_candidate)
                                        overall_confidence = self.confidence_weight[0] * 0.0 + \
                                                             self.confidence_weight[
                                                                 1] * 0.0 + self.confidence_weight[
                                                                 2] * 1.0
                                        if triplet not in processed_combinations:
                                            processed_combinations.add(triplet)
                                            valid_results.append(ValidationResult(
                                                brand=triplet[0],
                                                model=triplet[1],
                                                version=triplet[2],
                                                confidence=overall_confidence,
                                                match_details={
                                                    'brand': f"none()",
                                                    'model': f"none()",
                                                    'version': f"exact(1.0)"
                                                }
                                            ))

        # Sort by confidence in descending order
        valid_results.sort(key=lambda x: x.confidence, reverse=True)

        return valid_results, model_match_dict, version_match_dict

    def get_statistics(self) -> Dict[str, int]:
        """Get knowledge base statistics"""
        return {
            'total_brands': len(self.database_brand),
            'total_models': len(self.database_model),
            'total_versions': sum(len(versions) for brand_data in self.database_model.values()
                                  for versions in brand_data.values())
        }


class SimpleMatcher:
    """Simplified matcher (backup)"""

    def __init__(self, database_model):
        self.database_model = database_model

    def search(self, query, hint_brand=None, max_results=5):
        # Simplified implementation
        results = []
        for model in self.database_model.keys():
            if model in ["", "np", "NP", None]:
                continue
            if query.lower() in model.lower() or model.lower() in query.lower():
                results.append(type('Result', (), {
                    'model': model,
                    'confidence': 0.8,
                    'match_type': 'simple'
                })())
        return results[:max_results]


# Example usage
if __name__ == "__main__":
    # Example knowledge base
    with open('../data_analysis_module/deviceData/firmware_database_all_brand_0825.json', 'r', encoding='utf-8') as f:
        database_brand = json.load(f)
    with open('../data_analysis_module/deviceData/firmware_database_all_model_0825.json', 'r', encoding='utf-8') as f:
        database_model = json.load(f)

    # Create validator
    validator = TripletValidator(database_brand, database_model)

    # # Test specific cases: iphone14pro vs iPhone-15-Pro-Max
    # print("=== Specific Case Test ===")
    # print("Candidate: 'iphone14pro'")
    # print("In Knowledge Base: 'iPhone-15-Pro-Max'")
    #
    # # Test model component extraction
    # components_candidate = validator._extract_model_components('iphone14pro')
    # components_target = validator._extract_model_components('iPhone-15-Pro-Max')
    # print(f"\nCandidate Components: {components_candidate}")
    # print(f"Target Components: {components_target}")
    #
    # # Test semantic similarity
    # semantic_sim = validator._calculate_semantic_similarity('iphone14pro', 'iPhone-15-Pro-Max')
    # print(f"Semantic Similarity: {semantic_sim:.3f}")
    #
    # # Test fuzzy match
    # match_result = validator._fuzzy_match_model('iphone14pro', hint_brand='Apple', threshold=0.6)
    # if match_result:
    #     print(f"\nMatch Result:")
    #     print(f"  Original: {match_result.original}")
    #     print(f"  Matched: {match_result.matched}")
    #     print(f"  Confidence: {match_result.confidence:.3f}")
    #     print(f"  Match Type: {match_result.match_type}")
    # else:
    #     print("\nNo match found")
    #
    # print("\n" + "=" * 60)

    # Original full test
    candidate_brands = ["cisco", "tg-net"]
    candidate_models = ["c900", "c1900", "catalyst 1900"]
    candidate_versions = ["15.8(3)m6", "15.9(4)m1", "2.4.6"]

    print("=== Triplet Validation Test ===\n")
    print(f"Candidate Brands: {candidate_brands}")
    print(f"Candidate Models: {candidate_models}")
    print(f"Candidate Versions: {candidate_versions}")
    print(f"\nKnowledge Base Statistics: {validator.get_statistics()}")
    print("\n" + "=" * 60)

    # Validate triplets
    triplet_results, match_results = validator.validate_triplets(
        candidate_brands,
        candidate_models,
        candidate_versions,
        confidence_threshold=0.6
    )

    print(f"\nFound {len(triplet_results)} valid combinations:\n")

    for i, result in enumerate(triplet_results, 1):
        print(f"{i}. Brand: {result.brand}")
        print(f"   Model: {result.model}")
        print(f"   Version: {result.version}")
        print(f"   Overall Confidence: {result.confidence:.3f}")
        print(f"   Match Details: {result.match_details}")
        print(f"   Combination: ({result.brand}, {result.model}, {result.version})")
        print("-" * 50)
