"""
Multi-Agent Network Device Attribute Identification System
Integrates extractor, validator, and discriminator to implement a complete identification workflow
"""

import os, sys
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
from enum import Enum

# Get the absolute path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get parent directory (project root: project_path)
parent_dir = os.path.dirname(current_dir)

# Add parent directory to module search path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# Import existing modules
from query_llm_with_logprobs import VLLMModel, parse_llm_response, format_prompt
from Knowledge_graph_construct.result_verify import TripletValidator, ValidationResult
from hcs_construct.confidence_discriminator import ConfidenceDiscriminator
from analysis_token_logprobs import *
###@@@@@@@@@@###
from Prompt_templete_1012 import *
from tools.tool_global import *

class ExtractionStage(Enum):
    """Enumeration of extraction stages"""
    FIRST_PASS = "first_pass"  # First extraction
    KB_VERIFIED = "kb_verified"  # Knowledge base verification passed
    SECOND_PASS = "second_pass"  # Second extraction (with version list)
    FINAL_FILTERED = "final_filtered"  # Final filtering


@dataclass
class ExtractionResult:
    """Extraction result data class"""
    index: str
    stage: ExtractionStage

    # First extraction results
    brand: Optional[str] = None
    model: Optional[str] = None
    version: Optional[str] = None

    # Validation results
    validated_brand: Optional[str] = None
    validated_model: Optional[str] = None
    validated_version: Optional[str] = None
    validation_confidence: float = 0.0
    validation_matched: bool = False

    # Second extraction results (if needed)
    refined_brand: Optional[str] = None
    refined_model: Optional[str] = None
    refined_version: Optional[str] = None
    version_pattern_learned: bool = False

    # Logprobs and confidence analysis
    logprobs: Optional[List] = None
    confidence_analysis: Optional[Dict] = None

    # Discriminator results
    discriminator_scores: Optional[Dict] = None
    final_decision: str = "PENDING"
    final_confidence: float = 0.0

    # Complete raw outputs
    raw_outputs: Dict = None

    def __post_init__(self):
        if self.raw_outputs is None:
            self.raw_outputs = {}


class MultiAgentExtractionSystem:
    """
    Main controller for the Multi-Agent Device Attribute Identification System

    Workflow:
    1. First Extraction: Extract brand, model, version from banner using LLM
    2. Knowledge Base Verification: Look for matches in device_database
    3. Second Extraction (Optional): If not matched, extract again using version list as prompt
    4. Confidence Discrimination: Calculate final confidence using trained discriminator
    5. Decision Output: Accept/Reject based on confidence
    """

    def __init__(self,
                 vllm_model: VLLMModel,
                 validator: TripletValidator,
                 confidence_analyzer:  AttributeConfidenceAnalyzer,
                 discriminator: ConfidenceDiscriminator,
                 first_pass_prompt: str,
                 second_pass_prompt_template: str,
                 confidence_threshold: float = 0.7,
                 enable_second_pass: bool = True):
        """
        Initialize Multi-Agent System

        Args:
            vllm_model: Loaded VLLM model
            validator: Knowledge base validator
            discriminator: Confidence discriminator
            first_pass_prompt: Prompt template for first extraction
            second_pass_prompt_template: Prompt template for second extraction (contains {{VERSION_LIST}} placeholder)
            confidence_threshold: Confidence threshold for validator
            enable_second_pass: Whether to enable second extraction
        """
        self.vllm_model = vllm_model
        self.validator = validator
        self.confidence_analyzer = confidence_analyzer
        self.discriminator = discriminator
        self.first_pass_prompt = first_pass_prompt
        self.second_pass_prompt_template = second_pass_prompt_template
        self.confidence_threshold = confidence_threshold
        self.enable_second_pass = enable_second_pass

        # Statistics
        self.stats = {
            'total': 0,
            'first_pass_matched': 0,
            'second_pass_triggered': 0,
            'second_pass_matched': 0,
            'final_accepted': 0,
            'final_rejected': 0
        }

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    ###@@@@@@@@@@###
    def extract_once(self,
                     banner: str,
                     prompt_template: str,
                     context_info: Optional[list]=[]) -> Tuple[Dict, List, str]:
        """
        Perform one LLM extraction

        Args:
            banner: Device banner
            prompt_template: Prompt template
            context_info: Additional context information (e.g., version list)

        Returns:
            (parsed_output, logprobs, raw_text)
        """
        # Format prompt
        if context_info:
            if context_info[0] == None:
                context_info[0] = "[]"
            if context_info[1] == None:
                context_info[1] = "{}"
            prompt = prompt_template.replace("{{BANNER}}", banner)
            prompt = prompt.replace("{{KNOWN_VERSION_LIST}}", context_info[0])
            prompt.replace("{{FIRST_EXTRACTION_JSON}}", context_info[1])

        else:
            prompt = format_prompt(prompt_template, banner)

        # Call model
        output_text, token_logprobs = self.vllm_model.generate(
            prompt,
            stop=["</results>", "</RESULTS>", "\n\n\n\n"]
        )

        # Parse output
        parsed_output = parse_llm_response(output_text)

        return parsed_output, token_logprobs, output_text

    def validate_extraction(self,
                            brand: str,
                            model: str,
                            version: str) -> Tuple[List[ValidationResult], Dict, Dict]:
        """
        Validate extraction results using knowledge base

        Returns:
            (validation_results, model_matches, version_matches)
        """
        # Prepare candidate lists
        candidate_brands = [brand] if brand else []
        candidate_models = [model] if model else []
        candidate_versions = [version] if version else []

        # Call validator
        validation_results, model_matches, version_matches = self.validator.validate_triplets(
            candidate_brands,
            candidate_models,
            candidate_versions,
            confidence_threshold=self.confidence_threshold,
            max_model_results=5
        )

        return validation_results, model_matches, version_matches

    def get_version_list_context(self,
                                 model_matches: Dict) -> Optional[str]:
        """
        Extract version list from model matches for second extraction

        Args:
            model_matches: model_matches returned from validate_extraction

        Returns:
            Formatted version list string, or None if none
        """
        if not model_matches:
            return None

        # Get version list for the first brand
        versions = []
        for brand, matches in model_matches.items():
            if not matches or brand == "":
                continue

        # Try to get version list of the first matching model
        for match in matches:  # Take only top 3 matches
            try:
                version_list = json.loads(match.version_list)
                if version_list and isinstance(version_list, list):
                    for item in version_list:
                        if item.strip() not in versions:
                            versions.append(item.strip())
                    # Format version list
                    version_str = json.dumps(versions)  # Limit to at most 20
                    return f"{version_str}"
            except:
                continue

        return None

    def apply_discriminator(self,
                            confidence_analysis: Dict,
                            parsed_output: Dict) -> Dict:
        """
        Apply confidence discriminator

        Returns:
            discriminator_results: {brand: {...}, model: {...}, version: {...}}
        """
        discriminator_results = {}

        for attr in ['brand', 'model', 'firmware_version']:
            try:
                result = self.discriminator.predict(confidence_analysis, attr)
                discriminator_results[attr] = result
            except Exception as e:
                self.logger.warning(f"Discriminator prediction failed ({attr}): {e}")
                discriminator_results[attr] = {
                    'is_reliable': False,
                    'confidence_score': 0.0,
                    'decision': 'REJECT',
                    'error': str(e)
                }

        return discriminator_results

    def make_final_decision(self,
                            discriminator_results: Dict,
                            validation_matched: bool,
                            validation_confidence: float) -> Tuple[str, float]:
        """
        Make final decision combining discriminator and validation results

        Returns:
            (decision, final_confidence)
        """
        # If knowledge base verification passed and confidence is high, accept preferentially
        if validation_matched and validation_confidence > 0.8:
            return "ACCEPT", validation_confidence

        # Otherwise rely on discriminator
        # Calculate average confidence
        scores = []
        for attr in ['brand', 'model', 'firmware_version']:
            if attr in discriminator_results:
                scores.append(discriminator_results[attr].get('confidence_score', 0.0))

        avg_confidence = sum(scores) / len(scores) if scores else 0.0

        # Check if all attributes are accepted
        all_accepted = all(
            discriminator_results.get(attr, {}).get('decision') == 'ACCEPT'
            for attr in ['brand', 'model', 'firmware_version']
        )

        if all_accepted and avg_confidence > 0.5:
            return "ACCEPT", avg_confidence
        else:
            return "REJECT", avg_confidence

    def process_single_sample(self,
                              sample: Dict,
                              confidence_analyzer: AttributeConfidenceAnalyzer,
                              enable_debug: bool = False,
                              use_first_results: bool = False) -> ExtractionResult:
        """
        Process the complete workflow for a single sample

        Args:
            sample: Dictionary containing 'index', 'banner', 'label' etc.
            enable_debug: Whether to print debug information

        Returns:
            ExtractionResult object
        """
        index = sample.get('index', 'unknown')
        # print(f"Processing sample {index}...")
        banner = sample.get('banner', '').strip()

        if not banner:
            self.logger.warning(f"Banner for sample {index} is empty")
            return ExtractionResult(index=index, stage=ExtractionStage.FIRST_PASS)

        result = ExtractionResult(index=index, stage=ExtractionStage.FIRST_PASS)

        # ===== Stage 1: First Extraction =====
        # if enable_debug:
        #     print(f"\n{'=' * 60}")
        print(f"Processing sample: {index}")
        #     print(f"{'=' * 60}")
        logprobs = []
        raw_text = ''
        if not use_first_results:
            parsed_output, logprobs, raw_text = self.extract_once(
                banner, self.first_pass_prompt
            )

        ###@@@@@@@@@@###
        else:

            parsed_output, logprobs, raw_text = sample.get('parsed_output', {}), sample.get('logprobs', []), sample.get('init_output', '')
            if not parsed_output:
                parsed_output = sample.get('raw_outputs').get('first_pass').get('parsed')

        if logprobs:
            # Analyze
            analysis_results = confidence_analyzer.analyze_full_output(
                raw_text,
                logprobs
            )
        else:
            analysis_results = sample.get('raw_outputs').get('first_pass').get('parsed').get('confidence_analysis', {})

        # Add to data
        parsed_output['confidence_analysis'] = analysis_results
        ###@@@@@@@@@@###

        result.logprobs = logprobs
        result.raw_outputs['first_pass'] = {
            'parsed': parsed_output,
            'raw': raw_text
        }

        ###@@@@@@@@@@###
        # result.logprobs = logprobs
        # result.raw_outputs['first_pass'] = {
        #     'parsed': analysis_results,
        #     'raw': raw_text
        # }
        ###@@@@@@@@@@###

        # Extract results
        if parsed_output.get('parsed_json'):
            extraction = parsed_output['parsed_json']
            result.brand = extraction.get('brand', {}).get('value', 0.0)
            result.model = extraction.get('model', {}).get('value', 0.0)
            result.version = extraction.get('firmware_version', {}).get('value', 0.0)
            # result.confidence_analysis = extraction.get('confidence_analysis', {})
            ###@@@@@@@@@@###
            result.confidence_analysis = analysis_results

        # if enable_debug:
        print(f"First Extraction: Brand={result.brand}, Model={result.model}, Version={result.version}")

        # ===== Stage 2: Knowledge Base Verification =====
        validation_results, model_matches, version_matches = self.validate_extraction(
            result.brand or "", result.model or "", result.version or ""
        )


        # result.raw_outputs['validation'] = {
        #     'results': [vars(v) for v in validation_results],
        #     'model_matches': {k: [vars(m) for m in v] for k, v in model_matches.items()},
        #     'version_matches': {k: [vars(m) for m in v] for k, v in version_matches.items()}
        # }
        result.raw_outputs['validation'] = {
            'results': [],
            'model_matches': {},
            'version_matches': {}
        }
        if validation_results:
            for i, v in enumerate(validation_results):
                ### If identified brand is empty, but validated model is not, supplement brand info
                if v.brand == "" and v.model != "":
                    highest_confidence_item = max(model_matches[""], key=lambda item: item.confidence)
                    v.brand = highest_confidence_item.brand
                    validation_results[i].brand = highest_confidence_item.brand
                result.raw_outputs['validation']['results'].append(vars(v))
        if model_matches:
            for k, v in model_matches.items():
                if v != []:
                    result.raw_outputs['validation']['model_matches'][k] = [vars(m) for m in v]
                else:
                    result.raw_outputs['validation']['model_matches'][k] = []
        if version_matches:
            for k, v in version_matches.items():
                if k not in result.raw_outputs['validation']['version_matches'].keys():
                    result.raw_outputs['validation']['version_matches'][k] = []
                if v != []:
                    for m in v:
                        if isinstance(m, list):
                            for m_item in m:
                                if isinstance(m_item, list):
                                    for m_item_item in m_item:
                                        if vars(m_item_item) not in result.raw_outputs['validation']['version_matches'][k]:
                                            result.raw_outputs['validation']['version_matches'][k].append(vars(m_item_item))
                                else:
                                    if vars(m_item) not in result.raw_outputs['validation']['version_matches'][k]:
                                        result.raw_outputs['validation']['version_matches'][k].append(vars(m_item))
                        else:
                            if vars(m) not in result.raw_outputs['validation']['version_matches'][k]:
                                result.raw_outputs['validation']['version_matches'][k].append(vars(m))

                    # result.raw_outputs['validation']['version_matches'][k] = [vars(m) for m in v]



        # Decide whether validation is needed
        discriminator_flag = True
        # Check match
        print(f"True Label: {sample.get('new_label', sample.get('label'))}")
        try:
            print(f"KB Verification Result: [{validation_results[0].brand}, {validation_results[0].model}, {validation_results[0].version}]")
        except IndexError:
            print("KB Verification Result: No match")
        if validation_results and validation_results[0].confidence > self.confidence_threshold:
            print("KB Verification Passed, Confidence:", validation_results[0].confidence)
            result.stage = ExtractionStage.KB_VERIFIED
            result.validation_matched = True
            result.validation_confidence = validation_results[0].confidence
            result.validated_brand = validation_results[0].brand
            result.validated_model = validation_results[0].model
            result.validated_version = validation_results[0].version

            self.stats['first_pass_matched'] += 1
            discriminator_flag = False

            # if enable_debug:
            #     print(
            #         f"✓ KB Verification Passed: {result.validated_brand} / {result.validated_model} / {result.validated_version}")
            #     print(f"  Confidence: {result.validation_confidence:.3f}")

        # ===== Stage 3: Second Extraction (If needed and enabled) =====
        elif self.enable_second_pass:
            result.stage = ExtractionStage.SECOND_PASS
            self.stats['second_pass_triggered'] += 1

            # Get version list context
            version_context = self.get_version_list_context(model_matches)
            first_extract = parsed_output.get('parsed_json')
            print(f"Version Context: {version_context}")

            if version_context:
                result.version_pattern_learned = True

                # if enable_debug:
                #     print(f"Triggered Second Extraction, Version Context: {version_context[:100]}...")

            # Execute second extraction
            parsed_output2, logprobs2, raw_text2 = self.extract_once(
                banner, self.second_pass_prompt_template, [version_context, json.dumps(first_extract)]
            )

            # print(f"Second Extraction Result: {parsed_output2}")
            ###@@@@@@@@@@###
            analysis_results2 = {}
            # parsed_output2, logprobs2, raw_text2 = sample.get('parsed_output'), sample.get('token_logprobs'), sample.get('init_output')
            ###@@@@@@@@@###

            # Analyze
            analysis_results2 = confidence_analyzer.analyze_full_output(
                raw_text2,
                logprobs2
            )
            result.logprobs = logprobs2
            ###@@@@@@@@@@###
            parsed_output2["confidence_analysis"] = analysis_results2
            result.raw_outputs['second_pass'] = {
                'parsed': parsed_output2,
                'raw': raw_text2
            }

            # Update extraction results
            if parsed_output2.get('parsed_json'):
                extraction2 = parsed_output2['parsed_json']
                result.refined_brand = extraction2.get('brand', result.brand).get('value', 0.0)
                result.refined_model = extraction2.get('model', result.model).get('value', 0.0)
                result.refined_version = extraction2.get('firmware_version', result.version).get('value', 0.0)
                print(f"Second Extraction Result: [{result.refined_brand}, {result.refined_model}, {result.refined_version}]")
                # Update confidence analysis
                # if extraction2.get('confidence_analysis'):
                #     result.confidence_analysis = extraction2['confidence_analysis']
                ###@@@@@@@@@@###
                if analysis_results2:
                    result.confidence_analysis = analysis_results2
                ###@@@@@@@@@@###


                # Validate again
                validation_results2, _, _ = self.validate_extraction(
                    result.refined_brand or "",
                    result.refined_model or "",
                    result.refined_version or ""
                )
                if validation_results2:
                    if validation_results2[0].version.lower() not in banner.lower():
                        validation_results2[0].version = ""
                    if validation_results2[0].brand.lower() not in banner.lower():
                        validation_results2[0].brand = ""
                try:
                    print(f"Second Verification Result: [{validation_results2[0].brand}, {validation_results2[0].model}, {validation_results2[0].version}]")
                except IndexError:
                    print("Second Verification Result is Empty")
                if validation_results2 and validation_results2[0].confidence > self.confidence_threshold:
                    result.validation_matched = True
                    result.validation_confidence = validation_results2[0].confidence
                    result.validated_brand = validation_results2[0].brand
                    result.validated_model = validation_results2[0].model
                    result.validated_version = validation_results2[0].version

                    self.stats['second_pass_matched'] += 1

                    # if enable_debug:
                    print(f"✓ Validation passed after second extraction")
                    discriminator_flag = False

        # ===== Stage 4: Apply Discriminator =====
        result.stage = ExtractionStage.FINAL_FILTERED

        if result.confidence_analysis and discriminator_flag:
            result.discriminator_scores = self.apply_discriminator(
                result.confidence_analysis,
                parsed_output
            )

            print(f"Discriminator Result: brand: {result.discriminator_scores['brand']['decision']}, model: {result.discriminator_scores['model']['decision']}, firmware_version: {result.discriminator_scores['firmware_version']['decision']}")

            # Make final decision
            result.final_decision, result.final_confidence = self.make_final_decision(
                result.discriminator_scores,
                result.validation_matched,
                result.validation_confidence
            )

            if result.final_decision == "ACCEPT":
                self.stats['final_accepted'] += 1
            else:
                self.stats['final_rejected'] += 1

            if enable_debug:
                print(f"Final Decision: {result.final_decision} (Confidence: {result.final_confidence:.3f})")

        return result


    def process_dataset(self,
                        input_file: str,
                        output_file: str,
                        confidence_analyzer: AttributeConfidenceAnalyzer,
                        debug_mode: bool = False,
                        use_first_results: bool = False,
                        max_samples: Optional[int] = None):
        """
        Process dataset in batch

        Args:
            input_file: Input file path (JSONL format)
            output_file: Output file path
            debug_mode: Whether to enable debug mode (print detailed info)
            max_samples: Max number of samples to process (for testing)
        """
        self.logger.info(f"Start processing dataset: {input_file}")
        self.logger.info(f"Output will be saved to: {output_file}")

        # Read data
        samples = []
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        except json.decoder.JSONDecodeError:
            with open(input_file, 'r', encoding='utf-8') as f:
                tmp_data = json.load(f)
                for brand, sam in tmp_data.items():
                    samples += sam

        if max_samples:
            samples = samples[:max_samples]

        self.stats['total'] = len(samples)
        self.logger.info(f"Total {len(samples)} samples")

        # Process each sample
        results = []
        with open(output_file, 'a+', encoding='utf-8') as f_out:
            flag = False
            for idx, sample in enumerate(tqdm(samples, desc="Processing Samples")):
                try:
                    index_sample = sample.get("index", sample.get('sample_id', idx))
                    # if index_sample == '0721_35812@2':
                    #     flag = True
                    # if not flag:
                    #     continue
                    # if index_sample not in sft_index:
                    #     continue
                    # Process single sample
                    result = self.process_single_sample(
                        sample,
                        confidence_analyzer=confidence_analyzer,
                        enable_debug=debug_mode and idx < 3,  # Only debug first 3
                        use_first_results=use_first_results  # Use existing first extraction results
                    )

                    # Construct output
                    output_data = {
                        'index': result.index,
                        'banner': sample["banner"],
                        'label': sample.get('new_label', sample.get('label')),  # Add label
                        'stage': result.stage.value,
                        'first_extraction': {
                            'brand': result.brand,
                            'model': result.model,
                            'version': result.version
                        },
                        'validation': {
                            'matched': str(result.validation_matched),
                            'confidence': result.validation_confidence,
                            'validated_brand': result.validated_brand,
                            'validated_model': result.validated_model,
                            'validated_version': result.validated_version
                        },
                        'second_extraction': {
                            'triggered': str(result.stage == ExtractionStage.SECOND_PASS),
                            'pattern_learned': str(result.version_pattern_learned),
                            'refined_brand': result.refined_brand,
                            'refined_model': result.refined_model,
                            'refined_version': result.refined_version
                        },
                        'discriminator': result.discriminator_scores,
                        'final_decision': result.final_decision,
                        'final_confidence': result.final_confidence,
                        'confidence_analysis': result.confidence_analysis,
                        'raw_outputs': result.raw_outputs,
                        'logprobs': result.logprobs,
                        'original_data': sample
                    }

                    # Write to file
                    out_final = convert_to_serializable(output_data)
                    # sample["output_data"] = output_data
                    f_out.write(json.dumps(out_final, ensure_ascii=False) + '\n')
                    f_out.flush()

                    results.append(result)

                except Exception as e:
                    self.logger.error(f"Error processing sample {sample.get('index', idx)}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        # Print statistics
        self.print_statistics()

        return results

    def print_statistics(self):
        """Print processing statistics"""
        print("\n" + "=" * 80)
        print("Processing Statistics")
        print("=" * 80)
        print(f"Total Samples: {self.stats['total']}")
        print(
            f"Matched on First Pass: {self.stats['first_pass_matched']} ({self.stats['first_pass_matched'] / max(self.stats['total'], 1) * 100:.1f}%)")
        print(
            f"Triggered Second Pass: {self.stats['second_pass_triggered']} ({self.stats['second_pass_triggered'] / max(self.stats['total'], 1) * 100:.1f}%)")
        print(
            f"Matched after Second Pass: {self.stats['second_pass_matched']} ({self.stats['second_pass_matched'] / max(self.stats['total'], 1) * 100:.1f}%)")
        print(
            f"Final Accepted: {self.stats['final_accepted']} ({self.stats['final_accepted'] / max(self.stats['total'], 1) * 100:.1f}%)")
        print(
            f"Final Rejected: {self.stats['final_rejected']} ({self.stats['final_rejected'] / max(self.stats['total'], 1) * 100:.1f}%)")
        print("=" * 80)


# ==================== Main Program Entry ====================

def main():
    """Main program"""
    import sys

    # ===== Configuration Parameters =====
    CONFIG = {
        'model_path': "model_path", #/Qwen2___5-7B-Instruct",#/finetuned/agent_sft_Qwen2___5_7B_Instruct_1124_merged",
        'gpu_ids': "0",
        'database_brand_path': '../device_database/firmware_database_all_brand_1118.json',
        'database_model_path': '../device_database/firmware_database_all_model_1118.json',
        'discriminator_path': '../hcs_construct/discriminator_results/discriminator.pkl',
        'input_file': '../data/holdout_set.jsonl',
        # 'input_file': '../llm_results/relabel_merged_cydar_data_into_test_1012_Qwen2___5-7B-Instruct_confidence_detail_analysis_1105_1111.json',
        # 'first_extract_result_file': '../data/holdout_set_Qwen2___5-7B-Instruct_confidence_detail_analysis_1105_1114.json',
        'first_extract_result_file': '../results/test_set_multi_agent_results_sft_Qwen2___5_7B_Instruct_1130.jsonl',
        'output_file': '../results/test_set_multi_agent_results_sft_Qwen2___5_7B_Instruct_1130_secondround.jsonl',#test_set_multi_agent_results_Qwen2___5_7B_Instruct.jsonl', # agent_sft_Qwen2___5_7B_Instruct_1124 #_qwen2__5.jsonl',  # qwen2__5 suffix indicates the second round extraction uses Qwen2___5_7B_Instruct model instead of the finetuned one
        # 'output_file': '../llm_results/agent_test.jsonl',
        'first_pass_prompt_name': 'confidence_detail_analysis_1105',
        'second_pass_prompt_name': 'verify_with_knowledge_base_1127',
        'confidence_threshold': 0.7,
        'enable_second_pass': True,
        'debug_mode': True,
        'max_samples': None,  # Set to a number to limit processed samples
        'use_first_results': True   # Whether to use existing first extraction results
    }

    # Override configuration from command line
    if len(sys.argv) > 1:
        CONFIG['input_file'] = sys.argv[1]
    if len(sys.argv) > 2:
        CONFIG['output_file'] = sys.argv[2]

    print("=" * 80)
    print("Multi-Agent Device Attribute Identification System")
    print("=" * 80)
    print(f"Input File: {CONFIG['input_file']}")
    print(f"Output File: {CONFIG['output_file']}")
    print(f"Model: {CONFIG['model_path']}")
    print(f"Enable Second Extraction: {CONFIG['enable_second_pass']}")
    print("=" * 80)

    # ===== 1. Load VLLM Model =====
    print("\n[1/4] Loading VLLM Model...")
    vllm_model = VLLMModel(CONFIG['model_path'], gpu_ids=CONFIG['gpu_ids'])
    # Alternative: Load result file
    result_data = []
    # with open("", 'r', encoding='utf-8') as f:
    #     for line in f:
    #         result_data.append(json.loads(line))

    # ===== 2. Load Knowledge Base and Validator =====
    print("\n[2/4] Loading Knowledge Base...")
    with open(CONFIG['database_brand_path'], 'r', encoding='utf-8') as f:
        database_brand = json.load(f)
    with open(CONFIG['database_model_path'], 'r', encoding='utf-8') as f:
        database_model = json.load(f)

    validator = TripletValidator(database_brand, database_model)
    print(f"Knowledge Base Stats: {validator.get_statistics()}")

    # ===== 3. Load Discriminator =====
    print("\n[3/4] Loading Confidence Calculator...")
    analyzer = AttributeConfidenceAnalyzer()
    print("\n[3/4] Loading Confidence Discriminator...")
    discriminator = ConfidenceDiscriminator.load(CONFIG['discriminator_path'])

    # ===== 4. Load Prompt Templates =====
    print("\n[4/4] Loading Prompt Templates...")

    first_pass_prompt = globals()[CONFIG['first_pass_prompt_name']]
    second_pass_prompt = globals()[CONFIG['second_pass_prompt_name']]

    # ===== 5. Create Multi-Agent System =====
    print("\nInitializing Multi-Agent System...")
    system = MultiAgentExtractionSystem(
        vllm_model=vllm_model,
        # vllm_model=result_data,
        confidence_analyzer=analyzer,
        validator=validator,
        discriminator=discriminator,
        first_pass_prompt=first_pass_prompt,
        second_pass_prompt_template=second_pass_prompt,
        confidence_threshold=CONFIG['confidence_threshold'],
        enable_second_pass=CONFIG['enable_second_pass']
    )

    # ===== 6. Process Dataset =====
    print("\nStart Processing Dataset...\n")
    os.makedirs(os.path.dirname(CONFIG['output_file']), exist_ok=True)



    if CONFIG['use_first_results']:
        results = system.process_dataset(
            input_file=CONFIG['first_extract_result_file'],
            output_file=CONFIG['output_file'],
            confidence_analyzer=analyzer,
            debug_mode=CONFIG['debug_mode'],
            max_samples=CONFIG['max_samples'],
            use_first_results=CONFIG['use_first_results']
        )
    else:
        results = system.process_dataset(
            input_file=CONFIG['input_file'],
            output_file=CONFIG['output_file'],
            confidence_analyzer=analyzer,
            debug_mode=CONFIG['debug_mode'],
            max_samples=CONFIG['max_samples'],
            use_first_results=CONFIG['use_first_results']
        )

    print(f"\nProcessing Complete! Results saved to: {CONFIG['output_file']}")


if __name__ == "__main__":
    sft_index = []
    # Strictly ensure no overlap with the training data
    with open('../data/sft_dataset_CORRECT_only_1130_id.json', 'r', encoding='utf-8') as sft_file:
        sft_index = json.load(sft_file)
    main()
