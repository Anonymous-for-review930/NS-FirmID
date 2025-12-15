import json
import re
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def fix_json_string(s: str):
    """
    Attempts to fix common JSON format errors:
    - Remove extra commas
    - Remove markdown remnants
    - Replace single quotes with double quotes
    - Remove illegal control characters
    """
    s = s.strip()
    s = re.sub(r"```+", "", s)
    s = re.sub(r"json", "", s, flags=re.IGNORECASE)
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    s = s.replace("'", '"')
    s = re.sub(r"[\x00-\x1F]+", "", s)
    return s.strip()


class AttributeConfidenceAnalyzer:
    """
    Analyzes extraction confidence for each attribute
    Evaluates by parsing reasoning blocks and corresponding token logprobs
    """

    def __init__(self):
        # Attribute name mapping
        self.attributes = [
            'brand', 'model', 'firmware_version',
            'os', 'web_server', 'sdk'
        ]

        # Weights for key tokens (these tokens better reflect confidence)
        self.key_token_patterns = {
            'certainty_high': ['explicitly', 'clearly', 'states', 'labeled', 'mentioned'],
            'certainty_medium': ['appears', 'suggests', 'indicates', 'likely', 'probably'],
            'certainty_low': ['might', 'could', 'possibly', 'unclear', 'ambiguous'],
            'certainty_null': ['not found', 'no', 'cannot', 'unable', 'missing']
        }

    def extract_reasoning_sections(self, output_text: str) -> Dict[str, str]:
        """
        Extracts the reasoning section for each attribute from the output text

        Returns:
            dict: {attribute_name: reasoning_text}
        """
        reasoning_sections = {}

        for attr in self.attributes:
            pattern = f"<{attr}_reasoning>(.*?)</{attr}_reasoning>"
            match = re.search(pattern, output_text, re.DOTALL | re.IGNORECASE)

            if match:
                reasoning_sections[attr] = match.group(1).strip()
            else:
                reasoning_sections[attr] = ""

        return reasoning_sections

    def extract_analysis_block(self, output_text: str):
        """
        Extracts the <analysis> block from model output and handles common exceptions.
        """

        analysis = ""

        # 1. Standard format: <analysis>...</analysis>
        match = re.search(r"<analysis>(.*?)</analysis>", output_text, re.DOTALL | re.IGNORECASE)
        if match:
            analysis = match.group(1).strip()
            return analysis

        # 2. Compatible format: ```analysis ... </analysis>
        match = re.search(r"```analysis(.*?)</analysis>", output_text, re.DOTALL | re.IGNORECASE)
        if match:
            analysis = match.group(1).strip()
            # result["warning"] = "Found non-standard start marker ```analysis, automatically corrected."
            return analysis

        # 3. Compatible format: <analysis> ... ``` (Model missed the closing tag)
        match = re.search(r"<analysis>(.*?)```", output_text, re.DOTALL | re.IGNORECASE)
        if match:
            analysis = match.group(1).strip()
            # result["warning"] = "Missing </analysis> closing tag, attempted automatic truncation."
            return analysis

        # 4. Compatible format: ```analysis ... ```
        match = re.search(r"```analysis(.*?)```", output_text, re.DOTALL | re.IGNORECASE)
        if match:
            analysis = match.group(1).strip()
            # result["warning"] = "Used markdown format ```analysis```, automatically extracted."
            return analysis

        # 5. Fallback processing: If there are obvious analysis paragraph clues (e.g., containing reasoning, brand, model, etc.)
        fallback_match = re.search(
            r"(brand_reasoning|model_reasoning|firmware_version_reasoning).*?(?=```|{|$)",
            output_text,
            re.DOTALL | re.IGNORECASE
        )
        if fallback_match:
            analysis = fallback_match.group(0).strip()
            # result["warning"] = "No <analysis> tag detected, but found similar analysis content, automatically extracted."
            return analysis

        # ❌ 6. Really cannot find it
        analysis = None
        # result["warning"] = "No <analysis> block or identifiable analysis content found."
        return analysis


    def extract_json_result(self, output_text: str):
        """
        Extracts the JSON result block from model output, compatible with various non-standard formats:
        - ```json ... ```
        - <results> ... </results>
        - Multiple JSONs, automatically takes the last valid one
        - With extra explanatory text or comments
        """

        result = ""

        # ✅ 1. Prioritize matching ```json ... ```
        json_blocks = re.findall(r"```json(.*?)```", output_text, re.DOTALL | re.IGNORECASE)
        if json_blocks:
            # Take the last one (usually the final result)
            last_block = json_blocks[-1].strip()
            try:
                result = json.loads(last_block)
                return result
            except json.JSONDecodeError:
                # result["warning"] = "Found ```json``` block, but parsing failed, attempting to fix format."
                fixed = fix_json_string(last_block)
                try:
                    result = json.loads(fixed)
                    # result["warning"] += " Automatically fixed."
                    return result
                except Exception:
                    pass  # Continue to other modes

        # ✅ 2. Match <results> ... </results>
        match = re.search(r"<results>(.*?)</results>", output_text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                # result["warning"] = "Detected <results> block, but JSON parsing failed, attempting to fix."
                fixed = fix_json_string(content)
                try:
                    result = json.loads(fixed)
                    # result["warning"] += " Automatically fixed."
                    return result
                except Exception:
                    pass

        # ✅ 3. Extract the first JSON block directly from the text (most common fallback case)
        match = re.search(r"(\{.*\})", output_text, re.DOTALL)
        if match:
            raw_json = match.group(1)
            try:
                result = json.loads(raw_json)
                return result
            except json.JSONDecodeError:
                # result["warning"] = "Detected raw JSON block but parsing failed, attempting to fix."
                fixed = fix_json_string(raw_json)
                try:
                    result = json.loads(fixed)
                    # result["warning"] += " Automatically fixed."
                    return result
                except Exception:
                    pass

        # ❌ 4. Really cannot find it
        # result["data"] = None
        # result["warning"] = "No parsable JSON result block found."
        return result


    # def extract_results_sections(self, output_text: str) -> Dict[str, Dict]:
    #     """
    #     Extracts values and model-claimed confidence for each attribute from the output JSON
    #
    #     Returns:
    #         dict: {attribute_name: {'value': ..., 'claimed_confidence': ...}}
    #     """
    #     results = {}
    #
    #     # Extract JSON part
    #     json_match = re.search(
    #         r"<results>(.*?)</results>",
    #         output_text,
    #         re.DOTALL | re.IGNORECASE
    #     )
    #
    #     if not json_match:
    #         return {}
    #
    #     json_text = json_match.group(1).strip()
    #     json_match_inner = re.search(r"```json\s*(.*?)\s*```", json_text, re.DOTALL)
    #     if json_match_inner:
    #         json_text = json_match_inner.group(1)
    #
    #     try:
    #         start_idx = json_text.find('{')
    #         end_idx = json_text.rfind('}')
    #         if start_idx != -1 and end_idx > start_idx:
    #             parsed_json = json.loads(json_text[start_idx:end_idx + 1])
    #
    #             # Extract basic attributes
    #             for attr in ['brand', 'model', 'firmware_version']:
    #                 if attr in parsed_json:
    #                     results[attr] = {
    #                         'value': parsed_json[attr].get('value'),
    #                         'claimed_confidence': parsed_json[attr].get('confidence', 0.0)
    #                     }
    #
    #             # Extract component_versions
    #             if 'component_versions' in parsed_json:
    #                 for attr in ['os', 'web_server', 'sdk']:
    #                     if attr in parsed_json['component_versions']:
    #                         results[attr] = {
    #                             'value': parsed_json['component_versions'][attr].get('value'),
    #                             'claimed_confidence': parsed_json['component_versions'][attr].get('confidence', 0.0)
    #                         }
    #     except Exception as e:
    #         print(f"Error parsing JSON: {e}")
    #
    #     return results

    def extract_results_sections(self, output_text: str) -> Dict[str, Dict]:
        """
        Extracts values and model-claimed confidence for each attribute from the output JSON

        Returns:
            dict: {attribute_name: {'value': ..., 'claimed_confidence': ...}}
        """
        results = {}
        try:
            parsed_json = self.extract_json_result(output_text)

            # Extract basic attributes
            for attr in ['brand', 'model', 'firmware_version']:
                if attr in parsed_json:
                    results[attr] = {
                        'value': parsed_json[attr].get('value'),
                        'claimed_confidence': parsed_json[attr].get('confidence', 0.0)
                    }

            # Extract component_versions
            if 'component_versions' in parsed_json:
                for attr in ['os', 'web_server', 'sdk']:
                    if attr in parsed_json['component_versions']:
                        results[attr] = {
                            'value': parsed_json['component_versions'][attr].get('value'),
                            'claimed_confidence': parsed_json['component_versions'][attr].get('confidence', 0.0)
                        }
        except Exception as e:
            print(f"Error parsing JSON: {e}")

        return results

    def map_tokens_to_reasoning(self,
                                output_text: str,
                                token_logprobs: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Maps token logprobs to the corresponding reasoning section

        Returns:
            dict: {attribute_name: [token_logprob_dicts]}
        """
        if not token_logprobs:
            return {attr: [] for attr in self.attributes}

        # Reconstruct full text and token correspondence
        full_text = ""
        token_positions = []

        for token_data in token_logprobs:
            token_str = token_data.get('decoded_token', '')
            start_pos = len(full_text)
            full_text += token_str
            end_pos = len(full_text)
            token_positions.append({
                'start': start_pos,
                'end': end_pos,
                'token_data': token_data
            })

        # Find location of each reasoning section in the text
        attribute_tokens = {attr: [] for attr in self.attributes}

        for attr in self.attributes:
            pattern = f"<{attr}_reasoning>(.*?)</{attr}_reasoning>"
            match = re.search(pattern, output_text, re.DOTALL | re.IGNORECASE)

            if match:
                section_start = match.start(1)
                section_end = match.end(1)

                # Find all tokens within this range
                for token_pos in token_positions:
                    # If any part of the token is within the reasoning section range
                    if (token_pos['start'] < section_end and
                            token_pos['end'] > section_start):
                        attribute_tokens[attr].append(token_pos['token_data'])

        return attribute_tokens

    def calculate_reasoning_confidence(self,
                                       reasoning_text: str,
                                       token_logprobs: List[Dict]) -> Dict[str, float]:
        """
        Calculates confidence metrics based on reasoning text and corresponding token logprobs

        Returns:
            dict: Contains multiple confidence metrics
        """
        if not token_logprobs:
            return {
                'avg_logprob': 0.0,
                'min_logprob': 0.0,
                'perplexity': float('inf'),
                'certainty_score': 0.0,
                'token_count': 0
            }

        # Extract logprob values
        logprobs = [t.get('logprob', 0.0) for t in token_logprobs
                    if t.get('logprob') is not None]

        if not logprobs:
            return {
                'avg_logprob': 0.0,
                'min_logprob': 0.0,
                'perplexity': float('inf'),
                'certainty_score': 0.0,
                'token_count': 0
            }

        # Basic statistics
        avg_logprob = np.mean(logprobs)
        min_logprob = np.min(logprobs)
        perplexity = math.exp(-avg_logprob) if avg_logprob < 0 else 1.0

        # Semantic certainty score (based on keywords)
        certainty_score = self._calculate_certainty_score(reasoning_text)

        # Composite confidence (combining logprob and semantic analysis)
        # logprob closer to 0 is better, perplexity smaller is better
        logprob_confidence = math.exp(avg_logprob)  # Convert to probability

        return {
            'avg_logprob': float(avg_logprob),
            'min_logprob': float(min_logprob),
            'max_logprob': float(np.max(logprobs)),
            'std_logprob': float(np.std(logprobs)),
            'perplexity': float(perplexity),
            'certainty_score': certainty_score,
            'logprob_confidence': float(logprob_confidence),
            'token_count': len(logprobs),
            'low_confidence_token_ratio': sum(1 for lp in logprobs if lp < -5) / len(logprobs)
        }

    def _calculate_certainty_score(self, text: str) -> float:
        """
        Calculates semantic certainty of reasoning based on keywords

        Returns:
            float: 0.0 - 1.0
        """
        text_lower = text.lower()

        # Count occurrences of different certainty keywords
        high_count = sum(1 for word in self.key_token_patterns['certainty_high']
                         if word in text_lower)
        medium_count = sum(1 for word in self.key_token_patterns['certainty_medium']
                           if word in text_lower)
        low_count = sum(1 for word in self.key_token_patterns['certainty_low']
                        if word in text_lower)
        null_count = sum(1 for word in self.key_token_patterns['certainty_null']
                         if word in text_lower)

        # Calculate weighted score
        if null_count > 0:
            return 0.0  # Explicitly stated not found

        total_keywords = high_count + medium_count + low_count
        if total_keywords == 0:
            return 0.5  # No clear keywords, medium confidence

        # Weighting: high=1.0, medium=0.6, low=0.3
        weighted_score = (high_count * 1.0 + medium_count * 0.6 + low_count * 0.3) / total_keywords

        return weighted_score

    def calculate_composite_confidence(self,
                                       reasoning_confidence: Dict[str, float],
                                       claimed_confidence: float,
                                       value: Optional[str]) -> Dict[str, float]:
        """
        Calculates composite confidence, combining multiple signals

        Returns:
            dict: Contains final confidence and individual components
        """
        # If value is null, confidence should be very low
        # if value is None:
        #     return {
        #         'final_confidence': 0.0,
        #         'model_claimed': claimed_confidence,
        #         'reasoning_based': 0.0,
        #         'logprob_based': 0.0,
        #         'confidence_agreement': 0.0
        #     }

        # Extract confidence from reasoning analysis
        logprob_conf = reasoning_confidence.get('logprob_confidence', 0.5)
        certainty_score = reasoning_confidence.get('certainty_score', 0.5)
        perplexity = reasoning_confidence.get('perplexity', 10.0)

        # Convert Perplexity to confidence (lower perplexity is better)
        perplexity_conf = 1.0 / (1.0 + math.log(perplexity + 1))

        # Composite reasoning confidence
        reasoning_based_conf = (logprob_conf * 0.4 +
                                certainty_score * 0.4 +
                                perplexity_conf * 0.2)

        # Check consistency between model-claimed confidence and actual analysis
        confidence_diff = abs(claimed_confidence - reasoning_based_conf)
        confidence_agreement = 1.0 - min(confidence_diff, 1.0)

        # Final confidence: comprehensively consider all factors
        # If model-claimed confidence matches analysis, take average
        # If inconsistent, trust the logprob-based analysis more
        if confidence_agreement > 0.7:
            final_conf = (claimed_confidence * 0.4 + reasoning_based_conf * 0.6)
        else:
            # When inconsistent, lower the final confidence
            final_conf = reasoning_based_conf * 0.7

        return {
            'final_confidence': float(final_conf),
            'model_claimed': float(claimed_confidence),
            'reasoning_based': float(reasoning_based_conf),
            'logprob_based': float(logprob_conf),
            'certainty_score': float(certainty_score),
            'perplexity_score': float(perplexity_conf),
            'confidence_agreement': float(confidence_agreement),
            'agreement_penalty': float(1.0 - confidence_agreement) * 0.3
        }

    def analyze_full_output(self,
                            output_text: str,
                            token_logprobs: List[Dict]) -> Dict[str, Dict]:
        """
        Fully analyzes an output, returning detailed confidence analysis for each attribute

        Returns:
            dict: {
                'brand': {...},
                'model': {...},
                ...
            }
        """
        # 1. Extract sections
        reasoning_sections = self.extract_reasoning_sections(output_text)
        results = self.extract_results_sections(output_text)
        attribute_tokens = self.map_tokens_to_reasoning(output_text, token_logprobs)

        # 2. Analyze each attribute
        analysis_results = {}

        for attr in self.attributes:
            reasoning_text = reasoning_sections.get(attr, "")
            tokens = attribute_tokens.get(attr, [])
            result_data = results.get(attr, {'value': None, 'claimed_confidence': 0.0})

            # Calculate confidence for the reasoning section
            reasoning_conf = self.calculate_reasoning_confidence(reasoning_text, tokens)

            # Calculate composite confidence
            composite_conf = self.calculate_composite_confidence(
                reasoning_conf,
                result_data['claimed_confidence'],
                result_data['value']
            )

            analysis_results[attr] = {
                'extracted_value': result_data['value'],
                'claimed_confidence': result_data['claimed_confidence'],
                'reasoning_text': reasoning_text,  # Save first 200 chars
                'reasoning_confidence': reasoning_conf,
                'composite_confidence': composite_conf,
                'token_count': len(tokens),
                # Final recommended confidence
                'recommended_confidence': composite_conf['final_confidence']
            }

        return analysis_results

    def generate_report(self, analysis_results: Dict[str, Dict]) -> str:
        """
        Generates a readable analysis report
        """
        report = []
        report.append("=" * 80)
        report.append("Attribute-Level Confidence Analysis Report")
        report.append("=" * 80)
        report.append("")

        for attr in self.attributes:
            if attr not in analysis_results:
                continue

            data = analysis_results[attr]
            report.append(f"【{attr.upper()}】")
            report.append(f"  Extracted Value: {data['extracted_value']}")
            report.append(f"  Model Claimed Confidence: {data['claimed_confidence']:.3f}")
            report.append(f"  Recommended Confidence: {data['recommended_confidence']:.3f}")

            comp_conf = data['composite_confidence']
            report.append(f"  Confidence Decomposition:")
            report.append(f"    - Logprob-based: {comp_conf['logprob_based']:.3f}")
            report.append(f"    - Semantic Analysis-based: {comp_conf['certainty_score']:.3f}")
            report.append(f"    - Confidence Agreement: {comp_conf['confidence_agreement']:.3f}")

            reasoning_conf = data['reasoning_confidence']
            report.append(f"  Reasoning Quality:")
            report.append(f"    - Avg Logprob: {reasoning_conf['avg_logprob']:.4f}")
            report.append(f"    - Perplexity: {reasoning_conf['perplexity']:.2f}")
            report.append(f"    - Token Count: {reasoning_conf['token_count']}")

            # Confidence Judgment
            rec_conf = data['recommended_confidence']
            if rec_conf > 0.8:
                confidence_label = "✓ High Confidence"
            elif rec_conf > 0.5:
                confidence_label = "⚠ Medium Confidence"
            else:
                confidence_label = "✗ Low Confidence"
            report.append(f"  Assessment: {confidence_label}")
            report.append("")

        return "\n".join(report)


# ==================== Integration into Processing Workflow ====================

def process_data_with_confidence_analysis(data_dict: dict,
                                          result_path: str,
                                          vllm_model,  # VLLMModel instance
                                          prompt_template: str):
    """
    Processes data and performs confidence analysis
    """
    analyzer = AttributeConfidenceAnalyzer()
    count = 0

    for brand, samples in data_dict.items():
        print(f"\nProcessing Brand: {brand}")

        for idx, data in enumerate(samples):
            try:
                banner = data.get("banner", "").strip()
                if not banner:
                    continue

                # Format prompt
                from your_script import format_prompt  # Import your format_prompt function
                prompt_text = format_prompt(prompt_template, banner)

                # Call model generation
                print(f"\n=== Processing Sample {idx + 1} ===")
                output_text, token_logprobs = vllm_model.generate(
                    prompt_text,
                    stop=["</results>", "</RESULTS>"]
                )

                # Perform confidence analysis
                analysis_results = analyzer.analyze_full_output(
                    output_text,
                    token_logprobs
                )

                # Generate report
                report = analyzer.generate_report(analysis_results)
                print(report)

                # Save results
                data["init_output"] = output_text
                data["token_logprobs"] = token_logprobs[:100]
                data["confidence_analysis"] = analysis_results

                # Write to file
                with open(result_path, 'a+', encoding='utf-8') as f:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')

                count += 1

            except Exception as e:
                print(f"Processing Error: {e}")
                continue

        print(f"\nProcessing completed, processed {count} samples")


# ==================== Independent Analysis of Existing Results ====================

def analyze_existing_results(result_file: str, output_file: str = None):
    """
    Analyzes existing result files and adds confidence analysis
    """
    analyzer = AttributeConfidenceAnalyzer()

    if output_file is None:
        output_file = result_file.replace('.json', '_analyzed.json')

    print(f"Analyzing file: {result_file}")
    print(f"Output to: {output_file}")

    with open(result_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    analyzed_count = 0

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for idx, line in enumerate(lines):
            try:
                data = json.loads(line)

                output_text = data.get('init_output', '')
                token_logprobs = data.get('token_logprobs', [])

                if output_text and token_logprobs:
                    # Perform analysis
                    analysis_results = analyzer.analyze_full_output(
                        output_text,
                        token_logprobs
                    )

                    # Add to data
                    data['confidence_analysis'] = analysis_results

                    # Print brief report
                    if idx < 5:  # Only print detailed report for the first 5
                        print(f"\nSample {idx + 1}:")
                        print(analyzer.generate_report(analysis_results))

                    analyzed_count += 1

                # Write result
                out_f.write(json.dumps(data, ensure_ascii=False) + '\n')

            except Exception as e:
                print(f"Error processing line {idx + 1}: {e}")
                out_f.write(line)  # Keep as is
                continue

    print(f"\nAnalysis completed! Analyzed {analyzed_count} samples")
    print(f"Results saved to: {output_file}")


# ==================== Usage Example ====================

if __name__ == "__main__":
    # Example 1: Analyze a single output
    example_output = """
<analysis>
<brand_reasoning>
The banner explicitly states "TP-Link" at the beginning. This is a well-known brand. Confidence is very high.
</brand_reasoning>
<model_reasoning>
"WR841N" appears after the brand. This is a standard model format. Confidence is high.
</model_reasoning>
</analysis>
<results>
```json
{
  "brand": {"value": "TP-Link", "confidence": 0.98},
  "model": {"value": "WR841N", "confidence": 0.95}
}
