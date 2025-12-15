import os
import json
import time
import logging
import re
from tqdm import tqdm
from typing import Dict, Any, Tuple

# Assuming your prompt file is in the same directory
from Prompt_templete_1012 import * # Import all prompts
from tools.logprob_analysis import *
from tools.tools_for_data_processing import *

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(
    filename='inference_process_vllm.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# -----------------------------
# Load JSON Data
# -----------------------------
def load_json_file(file_path: str) -> dict:

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.endswith('jsonl'):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return {"jsonl": data}
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# -----------------------------
# Concatenate/Generate Model Input Prompt
# -----------------------------
def format_prompt(template: str, banner: str, label: list = None) -> str:
    """Format prompt template"""
    prompt = template.replace("{{BANNER}}", banner)

    # Replace label information if available (optional)
    if label:
        brand, model, version = label
        prompt = prompt.replace("{{BRAND}}", str(brand))
        prompt = prompt.replace("{{MODEL}}", str(model))
        prompt = prompt.replace("{{VERSION}}", str(version))

    return prompt


# -----------------------------
# Parse LLM Output - Optimized Version
# -----------------------------
def extract_analysis_block(response_text: str):
    """
    Extract <analysis> block from model output and handle common exceptions.
    """

    analysis = ""

    # 1. Standard format: <analysis>...</analysis>
    match = re.search(r"<analysis>(.*?)</analysis>", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        analysis = match.group(1).strip()
        return analysis

    # 2. Compatible format: ```analysis ... </analysis>
    match = re.search(r"```analysis(.*?)</analysis>", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        analysis = match.group(1).strip()
        # result["warning"] = "Found non-standard start marker ```analysis, automatically corrected."
        return analysis

    # 3. Compatible format: <analysis> ... ``` (Model missed the closing tag)
    match = re.search(r"<analysis>(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        analysis = match.group(1).strip()
        # result["warning"] = "Missing </analysis> closing tag, attempted automatic truncation."
        return analysis

    # 4. Compatible format: ```analysis ... ```
    match = re.search(r"```analysis(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        analysis = match.group(1).strip()
        # result["warning"] = "Used markdown format ```analysis```, automatically extracted."
        return analysis

    # 5. Fallback processing: If there are obvious analysis paragraph clues (e.g., containing reasoning, brand, model, etc.)
    fallback_match = re.search(
        r"(brand_reasoning|model_reasoning|firmware_version_reasoning).*?(?=```|{|$)",
        response_text,
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

def fix_json_string(s: str):
    """
    Try to fix common JSON format errors:
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


def extract_json_result(response_text: str):
    """
    Extract JSON result block from model output, compatible with various non-standard formats:
    - ```json ... ```
    - <results> ... </results>
    - Multiple JSONs, automatically take the last valid one
    - With extra explanatory text or comments
    """

    result = ""

    # ✅ 1. Prioritize matching ```json ... ```
    json_blocks = re.findall(r"```json(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
    if json_blocks:
        # Take the last one (usually the final result)
        last_block = json_blocks[-1].strip()
        try:
            result = json.loads(last_block)
            return result
        except json.JSONDecodeError:
            # result["warning"] = "Found ```json``` block, but parsing failed. Attempting to fix format."
            fixed = fix_json_string(last_block)
            try:
                result = json.loads(fixed)
                # result["warning"] += " Automatically fixed."
                return result
            except Exception:
                pass  # Continue to other modes

    # ✅ 2. Match <results> ... </results>
    match = re.search(r"<results>(.*?)</results>", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        try:
            result = json.loads(content)
            return result
        except json.JSONDecodeError:
            # result["warning"] = "Detected <results> block, but JSON parsing failed. Attempting to fix."
            fixed = fix_json_string(content)
            try:
                result = json.loads(fixed)
                # result["warning"] += " Automatically fixed."
                return result
            except Exception:
                pass

    # ✅ 3. Extract the first JSON block directly from the text (most common fallback case)
    match = re.search(r"(\{.*\})", response_text, re.DOTALL)
    if match:
        raw_json = match.group(1)
        try:
            result = json.loads(raw_json)
            return result
        except json.JSONDecodeError:
            # result["warning"] = "Detected raw JSON block but parsing failed. Attempting to fix."
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


def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """
    Parse LLM response, extract analysis and results
    """
    if not response_text:
        return {"error": "Empty response"}

    # Initialize return result
    result = {
        "analysis": "",
        "parsed_json": None,
        "raw_response": response_text
    }

    # 1. Extract <analysis> content
    # analysis_match = re.search(
    #     r"<analysis>(.*?)</analysis>",
    #     response_text,
    #     re.DOTALL | re.IGNORECASE
    # )
    analysis_match = extract_analysis_block(response_text)

    if analysis_match:
        result["analysis"] = analysis_match
    else:
        result["warning"] = "No <analysis> block found"

    # 2. Extract <results> content
    result["parsed_json"] = extract_json_result(response_text)
    # results_match = re.search(
    #     r"<results>(.*?)</results>",
    #     response_text,
    #     re.DOTALL | re.IGNORECASE
    # )
    #
    # if not results_match:
    #     return {
    #         "error": "No <results> block found",
    #         "analysis": result["analysis"],
    #         "raw_response": response_text[:500]  # Only save first 500 characters
    #     }
    #
    # results_content = results_match.group(1).strip()
    #
    # # 3. Extract JSON (from ```json or direct JSON)
    # json_match = re.search(
    #     r"```json\s*(.*?)\s*```",
    #     results_content,
    #     re.DOTALL | re.IGNORECASE
    # )

    # json_text = json_match.group(1).strip() if json_match else results_content

    # 4. Parse JSON
    # try:
    #     # Find first { and last }
    #     start_idx = json_text.find('{')
    #     end_idx = json_text.rfind('}')
    #
    #     if start_idx != -1 and end_idx > start_idx:
    #         clean_json = json_text[start_idx:end_idx + 1]
    #         parsed_json = json.loads(clean_json)
    #         result["parsed_json"] = parsed_json
    #         result["status"] = "success"
    #     else:
    #         raise json.JSONDecodeError("No valid JSON object found", "", 0)
    #
    # except json.JSONDecodeError as e:
    #     result["error"] = f"JSON parsing failed: {str(e)}"
    #     result["json_text"] = json_text[:300]  # Save first 300 characters for debugging

    return result


# -----------------------------
# vLLM Model Class - Optimized Version
# -----------------------------
from vllm import LLM, SamplingParams


class VLLMModel:
    """Encapsulate vLLM model, support loading once and calling multiple times"""

    def __init__(self, model_path: str, gpu_ids: str = "0"):
        """
        Initialize and load model
        Args:
            model_path: Path to the model
            gpu_ids: GPU IDs, e.g., "0" or "0,1"
        """
        # Set visible GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

        print(f"Loading model to GPU {gpu_ids}: {model_path}")

        # Determine whether to use tensor parallelism based on GPU count
        gpu_count = len(gpu_ids.split(','))
        tensor_parallel = gpu_count if gpu_count > 1 else 1

        self.llm = LLM(
            model=model_path,
            dtype="float16",
            gpu_memory_utilization=0.4,
            tensor_parallel_size=tensor_parallel,
            trust_remote_code=True,  # Required by some models
            # template="qwen"
        )
        print("Model loaded successfully!")

    def generate(self,
                 prompt: str,
                 temperature: float = 0.0,
                 max_tokens: int = 4096,
                 top_k: int = 5,
                 stop: List[str] = None) -> Tuple[str, list]:
        """
        Call model to generate results

        Args:
            prompt: Input prompt
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            top_k: top_k sampling
            stop: List of stop words

        Returns:
            (generated text, token_logprobs list)
        """
        # Set stop words
        if stop is None:
            stop = ["</results>", "</RESULTS>"]

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            stop=stop,
            logprobs=1,
            skip_special_tokens=True  # Skip special tokens
        )

        try:
            # Generate results
            outputs = self.llm.generate([prompt], sampling_params=sampling_params)

            # Check return results
            if not outputs or len(outputs) == 0:
                print("Warning: Model returned no output")
                return "", []

            # Get first output
            output = outputs[0]

            # Get generated text
            if hasattr(output, 'outputs') and len(output.outputs) > 0:
                text = output.outputs[0].text

                # Ensure </results> end tag is included
                if "</results>" not in text.lower():
                    text += "\n</results>"
            else:
                print("Warning: Unable to get output text")
                text = ""

            # Get token logprobs
            token_logprobs = []
            try:
                if hasattr(output, 'outputs') and len(output.outputs) > 0:
                    output_data = output.outputs[0]

                    if hasattr(output_data, 'logprobs') and output_data.logprobs:
                        for token_logprob_dict in output_data.logprobs:
                            if token_logprob_dict:
                                for token_id, logprob_obj in token_logprob_dict.items():
                                    token_logprobs.append({
                                        "token_id": token_id,
                                        "logprob": logprob_obj.logprob if hasattr(logprob_obj, 'logprob') else None,
                                        "decoded_token": logprob_obj.decoded_token if hasattr(logprob_obj,
                                                                                              'decoded_token') else ""
                                    })
                                    break
            except Exception as e:
                print(f"Error getting logprobs: {e}")

            # Output result preview
            preview_length = min(200, len(text))
            print(f"=========Generated Result Preview (First {preview_length} chars)=========")
            print(text[:preview_length])
            print(text)
            # if len(text) > preview_length:
            #     print(f"... ({len(text) - preview_length} more chars)")
            # print("=" * 50)

            return text, token_logprobs

        except Exception as e:
            print(f"Error during generation: {e}")
            logging.error(f"Error during generation: {e}")
            return "", []


# -----------------------------
# Data Processing - Optimized Version
# -----------------------------
def process_item_data(brand, data: dict, idx, vllm_model: VLLMModel,
                 prompt_template: str,
                 max_retries: int = 2):
    """
    Process single item data
    Args:
        data_dict:

    Returns:

    """
    index = data.get("index")
    banner = data.get("banner", "").strip()
    if not banner:
        print(f"Warning: Banner for sample {index} is empty, skipping")
        return None
    # if index != "0303NV_noB_6429@3@NONE":
    #     flag = True
    # if not flag:
    #     continue

    # Preprocess banner
    banner = sanitize_network_info(banner, brand)
    if len(banner) > 30000 or len(banner) < 25:
        print(f"banner too long or too short: {len(banner)}, index: {index}")
        return None

    label = data.get("new_label", ["", "", ""])

    # Format prompt
    prompt_text = format_prompt(prompt_template, banner, label)

    # Call model to generate
    print(f"\n=== Processing Sample {index} ===")
    output_text, token_logprobs = vllm_model.generate(
        prompt_text,
        stop=["</results>", "</RESULTS>", "\n\n\n\n"]  # Multiple stop conditions
    )

    # Parse output
    parsed_output = parse_llm_response(output_text)

    return output_text, token_logprobs, parsed_output


def process_data(data_dict: dict,
                 result_path: str,
                 vllm_model: VLLMModel,
                 prompt_template: str,
                 max_retries: int = 2):
    """
    Process data and save results

    Args:
        data_dict: Input data dictionary
        result_path: Result save path
        vllm_model: vLLM model instance
        prompt_template: prompt template
        max_retries: Maximum retries when parsing fails
    """
    count = 0
    success_count = 0
    fail_count = 0
    flag = False

    for brand, samples in data_dict.items():
        print(f"\nProcessing Brand: {brand}")

        for idx, data in enumerate(tqdm(samples, desc="Processing Samples")):
            try:

                output_text, parsed_output, token_logprobs = process_item_data(brand, data, idx, vllm_model, prompt_template, max_retries)
                # Save to data object
                data["init_output"] = output_text
                data["parsed_output"] = parsed_output
                data["token_logprobs"] = token_logprobs  # Only save logprobs for the first 100 tokens (implied context)

                # Check if parsing was successful
                if "error" in parsed_output:
                    fail_count += 1
                    print(f"❌ Parsing Failed: {parsed_output.get('error', 'Unknown error')}")
                    logging.warning(f"Parsing Failed - Banner: {data['banner'][:50]}... - Error: {parsed_output.get('error')}")
                else:
                    success_count += 1
                    print("✓ Parsing Successful")

                # Write to file (write after processing each one to avoid loss)
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                with open(result_path, 'a+', encoding='utf-8') as f:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')

                count += 1
                time.sleep(0.1)  # Short delay

            except Exception as e:
                fail_count += 1
                logging.error(f"Error processing data: {e}")
                print(f"❌ Processing Error: {e}")
                continue

    # Print statistics
    print("\n" + "=" * 60)
    print(f"Processing Completion Statistics:")
    print(f"  Total Samples: {count}")
    print(f"  Success: {success_count} ({success_count / count * 100:.1f}%)")
    print(f"  Failed: {fail_count} ({fail_count / count * 100:.1f}%)")
    print("=" * 60)


# -----------------------------
# Main Function
# -----------------------------
def main_init(data_file: str, prompt_name: str, model_name: str, gpu_ids: str = "0"):
    # Data loading
    print("=============Data Loading==============")
    data_dict = load_json_file(data_file)
    print(f"Loaded {sum(len(v) for v in data_dict.values())} samples")

    # Select prompt
    print("\n=============Select Prompt==============")
    try:
        prompt_template = globals()[prompt_name]
        print(f"Using prompt: {prompt_name}")
        print(f"Prompt length: {len(prompt_template)} characters")
    except KeyError:
        raise ValueError(f"Prompt name {prompt_name} not found in imported prompt file")

    # Construct model path
    model_path = os.path.join("/data/zhangfengshi/model", model_name)

    # Pre-load model (load only once)
    print("\n=============Loading Model==============")
    vllm_model = VLLMModel(model_path, gpu_ids=gpu_ids)

    # Automatically generate output file path
    data_filename = os.path.basename(data_file)
    data_base, _ = os.path.splitext(data_filename)
    output_path = os.path.join(
        os.path.dirname(data_file),
        f"{data_base}_{model_name}_{prompt_name}_1111.json"
    )

    # # Backup if output file exists
    # if os.path.exists(output_path):
    #     backup_path = output_path + f".backup_{int(time.time())}"
    #     os.rename(output_path, backup_path)
    #     print(f"Existing file backed up to: {backup_path}")

    # Process data
    print("\n=============Start Processing Data==============")
    process_data(data_dict, output_path, vllm_model, prompt_template)

    print(f"\nAll data processing completed, results saved in: {output_path}")


def main(data_file: str,
         prompt_name: str,
         model_name: str,
         gpu_ids: str = "0",
         lora_path: str = None,
         use_merged: bool = False):
    """
    Main function
    Args:
        data_file: Data file path
        prompt_name: Prompt name
        model_name: Model name
        gpu_ids: GPU IDs
        lora_path: LoRA adapter path (optional)
        use_merged: Whether to use merged model
    """
    # Data loading
    print("=============Data Loading==============")
    data_dict = load_json_file(data_file)
    print(f"Loaded {sum(len(v) for v in data_dict.values())} samples")

    # Select prompt
    print("\n=============Select Prompt==============")
    try:
        prompt_template = globals()[prompt_name]
        print(f"Using prompt: {prompt_name}")
        print(f"Prompt length: {len(prompt_template)} characters")
    except KeyError:
        raise ValueError(f"Prompt name {prompt_name} not found in imported prompt file")

    # Construct model path
    if use_merged:
        # Use merged model
        model_path = os.path.join("/data/zhangfengshi/model/finetuned", model_name + "_merged")
        print(f"Using merged model: {model_path}")
        vllm_model = VLLMModel(model_path, gpu_ids=gpu_ids)
    else:
        # Use base model + LoRA adapter
        base_model_path = os.path.join("/data/zhangfengshi/model", model_name)

        if lora_path:
            print(f"Using base model: {base_model_path}")
            print(f"Using LoRA adapter: {lora_path}")
            vllm_model = VLLMModel(
                base_model_path,
                gpu_ids=gpu_ids,
                lora_path=lora_path,
                enable_lora=True,
                max_lora_rank=8  # According to your training config: --lora_rank 8
            )
        else:
            print(f"Using base model (no LoRA): {base_model_path}")
            vllm_model = VLLMModel(base_model_path, gpu_ids=gpu_ids)

    # Automatically generate output file path
    data_filename = os.path.basename(data_file)
    data_base, _ = os.path.splitext(data_filename)

    # Adjust output filename based on LoRA usage
    if lora_path or use_merged:
        model_suffix = model_name + "_finetuned"
    else:
        model_suffix = model_name

    output_path = os.path.join(
        os.path.dirname(data_file),
        f"{data_base}_{model_suffix}_{prompt_name}_1114.json"
    )

    # Process data
    print("\n=============Start Processing Data==============")
    process_data(data_dict, output_path, vllm_model, prompt_template)

    print(f"\nAll data processing completed, results saved in: {output_path}")

# -----------------------------
# Script Entry Point
# -----------------------------
if __name__ == "__main__":
    import sys

    # Receive arguments from command line
    data_file = sys.argv[1] if len(sys.argv) > 1 else "./data/test_set.jsonl"
    prompt_name = sys.argv[2] if len(sys.argv) > 2 else "simplest_prompt_for_test"#confidence_detail_analysis_1105"
    model_name = sys.argv[3] if len(sys.argv) > 3 else "DeepSeek-R1-Distill-Qwen-14B" # sft_qwen2___5_7B
    gpu_ids = sys.argv[4] if len(sys.argv) > 4 else "0"
    lora_path = sys.argv[5] if len(sys.argv) > 5 else None
    use_merged = sys.argv[6].lower() == 'true' if len(sys.argv) > 6 else False

    print(f"Startup Arguments:")
    print(f"  Data File: {data_file}")
    print(f"  Prompt: {prompt_name}")
    print(f"  Model: {model_name}")
    print(f"  GPU: {gpu_ids}")
    if lora_path:
        print(f"  LoRA Path: {lora_path}")
    if use_merged:
        print(f"  Use Merged Model: True")
    print()

    main(data_file, prompt_name, model_name, gpu_ids, lora_path, use_merged)
