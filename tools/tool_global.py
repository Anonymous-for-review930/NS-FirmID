import os, json
import re
from typing import List, Dict, Tuple
import numpy as np

from tools.wrong_sample_record import recorrect_relabel_sample


def get_file_paths(file_path):  # file_path是想要获取的文件夹的路径
    """
    返回文件夹下所有文件，返回值是输入文件夹路径+文件的组合路径。只适用于无子文件夹的情况，否则路径会出错
    """
    files_paths = list()
    for i, j, files_names in os.walk(file_path):
        for file_name in files_names:
            files_paths.append(os.path.join(file_path, file_name))
    return files_paths

def get_file_path_form_folder(file_path):
    """
    获取文件夹下的所有文件，返回文件夹路径+文件名的路径列表，适用于有子文件夹的情况
    """
    path_list = []
    for i, j, files_names in os.walk(file_path):
        for file_name in files_names:
            path_list.append(os.path.join(i, file_name))
    return path_list

def get_subdirectories(folder_path):
    """
    获取文件夹下的所有子文件夹，返回的是子文件夹名列表
    """
    subdirectories = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    return subdirectories

def sort_brand_model_versions(data: dict) -> dict:
    """"""
    # 嵌套字典排序 形如{brand1: {model1: [version], model2: [version]}, brand2: {model1: [version]}}
    sorted_data = {}
    for brand in sorted(data.keys(), key=lambda x: x.lower()):
        models = data[brand]
        sorted_models = {}
        for model in sorted(models.keys(), key=lambda x: x.lower()):
            # 版本列表排序（按字符串）
            sorted_versions = sorted(models[model], key=lambda v: v.lower())
            sorted_models[model] = sorted_versions
        sorted_data[brand] = sorted_models
    return sorted_data


def extract_and_merge_keyword_contexts(keywords: List[str], text: str,
                                       context_length: int = 100,
                                       merge_distance: int = 50) -> List[Dict]:
    """
    提取多个关键词的上下文，如果关键词离得很近则合并为一个文本段

    Args:
        keywords: 关键词列表
        text: 目标文本
        context_length: 每个关键词的上下文长度
        merge_distance: 合并距离阈值（两个关键词上下文重叠或距离小于此值时合并）

    Returns:
        List[Dict]: 合并后的文本段列表，每个包含:
        - text_segment: 合并后的文本段
        - found_keywords: 该文本段中找到的关键词列表
        - keyword_positions: 每个关键词在原文中的位置
        - segment_start: 文本段在原文中的起始位置
        - segment_end: 文本段在原文中的结束位置
        - context_length: 实际上下文长度
    """

    # 第一步：收集所有关键词的匹配位置
    all_matches = []

    for keyword in keywords:
        if keyword in ["", None, "np"]:
            continue
        pattern = re.escape(keyword)
        matches = re.finditer(pattern, text, re.IGNORECASE)

        for match in matches:
            all_matches.append({
                'keyword': keyword,
                'matched_text': match.group(),
                'start_pos': match.start(),
                'end_pos': match.end(),
                'context_start': max(0, match.start() - context_length),
                'context_end': min(len(text), match.end() + context_length)
            })

    if not all_matches:
        return []

    # 第二步：按位置排序
    all_matches.sort(key=lambda x: x['start_pos'])

    # 第三步：合并近距离的匹配
    merged_segments = []
    current_segment = {
        'keywords': [all_matches[0]],
        'start': all_matches[0]['context_start'],
        'end': all_matches[0]['context_end']
    }

    for match in all_matches[1:]:
        # 检查是否应该合并
        distance = match['context_start'] - current_segment['end']
        overlap = current_segment['end'] >= match['context_start']

        if overlap or distance <= merge_distance:
            # 合并：扩展当前段的范围，添加关键词
            current_segment['keywords'].append(match)
            current_segment['end'] = max(current_segment['end'], match['context_end'])
        else:
            # 不合并：保存当前段，开始新段
            merged_segments.append(current_segment)
            current_segment = {
                'keywords': [match],
                'start': match['context_start'],
                'end': match['context_end']
            }

    # 添加最后一个段
    merged_segments.append(current_segment)

    # 第四步：生成最终结果
    results = []
    for i, segment in enumerate(merged_segments):
        # 提取文本段
        text_segment = text[segment['start']:segment['end']]

        # 整理关键词信息
        found_keywords = []
        keyword_positions = {}

        for kw_match in segment['keywords']:
            keyword = kw_match['keyword']
            if keyword not in found_keywords:
                found_keywords.append(keyword)

            if keyword not in keyword_positions:
                keyword_positions[keyword] = []

            keyword_positions[keyword].append({
                'absolute_pos': kw_match['start_pos'],
                'relative_pos': kw_match['start_pos'] - segment['start'],
                'matched_text': kw_match['matched_text']
            })

        results.append({
            'segment_id': i + 1,
            'text_segment': text_segment,
            'found_keywords': found_keywords,
            'keyword_positions': keyword_positions,
            'segment_start': segment['start'],
            'segment_end': segment['end'],
            'context_length': segment['end'] - segment['start'],
            'keywords_count': len(segment['keywords'])
        })

    return results


def extract_segments_with_smart_boundaries(keywords: List[str], text: str,
                                           context_length: int = 100,
                                           merge_distance: int = 50,
                                           use_sentence_boundaries: bool = True) -> List[Dict]:
    """
    智能边界的文本段提取（尝试在句子边界处切分）

    Args:
        keywords: 关键词列表
        text: 目标文本
        context_length: 基础上下文长度
        merge_distance: 合并距离
        use_sentence_boundaries: 是否使用句子边界优化切分

    Returns:
        List[Dict]: 优化边界后的文本段
    """
    # 先使用基础方法获取段落
    base_segments = extract_and_merge_keyword_contexts(keywords, text, context_length, merge_distance)

    if not use_sentence_boundaries:
        return base_segments

    # 优化边界
    optimized_segments = []

    for segment in base_segments:
        start = segment['segment_start']
        end = segment['segment_end']

        # 寻找更好的起始边界（向前找句号、换行符等）
        better_start = start
        for i in range(start, max(0, start - 50), -1):
            if text[i] in '.!?\n\r':
                better_start = i + 1
                break

        # 寻找更好的结束边界（向后找句号、换行符等）
        better_end = end
        for i in range(end, min(len(text), end + 50)):
            if text[i] in '.!?\n\r':
                better_end = i + 1
                break

        # 更新文本段
        optimized_segment = segment.copy()
        optimized_segment['text_segment'] = text[better_start:better_end]
        optimized_segment['segment_start'] = better_start
        optimized_segment['segment_end'] = better_end
        optimized_segment['context_length'] = better_end - better_start
        optimized_segment['boundary_optimized'] = True

        # 更新关键词的相对位置
        for keyword, positions in optimized_segment['keyword_positions'].items():
            for pos_info in positions:
                pos_info['relative_pos'] = pos_info['absolute_pos'] - better_start

        optimized_segments.append(optimized_segment)

    return optimized_segments


def clean_keyword(keyword):
    """
    清理关键词，去除除了字母数字之外的符号
    :param keyword:
    :return:
    """
    keyword = ''.join([c if c.isprintable() else ' ' for c in keyword])
    keyword = re.sub(r"[-/ ._]", " ", keyword).lower()
    keyword = re.sub(r'\s+', ' ', keyword).strip()
    return keyword


def sort_by_confidence_desc(model_match):
    """按confidence从高到低排序，返回新列表"""
    if model_match == []:
        return []
    return sorted(model_match, key=lambda x: x.confidence, reverse=True)


def convert_to_serializable(obj):
    """
    递归地遍历对象，将所有非 JSON 标准类型（包括 NumPy 类型、bool 类型和 inf）
    转换为 JSON 兼容的类型。
    """
    # 1. 处理字典
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    # 2. 处理列表/元组
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(elem) for elem in obj]
    # 3. 处理布尔值 (Python 内置 bool 和 NumPy bool)
    elif isinstance(obj, (bool, np.bool_)):
        # 将 Python/NumPy 的 True/False 转换为小写的字符串 "true"/"false"
        return str(obj).lower()
    # 4. 处理 NumPy 浮点数和整数 (确保它们被识别为标准类型)
    # elif isinstance(obj, (np.float64, np.int_)):
    #     return obj.item() # 使用 .item() 将 NumPy 标量转换为标准的 Python float/int
    # 5. 处理无穷大 (inf) 和 NaN (NaN 不在您的示例中，但通常需要处理)
    elif isinstance(obj, float) and (np.isinf(obj) or np.isnan(obj)):
        # JSON 标准不支持 inf 或 NaN，将其转换为字符串或 null
        return None # 或 str(obj), 这里推荐转换为 null 或 None

    # 6. 其他类型（如字符串、标准数字、None）直接返回
    else:
        return obj


def extract_ground_truth(data: Dict):
    """从 'new_label' 列表 提取真实标签"""
    labels = {}
    index = data.get("index", data.get("sample_id"))
    new_label = data.get('new_label', data.get('label', [None, None, None]))
    # new_label = renew_label_dict[index] if index in renew_label_dict else new_label
    if "new_label" not in data.keys() and "label" not in data.keys():
        try:
            ground_truth = json.loads(data.get('ground_truth_json'))
            new_label = [ground_truth['brand'], ground_truth['model'], ground_truth['firmware_version']]
        except Exception as e:
            pass

    if not isinstance(new_label, list): new_label = [None, None, None]
    while len(new_label) < 3: new_label.append(None)

    labels['brand'] = new_label[0] if new_label[0] else None
    labels['model'] = new_label[1] if new_label[1] else None
    labels['firmware_version'] = new_label[2] if new_label[2] else None
    try:
    #     if 'linux' in labels['firmware_version'].lower() or 'os ' in labels['firmware_version'].lower() or (len(labels['firmware_version'].lower()) > 3 and labels['firmware_version'].lower().isalpha()):
    #         labels['firmware_version'] = None
        labels['brand'], labels['model'], labels['firmware_version'] = recorrect_relabel_sample(index, labels['brand'], labels['model'], labels['firmware_version'])

    except:
        pass
    ## 修正不在banner中的品牌
    banner = data.get('banner', "")
    if labels['brand']:
        if labels['brand'] not in banner.lower():
            labels['brand'] = None
    if labels['brand'] == "---":
        labels['brand'] = None
    if labels['model']:
        if labels['model'] not in banner.lower():
            labels['model'] = None
    return labels


def read_data_file(path)-> list[dict]:
    """读取数据文件，返回数据列表"""
    result = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    result += v
    except Exception as e:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                result.append(json.loads(line))
    return result


if __name__ == '__main__':
    fold_path = 'F:/paper/paper_gu/firmware_version_identification/data_analyse_module/analysis_data_restore/KEY_WORDS0'
    i = get_subdirectories(fold_path)
    j = get_file_paths(fold_path)
    k = get_file_path_form_folder(fold_path)
    print(' ')