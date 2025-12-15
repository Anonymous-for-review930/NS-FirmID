#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :check_sample.py
# @Time      :2025/11/19 18:02
# @Author    :zfs
import json

from tools.tool_global import extract_ground_truth
from tools.wrong_sample_record import *
import pandas as pd
import numpy as np


def check_sft_sample(sample1_path, sample2_path):
    """
    检查两个sft样本是否一致
    Args:
        sample1_path:
        sample2_path:

    Returns:

    """
    with open(sample1_path, 'r', encoding='utf-8') as f1, open(sample2_path, 'r', encoding='utf-8') as f2:
        index1_list = []
        for line in f1:
            json_line = json.loads(line)
            index = json_line.get('index', json_line.get('sample_id'))
            index1_list.append(index)
        index2_list = []
        for line in f2:
            json_line = json.loads(line)
            index = json_line.get('index', json_line.get('sample_id'))
            index2_list.append(index)
        print(len(index1_list), len(index2_list))
        for index in index2_list:
            if index not in index1_list:
                print(index)

def add_sft_results(sft_result1, sft_result2):
    with open(sft_result1, 'a+', encoding='utf-8') as f1:
        with open(sft_result2, 'r', encoding='utf-8') as f2:
            for line in f2:
                json_line = json.loads(line)
                f1.write(json.dumps(json_line, ensure_ascii=False) + '\n')



def renew_label(sample_path, out_path, index_path):
    """
    更新数据标签。之前将os版本作为固件标注了，不够严谨
    Returns:
    生成新的标签文件以及另存的index-标签json用于后续评估
    """
    index_dict = {}
    new_data = {}
    with open(sample_path, 'r', encoding='utf-8') as f1:
        data = json.load(f1)
        for brand, samples in data.items():
            for sample in samples:
                index = sample.get('index', sample.get('sample_id'))
                new_label = sample.get('new_label')
                if index == "0721_12620@1":
                    pass
                try:
                    os_component = sample.get('parsed_output').get('new_label').get('component_versions').get('os')
                except AttributeError:
                    os_component = None
                try:
                    if new_label[2].lower() == os_component.lower():
                        new_label[2] = None
                except:
                    pass
                new_label[0], new_label[1], new_label[2] = recorrect_relabel_sample(index, new_label[0], new_label[1], new_label[2])
                sample['new_label'] = new_label
                index_dict[index] = new_label
                if brand not in new_data.keys():
                    new_data[brand] = []
                new_data[brand].append(sample)
    with open(out_path, 'w', encoding='utf-8') as f2, open(index_path, 'w', encoding='utf-8') as f3:
        json.dump(new_data, f2, ensure_ascii=False, indent=4)
        json.dump(index_dict, f3, ensure_ascii=False, indent=4)


def add_right_sample_to_sft():
    with open('../llm_results/error_samples_for_teacher_input_1120_deepseek_api_huoshan_v3_teacher_labeled.json', 'r', encoding='utf-8') as f0:
        index_done_list = []
        for line in f0:
            json_line = json.loads(line)
            index = json_line.get('index', json_line.get('sample_id'))
            index_done_list.append(index)
    with open('../llm_results/sft_teacher_input_combine_1119_deepseek_api_huoshan_v3_teacher_labeled.json', 'r', encoding='utf-8') as f, \
        open('../llm_results/error_samples_for_teacher_input_1120_deepseek_api_huoshan_v3_teacher_labeled.json', 'a+', encoding='utf-8') as f2, \
        open('../data/relabel_merged_cydar_index_label_dict.json', 'r', encoding='utf-8') as f3:
        index_dict = json.load(f3)
        for line in f:
            json_line = json.loads(line)
            index = json_line.get('index', json_line.get('sample_id'))
            if index in index_done_list:
                continue
            gt = json_line.get('gt', json_line.get('ground_truth_json')).replace("null", '""')
            gt = json.loads(gt)

            if index in index_dict.keys():
                gt_new = {"brand": index_dict[index][0], "model": index_dict[index][1], "firmware_version": index_dict[index][2]}

                if gt == gt_new:
                    f2.write(json.dumps(json_line, ensure_ascii=False) + '\n')


def check_label_inconsistent():
    """
    检查为微调样本的标签问题
    Returns:

    """
    sft_sample_path = "../llm_results/error_samples_for_teacher_input_1120_deepseek_api_huoshan_v3_teacher_labeled.json"
    index_label_path = "../data/relabel_merged_cydar_index_label_dict.json"
    with open(index_label_path, 'r', encoding='utf-8') as f3:
        index_dict = json.load(f3)
    with open(sft_sample_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_line = json.loads(line)
            index = json_line.get('sample_id')
            if index in index_dict.keys():
                gt = json_line.get('gt', json_line.get('ground_truth_json')).replace("null", '""')
                gt = json.loads(gt)
                gt_new = {"brand": index_dict[index][0], "model": index_dict[index][1], "firmware_version": index_dict[index][2] if index_dict[index][2] != None else ""}
                if gt != gt_new:
                    print(f"{gt}, {gt_new}, {index}")


def check_overlap_sample():

    # 1. 读取id列表
    with open(
            r'F:\paper\Multi-Agent_version_identification_using_LLMs\sft_sample\sft_dataset_CORRECT_only_1124_id.json',
            'r', encoding='utf-8') as f:
        id_list = json.load(f)
        id_set = set(id_list)

    # 2. 读取测试集样本
    test_samples = []
    with open(
            r'F:\paper\Multi-Agent_version_identification_using_LLMs\llm_results\hold-out_dataset\holdout_set_Qwen2___5-7B-Instruct_confidence_detail_analysis_1105.jsonl',
            'r', encoding='utf-8') as f:
        for line in f:
            json_line = json.loads(line)
            test_samples.append(json_line)
    print(len(test_samples))
    # 3. 统计index在id_set中的样本数
    count = 0
    for sample in test_samples:
        index = sample.get('index', sample.get('sample_id'))
        if index in id_set:
            count += 1

    print(f'有 {count} 个样本的index在id列表中')


def filter_sft_sample(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            json_line = json.loads(line)
            index = json_line.get('index', json_line.get('sample_id'))
            ground_truth_json = json.loads(json_line['ground_truth_json'])
            banner = json_line['banner']
            brand = ground_truth_json.get('brand', '')
            if brand:
                if brand not in banner.lower():
                    continue
            if brand == "---":
                continue
            with open(output_file, 'a', encoding='utf-8') as fw:
                fw.write(json.dumps(json_line, ensure_ascii=False) + '\n')


def load_data(sample_path, sample_path_2='', sft=False):
    """
    这里生成模拟数据。
    在实际使用中，请替换为读取您的数据库或CSV/JSON代码。
    例如: df = pd.read_json('your_data.json')
    """
    data = []
    try:
        with open(sample_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                # 如果数据是以品牌为键的字典，展开成列表
                expanded_data = []
                for brand, samples in data.items():
                    for sample in samples:
                        index = sample.get('index', sample.get('sample_id'))
                        if sft:
                            if index not in sft_index:
                                continue
                        else:
                            if index in sft_index:
                                continue
                        expanded_data.append(sample)
                data = expanded_data
    except Exception as e:
        with open(sample_path, 'r', encoding='utf-8') as f:
            for line in f:
                json_line = json.loads(line)
                index = json_line.get('index', json_line.get('sample_id'))
                if sft:
                    if index not in sft_index:
                        continue
                else:
                    if index in sft_index:
                        continue
                data.append(json_line)
    if sample_path_2:
        try:
            with open(sample_path_2, 'r', encoding='utf-8') as f:
                data_2 = json.load(f)
                if isinstance(data_2, dict):
                    # 如果数据是以品牌为键的字典，展开成列表
                    expanded_data = []
                    for brand, samples in data_2.items():
                        for sample in samples:
                            if sft:
                                if index not in sft_index:
                                    continue
                            else:
                                if index in sft_index:
                                    continue
                            expanded_data.append(sample)
                    data2 = expanded_data
                data += data2
        except Exception as e:
            with open(sample_path_2, 'r', encoding='utf-8') as f:
                for line in f:
                    json_line = json.loads(line)
                    index = json_line.get('index', json_line.get('sample_id'))
                    if sft:
                        if index not in sft_index:
                            continue
                    else:
                        if index in sft_index:
                            continue
                    data.append(json_line)


    dup_count = 0
    for i, json_line in enumerate(data):
        if json_line.get('index', json_line.get('sample_id')) in sft_index:
            dup_count += 1
        label_dict = extract_ground_truth(json_line)
        data[i]["new_label"] = [label_dict["brand"], label_dict["model"], label_dict["firmware_version"]]
    print("dup sample:", dup_count)
    return pd.DataFrame(data)


def process_data(df):
    # 确保 new_label 是列表类型
    # 如果读取的是字符串形式的列表 (e.g. "['a','b']"), 需要用 json.loads 或 ast.literal_eval 转换

    # 拆分 new_label 到单独的列
    # 假设结构固定为 [Brand, Model, Firmware]
    df[['Brand', 'Model', 'Firmware']] = pd.DataFrame(df['new_label'].tolist(), index=df.index)

    return df


# ==========================================
# 2. 统计逻辑
# ==========================================
def is_negative(val):
    """判断是否为负样本 (None 或 空字符串)"""
    if val is None:
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    return False


def get_stats(series, name):
    """计算单个字段的统计信息"""
    total = len(series)
    # 标记负样本
    neg_mask = series.apply(is_negative)
    neg_count = neg_mask.sum()
    pos_count = total - neg_count

    # 统计唯一值 (仅在正样本中统计)
    valid_series = series[~neg_mask]
    unique_count = valid_series.nunique()

    return {
        "Attribute": name,
        "Total_Samples": total,
        "Unique_Values": unique_count,
        "Positive_Samples": pos_count,
        "Negative_Samples": neg_count,
        "Negative_Rate": f"{(neg_count / total) * 100:.2f}%"
    }


# ==========================================
# 3. 主执行流程
# ==========================================
def count_sample_status(sample_path, output_file, sample_path_2=''):
    print("正在处理数据...")
    df = load_data(sample_path, sample_path_2, False)
    df = process_data(df)

    stats_list = []

    # 1. 统计 Protocol (假设 Protocol 没有所谓的负样本定义，或者空也算一种协议)
    # 如果 Protocol 也有空值作为负样本，逻辑同下
    proto_unique = df['protocol'].nunique()
    stats_list.append({
        "Attribute": "Protocol",
        "Total_Samples": len(df),
        "Unique_Values": proto_unique,
        "Positive_Samples": len(df),  # 暂定全部有效
        "Negative_Samples": 0,
        "Negative_Rate": "0.00%"
    })

    # 2. 统计 Brand, Model, Firmware
    target_cols = ['Brand', 'Model', 'Firmware']
    for col in target_cols:
        stats_list.append(get_stats(df[col], col))

    # 转换为 DataFrame
    stats_df = pd.DataFrame(stats_list)

    # 3. 导出到 Excel

    with pd.ExcelWriter(output_file) as writer:
        # Sheet 1: 总体概览
        stats_df.to_excel(writer, sheet_name='Summary', index=False)

        # Sheet 2: 协议分布详情
        proto_counts = df['protocol'].value_counts().reset_index()
        proto_counts.columns = ['Protocol', 'Count']
        proto_counts.to_excel(writer, sheet_name='Protocol_Distribution', index=False)

        # Sheet 3: 品牌分布详情 (Top 50)
        brand_counts = df[~df['Brand'].apply(is_negative)]['Brand'].value_counts().reset_index().head(50)
        brand_counts.columns = ['Brand', 'Count']
        brand_counts.to_excel(writer, sheet_name='Brand_Distribution', index=False)

    print(f"统计完成！结果已保存至: {output_file}")
    print("\n--- 概览 ---")
    print(stats_df.to_string())


if __name__ == "__main__":
    # check_sft_sample('../sft_sample/sft_teacher_input.jsonl', '../sft_sample/error_samples_for_teacher_input_1119_v2.jsonl')
    # add_sft_results('../llm_results/sft_teacher_input_deepseek_api_huoshan_v3_teacher_labeled.json', '../llm_results/test_dataset/sft_teacher_input_deepseek_api_huoshan_v3_teacher_labeled.json')
    # renew_label('../data/relabel_merged_cydar_data_into_test_1012.json', '../data/relabel_merged_cydar_data_into_test_1012_renew.json', '../data/relabel_merged_cydar_index_label_dict.json')
    # add_right_sample_to_sft()
    # check_label_inconsistent()
    # check_overlap_sample()
    # filter_sft_sample('../llm_results/error_samples_for_teacher_input_1124_deepseek_api_huoshan_v3_teacher_labeled.json',
    #                   '../llm_results/error_samples_for_teacher_input_1130_deepseek_api_huoshan_v3_teacher_labeled.json')

    # count_sample_status('../data/test_set.jsonl', '../data/test_set_status.xlsx')
    sft_index = []
    with open('../sft_sample/sft_dataset_CORRECT_only_1130_id.json', 'r', encoding='utf-8') as f:
        sft_index = json.load(f)

    # count_sample_status('../data/relabel_merged_cydar_data_into_test_1012_renew.json', '../sft_sample/sft_dataset_CORRECT_only_1130_status.xlsx', '../data/negative_sample_relabel_1108_notvalid_dict.json')
    count_sample_status('../data/test_set.jsonl',
                        '../data/test_holdout_status_no_sft.xlsx',
                        '../data/holdout_set.jsonl')



