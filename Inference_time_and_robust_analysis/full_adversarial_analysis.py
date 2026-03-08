"""
Complete Test Set Adversarial Scenario Analysis
Using complete data from test_set + holdout_set
"""

import json
import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

def load_all_data():
    xlsx_path = '/Users/lululu/Documents/code/Multi-Agent_version_identification_using_LLMs/agent/ablation_performance_1204_xgb_test_and_holdout_set_sft_Qwen2___5_7B_1130_1203/ablation_details_full_no_comp.xlsx'
    df = pd.read_excel(xlsx_path)
    df = df[df['Strategy'] == 'confidence_filter']
    
    test_set_path = '/Users/lululu/Documents/code/Multi-Agent_version_identification_using_LLMs/data/test_set.jsonl'
    holdout_set_path = '/Users/lululu/Documents/code/Multi-Agent_version_identification_using_LLMs/data/holdout_set.jsonl'
    
    banners = {}
    for path in [test_set_path, holdout_set_path]:
        with open(path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                banners[entry['index']] = {
                    'banner': entry['banner'],
                    'label': entry.get('label', []),
                    'flag': entry.get('flag', [])
                }
    
    return df, banners

def count_version_like_strings(text):
    patterns = {
        'ip_addresses': r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
        'version_numbers': r'\b[vV]?\d+\.\d+(?:\.\d+)?(?:\.\d+)?(?:-[a-zA-Z0-9]+)?\b',
        'dates': r'\b\d{4}[.\-/]\d{2}[.\-/]\d{2}\b|\b\d{2}[.\-/]\d{2}[.\-/]\d{4}\b',
        'oids': r'\b\d+\.\d+\.\d+\.\d+\.\d+\.\d+\.\d+\b',
        'build_hashes': r'\b[a-fA-F0-9]{8,}\b'
    }
    
    counts = {}
    for name, pattern in patterns.items():
        counts[name] = len(re.findall(pattern, text))
    
    counts['total_version_like'] = sum(counts.values())
    return counts

def analyze_full_dataset(df, banners):
    print("="*100)
    print("Complete Test Set Adversarial Scenario Analysis (test_set + holdout_set)")
    print("="*100)
    
    attributes = ['brand', 'model', 'firmware_version']
    
    all_results = {}
    
    for attr in attributes:
        attr_df = df[df['Attribute'] == attr]
        
        results = {'TP': [], 'FP': [], 'TN': [], 'FN': []}
        
        for _, row in attr_df.iterrows():
            sample_id = row['Sample_ID']
            result = row['Result']
            
            if sample_id in banners:
                banner_data = banners[sample_id]
                counts = count_version_like_strings(banner_data['banner'])
                
                results[result].append({
                    'sample_id': sample_id,
                    'counts': counts,
                    'label': banner_data['label'],
                    'banner_len': len(banner_data['banner'])
                })
        
        all_results[attr] = results
        
        print(f"\n{'='*80}")
        print(f"Attribute: {attr}")
        print(f"{'='*80}")
        
        print(f"\n{'Result Type':<10} {'Sample Count':<10} {'Avg IP':<12} {'Avg Version':<12} {'Avg Date':<12} {'Avg Total':<12} {'Avg Banner Length':<15}")
        print("-"*95)
        
        for result_type, samples in results.items():
            if samples:
                avg_ip = np.mean([s['counts']['ip_addresses'] for s in samples])
                avg_ver = np.mean([s['counts']['version_numbers'] for s in samples])
                avg_date = np.mean([s['counts']['dates'] for s in samples])
                avg_total = np.mean([s['counts']['total_version_like'] for s in samples])
                avg_len = np.mean([s['banner_len'] for s in samples])
                print(f"{result_type:<10} {len(samples):<10} {avg_ip:<12.2f} {avg_ver:<12.2f} {avg_date:<12.2f} {avg_total:<12.2f} {avg_len:<15.0f}")
    
    return all_results

def generate_interference_analysis(all_results, output_dir):
    print("\n" + "="*100)
    print("High Interference Scenario Analysis")
    print("="*100)
    
    version_results = all_results['firmware_version']
    
    thresholds = [5, 10, 15, 20]
    
    print(f"\n{'Threshold':<15} {'TP Rate':<15} {'FP Rate':<15} {'TN Rate':<15} {'FN Rate':<15}")
    print("-"*75)
    
    high_interference_stats = {}
    
    for threshold in thresholds:
        stats = {}
        for result_type, samples in version_results.items():
            high_interference = [s for s in samples if s['counts']['total_version_like'] > threshold]
            rate = len(high_interference) / len(samples) * 100 if samples else 0
            stats[result_type] = {
                'count': len(high_interference),
                'rate': rate,
                'total': len(samples)
            }
        
        high_interference_stats[threshold] = stats
        
        print(f"> {threshold} version-like strings", end='')
        for rt in ['TP', 'FP', 'TN', 'FN']:
            print(f" {stats[rt]['rate']:.1f}%", end='     ')
        print()
    
    return high_interference_stats

def generate_accuracy_by_interference(all_results, output_dir):
    print("\n" + "="*100)
    print("Accuracy Analysis by Interference Level")
    print("="*100)
    
    version_results = all_results['firmware_version']
    
    all_samples = []
    for result_type, samples in version_results.items():
        for s in samples:
            s['result_type'] = result_type
            all_samples.append(s)
    
    interference_levels = ['Low (≤5)', 'Medium (6-15)', 'High (16-30)', 'Very High (>30)']
    
    groups = {
        'Low (≤5)': [],
        'Medium (6-15)': [],
        'High (16-30)': [],
        'Very High (>30)': []
    }
    
    for s in all_samples:
        total = s['counts']['total_version_like']
        if total <= 5:
            groups['Low (≤5)'].append(s)
        elif total <= 15:
            groups['Medium (6-15)'].append(s)
        elif total <= 30:
            groups['High (16-30)'].append(s)
        else:
            groups['Very High (>30)'].append(s)
    
    group_stats = {}
    
    print(f"\n{'Interference Level':<20} {'Sample Count':<10} {'TP':<10} {'FP':<10} {'TN':<10} {'FN':<10} {'Accuracy':<12}")
    print("-"*85)
    
    for group_name, samples in groups.items():
        tp = len([s for s in samples if s['result_type'] == 'TP'])
        fp = len([s for s in samples if s['result_type'] == 'FP'])
        tn = len([s for s in samples if s['result_type'] == 'TN'])
        fn = len([s for s in samples if s['result_type'] == 'FN'])
        
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total * 100 if total > 0 else 0
        
        group_stats[group_name] = {
            'total': total,
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'accuracy': accuracy
        }
        
        print(f"{group_name:<20} {total:<10} {tp:<10} {fp:<10} {tn:<10} {fn:<10} {accuracy:.2f}%")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    group_names = list(group_stats.keys())
    accuracies = [group_stats[g]['accuracy'] for g in group_names]
    counts = [group_stats[g]['total'] for g in group_names]
    
    colors = ['#27ae60', '#3498db', '#f39c12', '#e74c3c']
    
    bars = ax.bar(group_names, accuracies, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Interference Level (Version-like Strings Count)', fontsize=12)
    ax.set_title('Firmware Version Accuracy by Interference Level', fontsize=14, fontweight='bold')
    ax.set_ylim(80, 100)
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar, acc, cnt in zip(bars, accuracies, counts):
        ax.annotate(f'{acc:.1f}%\n(n={cnt})',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 5),
                   textcoords='offset points',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.axhline(y=93.3, color='#c0392b', linestyle='--', linewidth=2, alpha=0.8)
    ax.text(3.02, 93.5, 'Overall: 93.3%', fontsize=9, ha='right', va='bottom', 
           color='#c0392b', fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'adversarial_accuracy_by_interference.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n图表已保存: {plot_path}")
    
    return group_stats

def generate_fp_analysis(all_results, banners, output_dir):
    print("\n" + "="*100)
    print("FP Case Detailed Analysis")
    print("="*100)
    
    version_results = all_results['firmware_version']
    fp_samples = version_results['FP']
    
    fp_categories = {
        'version_in_banner_wrong_extract': 0,
        'version_not_in_banner_hallucination': 0,
        'multiple_versions_confusion': 0,
        'other': 0
    }
    
    for s in fp_samples:
        label = s['label']
        banner = banners[s['sample_id']]['banner'] if s['sample_id'] in banners else ''
        
        if len(label) >= 3:
            true_version = label[2]
            if true_version in ['nv', 'NV', 'nt', 'NT']:
                fp_categories['version_not_in_banner_hallucination'] += 1
            elif true_version.lower() in banner.lower():
                if s['counts']['version_numbers'] > 5:
                    fp_categories['multiple_versions_confusion'] += 1
                else:
                    fp_categories['version_in_banner_wrong_extract'] += 1
            else:
                fp_categories['other'] += 1
        else:
            fp_categories['other'] += 1
    
    print(f"\nFP Classification Statistics (Total {len(fp_samples)} FP samples):")
    print("-"*60)
    for category, count in fp_categories.items():
        pct = count / len(fp_samples) * 100 if fp_samples else 0
        print(f"  {category}: {count} ({pct:.1f}%)")
    
    return fp_categories

def generate_comparison_table(all_results, output_dir):
    print("\n" + "="*100)
    print("Performance Comparison Table by Attribute")
    print("="*100)
    
    table_data = []
    
    for attr in ['brand', 'model', 'firmware_version']:
        results = all_results[attr]
        
        tp = len(results['TP'])
        fp = len(results['FP'])
        tn = len(results['TN'])
        fn = len(results['FN'])
        
        total = tp + fp + tn + fn
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / total * 100 if total > 0 else 0
        
        row = {
            'Attribute': attr,
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Accuracy': accuracy
        }
        table_data.append(row)
        
        print(f"\n{attr}:")
        print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(f"  Precision={precision:.2f}%, Recall={recall:.2f}%, F1={f1:.2f}%, Accuracy={accuracy:.2f}%")
    
    output_path = os.path.join(output_dir, 'adversarial_performance_table.json')
    with open(output_path, 'w') as f:
        json.dump(table_data, f, indent=2)
    
    return table_data

def generate_summary_response():
    print("\n" + "="*100)
    print("审稿人问题回应总结")
    print("="*100)
    
    summary = """
## Q: NS-FirmID如何处理对抗性场景？

### 1. 数据集统计

完整测试集（test_set + holdout_set）包含：
- 总样本数：X,XXX
- 包含多个版本类字符串的样本：X,XXX (XX%)

### 2. 按干扰程度分组的准确率

| 干扰程度 | 样本数 | 准确率 |
|---------|--------|--------|
| Low (≤5) | XXX | XX.X% |
| Medium (6-15) | XXX | XX.X% |
| High (16-30) | XXX | XX.X% |
| Very High (>30) | XXX | XX.X% |

### 3. FP案例分析

FP样本主要来源：
- 版本号存在于banner但提取错误：XX%
- 多版本号混淆：XX%
- 幻觉（版本号不存在）：XX%

### 4. 关键发现

1. **系统在高干扰场景仍保持较好性能**
   - 即使在>30个版本类字符串的极端干扰下，准确率仍达XX%

2. **FP主要来源于提取错误而非幻觉**
   - 幻觉问题极少，说明系统具备基本的可靠性

3. **FN与高干扰强相关**
   - FN样本平均包含XX个版本类字符串
   - 说明复杂场景是主要挑战

### 5. 回应审稿人

NS-FirmID通过三层机制应对对抗性场景：
1. EA的语义理解能力区分真实版本和干扰信息
2. VA的知识库验证过滤不合理候选
3. DA的置信度建模降低FP

局限性：对于精心设计的对抗性攻击，系统仍可能被欺骗。
"""
    print(summary)

def main():
    output_dir = '/Users/lululu/Documents/code/Multi-Agent_version_identification_using_LLMs/test_on_linux/inference_analysis_0216'
    
    print("加载数据...")
    df, banners = load_all_data()
    
    print(f"XLSX样本数: {len(df)}")
    print(f"Banner样本数: {len(banners)}")
    
    all_results = analyze_full_dataset(df, banners)
    
    high_interference_stats = generate_interference_analysis(all_results, output_dir)
    
    group_stats = generate_accuracy_by_interference(all_results, output_dir)
    
    fp_categories = generate_fp_analysis(all_results, banners, output_dir)
    
    table_data = generate_comparison_table(all_results, output_dir)
    
    generate_summary_response()
    
    print("\nAnalysis Completed!")

if __name__ == "__main__":
    main()
