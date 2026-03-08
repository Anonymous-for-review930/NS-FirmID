import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter
import re
import pandas as pd

def load_xlsx_results(file_path):
    df = pd.read_excel(file_path)
    return df

def load_jsonl_data(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    index = entry.get('index', '')
                    banner = entry.get('banner', '')
                    data[index] = banner
                except json.JSONDecodeError:
                    pass
    return data

def load_nsfirmid_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def process_xlsx_results(df):
    results = {}
    
    df = df[df['Strategy'] == 'confidence_filter']
    
    for _, row in df.iterrows():
        sample_id = row['Sample_ID']
        attribute = row['Attribute']
        result = row['Result']
        
        if sample_id not in results:
            results[sample_id] = {
                'version_TP': 0, 'version_FP': 0, 'version_TN': 0, 'version_FN': 0
            }
        
        attr_map = {
            'firmware_version': 'version'
        }
        
        if attribute in attr_map:
            attr_name = attr_map[attribute]
            if result == 'TP':
                results[sample_id][f'{attr_name}_TP'] = 1
            elif result == 'FP':
                results[sample_id][f'{attr_name}_FP'] = 1
            elif result == 'TN':
                results[sample_id][f'{attr_name}_TN'] = 1
            elif result == 'FN':
                results[sample_id][f'{attr_name}_FN'] = 1
    
    return results

def calculate_accuracy_from_results(sample_results):
    version_correct = 1 if (sample_results['version_TP'] == 1 or sample_results['version_TN'] == 1) else 0
    
    return {
        'version_correct': version_correct
    }

def calculate_shannon_entropy(text):
    if not text:
        return 0
    char_counts = Counter(text)
    total_chars = len(text)
    entropy = 0
    for count in char_counts.values():
        probability = count / total_chars
        if probability > 0:
            entropy -= probability * np.log2(probability)
    return entropy

def calculate_heterogeneity_metrics(banner):
    if not banner:
        return {
            'length': 0,
            'entropy': 0,
            'unique_chars': 0,
            'digit_ratio': 0,
            'letter_ratio': 0,
            'special_char_ratio': 0,
            'line_count': 0,
            'structure_score': 0,
            'word_count': 0,
            'avg_word_length': 0
        }
    
    banner_str = str(banner)
    length = len(banner_str)
    
    entropy = calculate_shannon_entropy(banner_str)
    
    unique_chars = len(set(banner_str))
    
    digit_count = sum(1 for c in banner_str if c.isdigit())
    letter_count = sum(1 for c in banner_str if c.isalpha())
    special_count = length - digit_count - letter_count
    
    digit_ratio = digit_count / length if length > 0 else 0
    letter_ratio = letter_count / length if length > 0 else 0
    special_char_ratio = special_count / length if length > 0 else 0
    
    line_count = banner_str.count('\n') + 1
    
    kv_patterns = [
        r'[\w\-]+:\s*[\w\-\.\/]+',
        r'[\w\-]+=\s*[\w\-\.\/]+',
        r'[\w\-]+\s*=\s*["\'][^"\']*["\']'
    ]
    kv_matches = 0
    for pattern in kv_patterns:
        kv_matches += len(re.findall(pattern, banner_str))
    structure_score = kv_matches / line_count if line_count > 0 else 0
    
    words = re.findall(r'\b\w+\b', banner_str)
    word_count = len(words)
    avg_word_length = np.mean([len(w) for w in words]) if words else 0
    
    return {
        'length': length,
        'entropy': entropy,
        'unique_chars': unique_chars,
        'digit_ratio': digit_ratio,
        'letter_ratio': letter_ratio,
        'special_char_ratio': special_char_ratio,
        'line_count': line_count,
        'structure_score': structure_score,
        'word_count': word_count,
        'avg_word_length': avg_word_length
    }

def estimate_sample_times(nsfirmid_data):
    results = nsfirmid_data.get('results', [])
    total_duration = nsfirmid_data.get('total_duration', 0)
    
    total_tokens = 0
    token_counts = []
    
    for result in results:
        input_tokens = result.get('input_tokens', 0)
        output_tokens = result.get('output_tokens', 0)
        total_tokens += input_tokens + output_tokens
        token_counts.append(input_tokens + output_tokens)
    
    sample_times = []
    for i, result in enumerate(results):
        sample_time = (token_counts[i] / total_tokens * total_duration) if total_tokens > 0 else 0
        sample_times.append(sample_time * 1000)
    
    return sample_times

def calculate_correlation(heterogeneity_metrics, performance_metrics):
    correlations = {}
    
    het_names = list(heterogeneity_metrics[0].keys())
    perf_names = list(performance_metrics[0].keys())
    
    for het_name in het_names:
        correlations[het_name] = {}
        het_values = [h[het_name] for h in heterogeneity_metrics]
        
        for perf_name in perf_names:
            perf_values = [p[perf_name] for p in performance_metrics]
            
            valid_pairs = [(h, p) for h, p in zip(het_values, perf_values) 
                          if not (np.isnan(h) or np.isnan(p))]
            
            if len(valid_pairs) > 10:
                valid_h = [p[0] for p in valid_pairs]
                valid_p = [p[1] for p in valid_pairs]
                
                if len(set(valid_h)) > 1 and len(set(valid_p)) > 1:
                    corr, p_value = stats.pearsonr(valid_h, valid_p)
                    correlations[het_name][perf_name] = {
                        'correlation': corr,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'n_samples': len(valid_pairs)
                    }
                else:
                    correlations[het_name][perf_name] = {
                        'correlation': 0,
                        'p_value': 1,
                        'significant': False,
                        'n_samples': len(valid_pairs)
                    }
            else:
                correlations[het_name][perf_name] = {
                    'correlation': 0,
                    'p_value': 1,
                    'significant': False,
                    'n_samples': len(valid_pairs)
                }
    
    return correlations

def generate_correlation_heatmap(correlations, output_dir):
    het_names = list(correlations.keys())
    perf_names = list(correlations[het_names[0]].keys())
    
    matrix = np.zeros((len(het_names), len(perf_names)))
    
    for i, het_name in enumerate(het_names):
        for j, perf_name in enumerate(perf_names):
            matrix[i, j] = correlations[het_name][perf_name]['correlation']
    
    fig, ax = plt.subplots(figsize=(10, 14))
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    ax.set_xticks(np.arange(len(perf_names)))
    ax.set_yticks(np.arange(len(het_names)))
    ax.set_xticklabels(perf_names, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(het_names, fontsize=12)
    
    for i in range(len(het_names)):
        for j in range(len(perf_names)):
            corr = matrix[i, j]
            p_val = correlations[het_names[i]][perf_names[j]]['p_value']
            sig_marker = '*' if p_val < 0.05 else ''
            text = f'{corr:.2f}{sig_marker}'
            ax.text(j, i, text, ha='center', va='center', fontsize=11,
                   color='white' if abs(corr) > 0.5 else 'black')
    
    plt.colorbar(im, ax=ax, label='Correlation Coefficient')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'heterogeneity_correlation_heatmap_v3_no_title.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def generate_scatter_plots(heterogeneity_metrics, performance_metrics, output_dir):
    key_het_metrics = ['length', 'entropy', 'structure_score', 'digit_ratio']
    key_perf_metrics = ['sample_time', 'version_correct']
    
    fig, axes = plt.subplots(len(key_het_metrics), len(key_perf_metrics), 
                             figsize=(10, 16))
    
    for i, het_name in enumerate(key_het_metrics):
        for j, perf_name in enumerate(key_perf_metrics):
            ax = axes[i, j]
            
            het_values = [h[het_name] for h in heterogeneity_metrics]
            perf_values = [p[perf_name] for p in performance_metrics]
            
            valid_pairs = [(h, p) for h, p in zip(het_values, perf_values) 
                          if not (np.isnan(h) or np.isnan(p))]
            
            if valid_pairs:
                valid_h = [p[0] for p in valid_pairs]
                valid_p = [p[1] for p in valid_pairs]
                
                if perf_name == 'version_correct':
                    x_jitter = np.random.normal(0, 0.02, len(valid_h))
                    ax.scatter(np.array(valid_h) + x_jitter * np.std(valid_h), 
                              valid_p, alpha=0.2, s=15)
                else:
                    ax.scatter(valid_h, valid_p, alpha=0.3, s=20)
                
                if len(set(valid_h)) > 1 and len(set(valid_p)) > 1:
                    corr, p_val = stats.pearsonr(valid_h, valid_p)
                    sig_text = '*' if p_val < 0.05 else ''
                else:
                    sig_text = ''
            else:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            
            ax.set_xlabel(het_name, fontsize=11)
            ax.set_ylabel(perf_name, fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'heterogeneity_scatter_plots_v3_no_title.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def generate_grouped_analysis(heterogeneity_metrics, performance_metrics, output_dir):
    lengths = [h['length'] for h in heterogeneity_metrics]
    
    q33 = np.percentile(lengths, 33)
    q66 = np.percentile(lengths, 66)
    
    groups = {'Short': [], 'Medium': [], 'Long': []}
    
    for i, length in enumerate(lengths):
        if length <= q33:
            groups['Short'].append(i)
        elif length <= q66:
            groups['Medium'].append(i)
        else:
            groups['Long'].append(i)
    
    group_stats = {}
    for group_name, indices in groups.items():
        times = [performance_metrics[i]['sample_time'] for i in indices 
                if not np.isnan(performance_metrics[i]['sample_time'])]
        version_corrects = [performance_metrics[i]['version_correct'] for i in indices]
        
        group_stats[group_name] = {
            'count': len(indices),
            'time_count': len(times),
            'avg_time': np.mean(times) if times else 0,
            'time_std': np.std(times) if times else 0,
            'version_accuracy': np.mean(version_corrects) * 100
        }
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    group_names = list(group_stats.keys())
    colors = ['#3498db', '#f39c12', '#e74c3c']
    
    ax1 = axes[0]
    avg_times = [group_stats[g]['avg_time'] for g in group_names]
    time_stds = [group_stats[g]['time_std'] for g in group_names]
    bars1 = ax1.bar(group_names, avg_times, yerr=time_stds, capsize=5, 
                   color=colors, alpha=0.8)
    ax1.set_ylabel('Time (ms)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
    for bar, val, count in zip(bars1, avg_times, [group_stats[g]['time_count'] for g in group_names]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{val:.1f}\n(n={count})', ha='center', va='bottom', fontsize=10)
    
    ax2 = axes[1]
    version_accs = [group_stats[g]['version_accuracy'] for g in group_names]
    bars2 = ax2.bar(group_names, version_accs, color=colors, alpha=0.8)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
    for bar, val in zip(bars2, version_accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'heterogeneity_grouped_analysis_v3_no_title.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return group_stats, plot_path

def generate_entropy_analysis(heterogeneity_metrics, performance_metrics, output_dir):
    entropies = [h['entropy'] for h in heterogeneity_metrics]
    
    q33 = np.percentile(entropies, 33)
    q66 = np.percentile(entropies, 66)
    
    groups = {'Low Entropy': [], 'Medium Entropy': [], 'High Entropy': []}
    
    for i, entropy in enumerate(entropies):
        if entropy <= q33:
            groups['Low Entropy'].append(i)
        elif entropy <= q66:
            groups['Medium Entropy'].append(i)
        else:
            groups['High Entropy'].append(i)
    
    group_stats = {}
    for group_name, indices in groups.items():
        times = [performance_metrics[i]['sample_time'] for i in indices 
                if not np.isnan(performance_metrics[i]['sample_time'])]
        version_corrects = [performance_metrics[i]['version_correct'] for i in indices]
        
        group_stats[group_name] = {
            'count': len(indices),
            'time_count': len(times),
            'avg_time': np.mean(times) if times else 0,
            'time_std': np.std(times) if times else 0,
            'version_accuracy': np.mean(version_corrects) * 100
        }
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    group_names = list(group_stats.keys())
    colors = ['#2ecc71', '#f39c12', '#9b59b6']
    
    ax1 = axes[0]
    avg_times = [group_stats[g]['avg_time'] for g in group_names]
    time_stds = [group_stats[g]['time_std'] for g in group_names]
    bars1 = ax1.bar(group_names, avg_times, yerr=time_stds, capsize=5,
                   color=colors, alpha=0.8)
    ax1.set_ylabel('Time (ms)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
    for bar, val, count in zip(bars1, avg_times, [group_stats[g]['time_count'] for g in group_names]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{val:.1f}\n(n={count})', ha='center', va='bottom', fontsize=10)
    
    ax2 = axes[1]
    version_accs = [group_stats[g]['version_accuracy'] for g in group_names]
    bars2 = ax2.bar(group_names, version_accs, color=colors, alpha=0.8)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
    for bar, val in zip(bars2, version_accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'entropy_grouped_analysis_v3_no_title.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return group_stats, plot_path

def generate_structure_analysis(heterogeneity_metrics, performance_metrics, output_dir):
    structure_scores = [h['structure_score'] for h in heterogeneity_metrics]
    
    q33 = np.percentile(structure_scores, 33)
    q66 = np.percentile(structure_scores, 66)
    
    groups = {'Low Structure': [], 'Medium Structure': [], 'High Structure': []}
    
    for i, score in enumerate(structure_scores):
        if score <= q33:
            groups['Low Structure'].append(i)
        elif score <= q66:
            groups['Medium Structure'].append(i)
        else:
            groups['High Structure'].append(i)
    
    group_stats = {}
    for group_name, indices in groups.items():
        times = [performance_metrics[i]['sample_time'] for i in indices 
                if not np.isnan(performance_metrics[i]['sample_time'])]
        version_corrects = [performance_metrics[i]['version_correct'] for i in indices]
        
        group_stats[group_name] = {
            'count': len(indices),
            'time_count': len(times),
            'avg_time': np.mean(times) if times else 0,
            'time_std': np.std(times) if times else 0,
            'version_accuracy': np.mean(version_corrects) * 100
        }
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    group_names = list(group_stats.keys())
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    
    ax1 = axes[0]
    avg_times = [group_stats[g]['avg_time'] for g in group_names]
    time_stds = [group_stats[g]['time_std'] for g in group_names]
    bars1 = ax1.bar(group_names, avg_times, yerr=time_stds, capsize=5,
                   color=colors, alpha=0.8)
    ax1.set_ylabel('Time (ms)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
    for bar, val, count in zip(bars1, avg_times, [group_stats[g]['time_count'] for g in group_names]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{val:.1f}\n(n={count})', ha='center', va='bottom', fontsize=10)
    
    ax2 = axes[1]
    version_accs = [group_stats[g]['version_accuracy'] for g in group_names]
    bars2 = ax2.bar(group_names, version_accs, color=colors, alpha=0.8)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
    for bar, val in zip(bars2, version_accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'structure_grouped_analysis_v3_no_title.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return group_stats, plot_path

def generate_digit_ratio_analysis(heterogeneity_metrics, performance_metrics, output_dir):
    digit_ratios = [h['digit_ratio'] for h in heterogeneity_metrics]
    
    q33 = np.percentile(digit_ratios, 33)
    q66 = np.percentile(digit_ratios, 66)
    
    groups = {'Low Digit Ratio': [], 'Medium Digit Ratio': [], 'High Digit Ratio': []}
    
    for i, ratio in enumerate(digit_ratios):
        if ratio <= q33:
            groups['Low Digit Ratio'].append(i)
        elif ratio <= q66:
            groups['Medium Digit Ratio'].append(i)
        else:
            groups['High Digit Ratio'].append(i)
    
    group_stats = {}
    for group_name, indices in groups.items():
        times = [performance_metrics[i]['sample_time'] for i in indices 
                if not np.isnan(performance_metrics[i]['sample_time'])]
        version_corrects = [performance_metrics[i]['version_correct'] for i in indices]
        
        group_stats[group_name] = {
            'count': len(indices),
            'time_count': len(times),
            'avg_time': np.mean(times) if times else 0,
            'time_std': np.std(times) if times else 0,
            'version_accuracy': np.mean(version_corrects) * 100
        }
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    group_names = list(group_stats.keys())
    colors = ['#9b59b6', '#f39c12', '#3498db']
    
    ax1 = axes[0]
    avg_times = [group_stats[g]['avg_time'] for g in group_names]
    time_stds = [group_stats[g]['time_std'] for g in group_names]
    bars1 = ax1.bar(group_names, avg_times, yerr=time_stds, capsize=5,
                   color=colors, alpha=0.8)
    ax1.set_ylabel('Time (ms)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
    for bar, val, count in zip(bars1, avg_times, [group_stats[g]['time_count'] for g in group_names]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{val:.1f}\n(n={count})', ha='center', va='bottom', fontsize=10)
    
    ax2 = axes[1]
    version_accs = [group_stats[g]['version_accuracy'] for g in group_names]
    bars2 = ax2.bar(group_names, version_accs, color=colors, alpha=0.8)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
    for bar, val in zip(bars2, version_accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'digit_ratio_grouped_analysis_v3_no_title.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return group_stats, plot_path

def save_correlation_table(correlations, output_dir):
    het_names = list(correlations.keys())
    perf_names = list(correlations[het_names[0]].keys())
    
    table_data = []
    
    print("\n" + "="*100)
    print("CORRELATION ANALYSIS: Banner Heterogeneity vs Firmware Version Performance")
    print("="*100)
    print(f"{'Heterogeneity Metric':<25} {'Performance Metric':<20} {'Correlation':<12} {'P-value':<12} {'Significant':<10}")
    print("-"*100)
    
    for het_name in het_names:
        for perf_name in perf_names:
            corr = correlations[het_name][perf_name]['correlation']
            p_val = correlations[het_name][perf_name]['p_value']
            sig = correlations[het_name][perf_name]['significant']
            
            row = {
                'heterogeneity_metric': het_name,
                'performance_metric': perf_name,
                'correlation': round(corr, 4),
                'p_value': round(p_val, 6),
                'significant': 'Yes' if sig else 'No'
            }
            table_data.append(row)
            
            sig_marker = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
            print(f"{het_name:<25} {perf_name:<20} {corr:<12.4f} {p_val:<12.6f} {sig_marker:<10}")
    
    print("="*100)
    print("Significance levels: * p<0.05, ** p<0.01, *** p<0.001")
    
    table_path = os.path.join(output_dir, 'heterogeneity_correlation_table_v3.json')
    with open(table_path, 'w', encoding='utf-8') as f:
        json.dump(table_data, f, indent=2)
    
    return table_data, table_path

def main():
    xlsx_file = '/Users/lululu/Documents/code/Multi-Agent_version_identification_using_LLMs/agent/ablation_performance_1204_xgb_test_and_holdout_set_sft_Qwen2___5_7B_1130_1203/ablation_details_full_no_comp.xlsx'
    test_set_file = '/Users/lululu/Documents/code/Multi-Agent_version_identification_using_LLMs/data/test_set.jsonl'
    holdout_set_file = '/Users/lululu/Documents/code/Multi-Agent_version_identification_using_LLMs/data/holdout_set.jsonl'
    nsfirmid_file = '/Users/lululu/Documents/code/Multi-Agent_version_identification_using_LLMs/test_on_linux/inference_analysis_0216/test_set_for_plot.json'
    output_dir = '/Users/lululu/Documents/code/Multi-Agent_version_identification_using_LLMs/test_on_linux/inference_analysis_0216'
    
    print("Loading XLSX results...")
    xlsx_df = load_xlsx_results(xlsx_file)
    print(f"Loaded {len(xlsx_df)} rows from XLSX")
    
    print("\nProcessing XLSX results (firmware_version only)...")
    xlsx_results = process_xlsx_results(xlsx_df)
    print(f"Processed {len(xlsx_results)} unique samples")
    
    print("\nLoading banner data from JSONL files...")
    test_banners = load_jsonl_data(test_set_file)
    holdout_banners = load_jsonl_data(holdout_set_file)
    all_banners = {**test_banners, **holdout_banners}
    print(f"Loaded {len(all_banners)} banners (test: {len(test_banners)}, holdout: {len(holdout_banners)})")
    
    print("\nLoading NS-FirmID timing data...")
    nsfirmid_data = load_nsfirmid_data(nsfirmid_file)
    sample_times = estimate_sample_times(nsfirmid_data)
    print(f"Estimated {len(sample_times)} sample times")
    
    test_indices_list = list(test_banners.keys())
    test_banner_lengths = {idx: len(test_banners[idx]) for idx in test_indices_list}
    
    nsfirmid_results = nsfirmid_data.get('results', [])
    nsfirmid_banner_lengths = [int(r.get('banner_len', 0)) for r in nsfirmid_results]
    
    nsfirmid_index_map = {}
    for i, ns_len in enumerate(nsfirmid_banner_lengths):
        for j, (idx, test_len) in enumerate(test_banner_lengths.items()):
            if abs(test_len - ns_len) <= 5 and idx not in nsfirmid_index_map:
                nsfirmid_index_map[idx] = i
                break
    
    print(f"Matched {len(nsfirmid_index_map)} NS-FirmID results with test set indices")
    
    print("\nMatching data and calculating metrics...")
    heterogeneity_metrics = []
    performance_metrics = []
    matched_count = 0
    time_matched_count = 0
    
    for sample_id, sample_results in xlsx_results.items():
        if sample_id in all_banners:
            banner = all_banners[sample_id]
            
            het_metrics = calculate_heterogeneity_metrics(banner)
            heterogeneity_metrics.append(het_metrics)
            
            perf_metrics = calculate_accuracy_from_results(sample_results)
            
            if sample_id in nsfirmid_index_map:
                idx = nsfirmid_index_map[sample_id]
                perf_metrics['sample_time'] = sample_times[idx]
                time_matched_count += 1
            else:
                perf_metrics['sample_time'] = np.nan
            
            performance_metrics.append(perf_metrics)
            matched_count += 1
    
    print(f"Matched {matched_count} samples with banners")
    print(f"Matched {time_matched_count} samples with timing data")
    
    valid_indices = [i for i, p in enumerate(performance_metrics) if not np.isnan(p['sample_time'])]
    print(f"Samples with valid timing data: {len(valid_indices)}")
    
    overall_version_accuracy = np.mean([p['version_correct'] for p in performance_metrics]) * 100
    print(f"\nOverall Firmware Version Accuracy: {overall_version_accuracy:.2f}%")
    
    print("\nCalculating correlations...")
    correlations = calculate_correlation(heterogeneity_metrics, performance_metrics)
    
    print("\nSaving correlation table...")
    table_data, table_path = save_correlation_table(correlations, output_dir)
    print(f"Correlation table saved to: {table_path}")
    
    print("\nGenerating correlation heatmap...")
    plot_path = generate_correlation_heatmap(correlations, output_dir)
    print(f"Heatmap saved to: {plot_path}")
    
    print("\nGenerating scatter plots...")
    plot_path = generate_scatter_plots(heterogeneity_metrics, performance_metrics, output_dir)
    print(f"Scatter plots saved to: {plot_path}")
    
    print("\nGenerating grouped analysis by banner length...")
    group_stats, plot_path = generate_grouped_analysis(heterogeneity_metrics, performance_metrics, output_dir)
    print(f"Grouped analysis saved to: {plot_path}")
    
    print("\nGenerating grouped analysis by entropy...")
    entropy_stats, plot_path = generate_entropy_analysis(heterogeneity_metrics, performance_metrics, output_dir)
    print(f"Entropy analysis saved to: {plot_path}")
    
    print("\nGenerating grouped analysis by structure score...")
    structure_stats, plot_path = generate_structure_analysis(heterogeneity_metrics, performance_metrics, output_dir)
    print(f"Structure analysis saved to: {plot_path}")
    
    print("\nGenerating grouped analysis by digit ratio...")
    digit_stats, plot_path = generate_digit_ratio_analysis(heterogeneity_metrics, performance_metrics, output_dir)
    print(f"Digit ratio analysis saved to: {plot_path}")
    
    print("\n" + "="*80)
    print("HETEROGENEITY ANALYSIS COMPLETE (Firmware Version Only)")
    print("="*80)
    
    print("\nKey Findings:")
    print("-" * 40)
    
    length_time_corr = correlations['length']['sample_time']['correlation']
    length_time_sig = correlations['length']['sample_time']['significant']
    print(f"1. Banner Length vs Processing Time: r={length_time_corr:.3f} {'(significant)' if length_time_sig else ''}")
    
    length_version_corr = correlations['length']['version_correct']['correlation']
    length_version_sig = correlations['length']['version_correct']['significant']
    print(f"2. Banner Length vs Version Accuracy: r={length_version_corr:.3f} {'(significant)' if length_version_sig else ''}")
    
    structure_version_corr = correlations['structure_score']['version_correct']['correlation']
    structure_version_sig = correlations['structure_score']['version_correct']['significant']
    print(f"3. Structure Score vs Version Accuracy: r={structure_version_corr:.3f} {'(significant)' if structure_version_sig else ''}")
    
    digit_version_corr = correlations['digit_ratio']['version_correct']['correlation']
    digit_version_sig = correlations['digit_ratio']['version_correct']['significant']
    print(f"4. Digit Ratio vs Version Accuracy: r={digit_version_corr:.3f} {'(significant)' if digit_version_sig else ''}")
    
    entropy_version_corr = correlations['entropy']['version_correct']['correlation']
    entropy_version_sig = correlations['entropy']['version_correct']['significant']
    print(f"5. Entropy vs Version Accuracy: r={entropy_version_corr:.3f} {'(significant)' if entropy_version_sig else ''}")

if __name__ == "__main__":
    main()
