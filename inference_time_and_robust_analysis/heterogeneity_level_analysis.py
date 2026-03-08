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

def classify_heterogeneity_level(het_metrics, thresholds):
    level_scores = {
        'low': 0,
        'medium': 0,
        'high': 0
    }
    
    metric_weights = {
        'length': 1.0,
        'entropy': 1.2,
        'digit_ratio': 0.8,
        'structure_score': 1.0,
        'word_count': 0.9
    }
    
    for metric, weight in metric_weights.items():
        value = het_metrics.get(metric, 0)
        low_thresh = thresholds[metric]['low']
        high_thresh = thresholds[metric]['high']
        
        if value <= low_thresh:
            level_scores['low'] += weight
        elif value <= high_thresh:
            level_scores['medium'] += weight
        else:
            level_scores['high'] += weight
    
    max_score = max(level_scores.values())
    for level, score in level_scores.items():
        if score == max_score:
            return level
    
    return 'medium'

def calculate_heterogeneity_thresholds(heterogeneity_metrics):
    thresholds = {}
    
    metric_names = ['length', 'entropy', 'digit_ratio', 'structure_score', 'word_count']
    
    for metric in metric_names:
        values = [h[metric] for h in heterogeneity_metrics]
        q33 = np.percentile(values, 33)
        q66 = np.percentile(values, 66)
        
        thresholds[metric] = {
            'low': q33,
            'high': q66,
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
    return thresholds

def analyze_heterogeneity_levels(heterogeneity_metrics, performance_metrics, thresholds):
    levels = {'low': [], 'medium': [], 'high': []}
    
    for i, het in enumerate(heterogeneity_metrics):
        level = classify_heterogeneity_level(het, thresholds)
        levels[level].append(i)
    
    level_stats = {}
    for level, indices in levels.items():
        het_values = [heterogeneity_metrics[i] for i in indices]
        perf_values = [performance_metrics[i] for i in indices]
        
        stats = {
            'count': len(indices),
            'percentage': len(indices) / len(heterogeneity_metrics) * 100
        }
        
        for metric in ['length', 'entropy', 'digit_ratio', 'structure_score', 'word_count']:
            values = [h[metric] for h in het_values]
            stats[f'{metric}_mean'] = np.mean(values)
            stats[f'{metric}_std'] = np.std(values)
        
        version_corrects = [p['version_correct'] for p in perf_values]
        stats['version_accuracy'] = np.mean(version_corrects) * 100
        
        times = [p['sample_time'] for p in perf_values if not np.isnan(p['sample_time'])]
        if times:
            stats['avg_time'] = np.mean(times)
            stats['time_std'] = np.std(times)
        else:
            stats['avg_time'] = 0
            stats['time_std'] = 0
        
        level_stats[level] = stats
    
    return level_stats

def generate_heterogeneity_level_comparison(level_stats, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    levels = ['low', 'medium', 'high']
    level_names = ['Low Heterogeneity', 'Medium Heterogeneity', 'High Heterogeneity']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    ax1 = axes[0, 0]
    counts = [level_stats[l]['count'] for l in levels]
    bars = ax1.bar(level_names, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Number of Samples', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax1.set_ylim(0, max(counts) * 1.2)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + max(counts) * 0.02, 
                f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2 = axes[0, 1]
    version_accs = [level_stats[l]['version_accuracy'] for l in levels]
    bars = ax2.bar(level_names, version_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_ylim(85, 100)
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
    for bar, acc in zip(bars, version_accs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.5, 
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3 = axes[1, 0]
    avg_times = [level_stats[l]['avg_time'] for l in levels]
    time_stds = [level_stats[l]['time_std'] for l in levels]
    bars = ax3.bar(level_names, avg_times, yerr=time_stds, capsize=5,
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Time (ms)', fontsize=11)
    ax3.grid(True, linestyle='--', alpha=0.5, axis='y')
    max_time = max(avg_times) if avg_times else 1
    ax3.set_ylim(0, max_time * 1.3)
    for bar, time_val in zip(bars, avg_times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + max_time * 0.05, 
                f'{time_val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax4 = axes[1, 1]
    lengths = [level_stats[l]['length_mean'] for l in levels]
    length_stds = [level_stats[l]['length_std'] for l in levels]
    bars = ax4.bar(level_names, lengths, yerr=length_stds, capsize=5,
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Banner Length (chars)', fontsize=11)
    ax4.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax4.set_yscale('log')
    ax4.set_ylim(100, 10000)
    for bar, length_val in zip(bars, lengths):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height * 1.2, 
                f'{length_val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'heterogeneity_level_comparison_no_title.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def generate_heterogeneity_radar_plot(level_stats, thresholds, output_dir):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    levels = ['low', 'medium', 'high']
    level_names = ['Low Heterogeneity', 'Medium Heterogeneity', 'High Heterogeneity']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    metric_names = ['length', 'entropy', 'digit_ratio', 'structure_score', 'word_count']
    metric_labels = ['Length', 'Entropy', 'Digit\nRatio', 'Structure\nScore', 'Word\nCount']
    
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]
    
    for i, level in enumerate(levels):
        normalized_values = []
        for metric in metric_names:
            mean_val = level_stats[level][f'{metric}_mean']
            max_val = thresholds[metric]['high'] * 1.5
            normalized_val = min(mean_val / max_val, 1.0)
            normalized_values.append(normalized_val)
        
        normalized_values += normalized_values[:1]
        
        ax.plot(angles, normalized_values, 'o-', linewidth=2.5, label=level_names[i], 
                color=colors[i], markersize=10)
        ax.fill(angles, normalized_values, alpha=0.2, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=14)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'heterogeneity_radar_plot_no_title.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def generate_metric_distribution_plots(heterogeneity_metrics, thresholds, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    metrics = ['length', 'entropy', 'digit_ratio', 'structure_score', 'word_count']
    metric_labels = ['Banner Length (chars)', 'Shannon Entropy', 'Digit Ratio', 
                     'Structure Score', 'Word Count']
    colors = ['#3498db', '#f39c12', '#e74c3c']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        values = [h[metric] for h in heterogeneity_metrics]
        
        low_thresh = thresholds[metric]['low']
        high_thresh = thresholds[metric]['high']
        
        ax.hist(values, bins=50, color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(low_thresh, color='#f39c12', linestyle='--', linewidth=2, label='Low Threshold')
        ax.axvline(high_thresh, color='#e74c3c', linestyle='--', linewidth=2, label='High Threshold')
        
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')
        
        if i == 0:
            ax.legend(fontsize=9, loc='upper right')
    
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'heterogeneity_metric_distributions.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def save_heterogeneity_level_statistics(level_stats, thresholds, output_dir):
    print("\n" + "="*100)
    print("HETEROGENEITY LEVEL ANALYSIS RESULTS")
    print("="*100)
    
    print("\n" + "-"*100)
    print("THRESHOLDS FOR HETEROGENEITY CLASSIFICATION")
    print("-"*100)
    print(f"{'Metric':<20} {'Low Threshold':<20} {'High Threshold':<20} {'Mean':<15} {'Std':<15}")
    print("-"*100)
    
    for metric, thresh in thresholds.items():
        print(f"{metric:<20} {thresh['low']:<20.3f} {thresh['high']:<20.3f} {thresh['mean']:<15.3f} {thresh['std']:<15.3f}")
    
    print("\n" + "-"*100)
    print("HETEROGENEITY LEVEL STATISTICS")
    print("-"*100)
    
    for level in ['low', 'medium', 'high']:
        stats = level_stats[level]
        print(f"\n{'='*100}")
        print(f"{level.upper()} HETEROGENEITY")
        print(f"{'='*100}")
        print(f"Sample Count: {stats['count']} ({stats['percentage']:.2f}%)")
        print(f"Version Accuracy: {stats['version_accuracy']:.2f}%")
        print(f"Average Processing Time: {stats['avg_time']:.2f} ± {stats['time_std']:.2f} ms")
        print(f"\nBanner Characteristics:")
        print(f"  - Length: {stats['length_mean']:.0f} ± {stats['length_std']:.0f} chars")
        print(f"  - Entropy: {stats['entropy_mean']:.3f} ± {stats['entropy_std']:.3f}")
        print(f"  - Digit Ratio: {stats['digit_ratio_mean']:.3f} ± {stats['digit_ratio_std']:.3f}")
        print(f"  - Structure Score: {stats['structure_score_mean']:.3f} ± {stats['structure_score_std']:.3f}")
        print(f"  - Word Count: {stats['word_count_mean']:.0f} ± {stats['word_count_std']:.0f}")
    
    stats_path = os.path.join(output_dir, 'heterogeneity_level_statistics.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({
            'thresholds': thresholds,
            'level_stats': level_stats
        }, f, indent=2)
    
    return stats_path

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
    
    print("\nCalculating heterogeneity thresholds...")
    thresholds = calculate_heterogeneity_thresholds(heterogeneity_metrics)
    
    print("\nAnalyzing heterogeneity levels...")
    level_stats = analyze_heterogeneity_levels(heterogeneity_metrics, performance_metrics, thresholds)
    
    print("\nGenerating heterogeneity level comparison plots...")
    plot_path = generate_heterogeneity_level_comparison(level_stats, output_dir)
    print(f"Heterogeneity level comparison saved to: {plot_path}")
    
    print("\nGenerating heterogeneity radar plot...")
    plot_path = generate_heterogeneity_radar_plot(level_stats, thresholds, output_dir)
    print(f"Heterogeneity radar plot saved to: {plot_path}")
    
    print("\nGenerating heterogeneity level comparison plots (no title)...")
    plot_path = generate_heterogeneity_level_comparison(level_stats, output_dir)
    print(f"Heterogeneity level comparison (no title) saved to: {plot_path}")
    
    print("\nGenerating heterogeneity radar plot (no title)...")
    plot_path = generate_heterogeneity_radar_plot(level_stats, thresholds, output_dir)
    print(f"Heterogeneity radar plot (no title) saved to: {plot_path}")
    
    print("\nGenerating metric distribution plots...")
    plot_path = generate_metric_distribution_plots(heterogeneity_metrics, thresholds, output_dir)
    print(f"Metric distribution plots saved to: {plot_path}")
    
    print("\nSaving heterogeneity level statistics...")
    stats_path = save_heterogeneity_level_statistics(level_stats, thresholds, output_dir)
    print(f"Heterogeneity level statistics saved to: {stats_path}")
    
    print("\n" + "="*80)
    print("HETEROGENEITY LEVEL ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
