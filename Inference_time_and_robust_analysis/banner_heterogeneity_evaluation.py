"""
Banner Heterogeneity Evaluation Experiment
Includes four experiments:
1. Heterogeneity metric statistical distribution
2. Heterogeneity clustering analysis
3. Heterogeneity level classification and radar chart
4. Representative sample display
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter
import re
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.dpi'] = 300

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
                    data[index] = {
                        'banner': banner,
                        'label': entry.get('label', entry.get('new_label', []))
                    }
                except json.JSONDecodeError:
                    pass
    return data

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

def count_version_like_patterns(text):
    patterns = [
        r'\b[vV]?\d+\.\d+(?:\.\d+)?(?:\.\d+)?(?:-[a-zA-Z0-9]+)?(?:\.[a-zA-Z0-9]+)*\b',
        r'\b[vV]\d+[a-zA-Z0-9\-\.]*\b',
        r'\bbuild\s*\d+\b',
        r'\brelease[_\-]?\d+(?:\.\d+)*\b',
    ]
    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        count += len(matches)
    return count

def count_ip_addresses(text):
    pattern = r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    return len(re.findall(pattern, text))

def count_chinese_chars(text):
    return len(re.findall(r'[\u4e00-\u9fff]', text))

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
            'version_like_count': 0,
            'ip_count': 0,
            'chinese_ratio': 0,
            'word_count': 0,
            'avg_word_length': 0
        }
    
    banner_str = str(banner)
    length = len(banner_str)
    
    entropy = calculate_shannon_entropy(banner_str)
    unique_chars = len(set(banner_str))
    
    digit_count = sum(1 for c in banner_str if c.isdigit())
    letter_count = sum(1 for c in banner_str if c.isalpha())
    chinese_count = count_chinese_chars(banner_str)
    special_count = length - digit_count - letter_count
    
    digit_ratio = digit_count / length if length > 0 else 0
    letter_ratio = letter_count / length if length > 0 else 0
    special_char_ratio = special_count / length if length > 0 else 0
    chinese_ratio = chinese_count / length if length > 0 else 0
    
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
    
    version_like_count = count_version_like_patterns(banner_str)
    ip_count = count_ip_addresses(banner_str)
    
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
        'version_like_count': version_like_count,
        'ip_count': ip_count,
        'chinese_ratio': chinese_ratio,
        'word_count': word_count,
        'avg_word_length': avg_word_length
    }

def experiment1_statistics(heterogeneity_data, output_dir):
    print("\n" + "="*100)
    print("Experiment 1: Heterogeneity Metric Statistical Distribution")
    print("="*100)
    
    metrics = ['length', 'entropy', 'unique_chars', 'digit_ratio', 'letter_ratio', 
               'special_char_ratio', 'line_count', 'structure_score', 
               'version_like_count', 'ip_count', 'chinese_ratio', 'word_count']
    
    metric_labels = {
        'length': 'Banner Length',
        'entropy': 'Shannon Entropy',
        'unique_chars': 'Unique Chars',
        'digit_ratio': 'Digit Ratio',
        'letter_ratio': 'Letter Ratio',
        'special_char_ratio': 'Special Char Ratio',
        'line_count': 'Line Count',
        'structure_score': 'Structure Score',
        'version_like_count': 'Version-like Count',
        'ip_count': 'IP Address Count',
        'chinese_ratio': 'Chinese Ratio',
        'word_count': 'Word Count'
    }
    
    stats_table = []
    for metric in metrics:
        values = [d[metric] for d in heterogeneity_data]
        values = np.array(values)
        
        stats_table.append({
            'Metric': metric_labels[metric],
            'Mean': np.mean(values),
            'Median': np.median(values),
            'Std': np.std(values),
            'Min': np.min(values),
            'Q25': np.percentile(values, 25),
            'Q50': np.percentile(values, 50),
            'Q75': np.percentile(values, 75),
            'Q95': np.percentile(values, 95),
            'Max': np.max(values)
        })
    
    df_stats = pd.DataFrame(stats_table)
    print(df_stats.to_string(index=False))
    
    stats_path = os.path.join(output_dir, 'heterogeneity_statistics.csv')
    df_stats.to_csv(stats_path, index=False)
    print(f"\nStatistics table saved: {stats_path}")
    
    key_metrics = ['length', 'entropy', 'digit_ratio', 'structure_score', 'version_like_count', 'ip_count']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, metric in enumerate(key_metrics):
        ax = axes[idx // 3, idx % 3]
        values = [d[metric] for d in heterogeneity_data]
        
        ax.hist(values, bins=50, color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_xlabel(metric_labels[metric], fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{metric_labels[metric]} Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        mean_val = np.mean(values)
        median_val = np.median(values)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle=':', linewidth=2, label=f'Median: {median_val:.2f}')
        ax.legend(fontsize=9)
    
    plt.suptitle('Banner Heterogeneity Metrics Distribution', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'heterogeneity_distribution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Distribution plot saved: {plot_path}")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for idx, metric in enumerate(key_metrics):
        ax = axes[idx // 3, idx % 3]
        values = [d[metric] for d in heterogeneity_data]
        
        bp = ax.boxplot(values, patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][0].set_alpha(0.7)
        
        ax.set_ylabel(metric_labels[metric], fontsize=11)
        ax.set_title(f'{metric_labels[metric]} Box Plot', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')
        ax.set_xticklabels([''])
    
    plt.suptitle('Banner Heterogeneity Metrics Box Plots', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'heterogeneity_boxplots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Box plot saved: {plot_path}")
    
    return df_stats

def experiment2_clustering(heterogeneity_data, output_dir):
    print("\n" + "="*100)
    print("Experiment 2: Heterogeneity Clustering Analysis")
    print("="*100)
    
    clustering_features = ['length', 'entropy', 'digit_ratio', 'structure_score', 
                          'version_like_count', 'ip_count', 'line_count']
    
    X = np.array([[d[f] for f in clustering_features] for d in heterogeneity_data])
    
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    K_range = range(2, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Inertia', fontsize=12)
    ax.set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plot_path = os.path.join(output_dir, 'clustering_elbow.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Elbow method plot saved: {plot_path}")
    
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    print(f"\nPerforming clustering with K={optimal_k}...")
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i in range(optimal_k):
        mask = cluster_labels == i
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                  c=colors[i], alpha=0.6, s=20, label=f'Cluster {i+1}')
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('Banner Heterogeneity Clustering (t-SNE)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plot_path = os.path.join(output_dir, 'clustering_tsne.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Clustering visualization plot saved: {plot_path}")
    
    cluster_stats = []
    for i in range(optimal_k):
        mask = cluster_labels == i
        cluster_data = [heterogeneity_data[j] for j in range(len(heterogeneity_data)) if mask[j]]
        
        stats = {'Cluster': f'Cluster {i+1}', 'Count': len(cluster_data)}
        for f in clustering_features:
            values = [d[f] for d in cluster_data]
            stats[f'{f}_mean'] = np.mean(values)
        cluster_stats.append(stats)
    
    df_cluster = pd.DataFrame(cluster_stats)
    print("\nClustering statistics:")
    print(df_cluster.to_string(index=False))
    
    cluster_path = os.path.join(output_dir, 'clustering_statistics.csv')
    df_cluster.to_csv(cluster_path, index=False)
    print(f"\nClustering statistics saved: {cluster_path}")
    
    return cluster_labels, df_cluster

def experiment3_radar_chart(heterogeneity_data, output_dir):
    print("\n" + "="*100)
    print("Experiment 3: Heterogeneity Level Classification and Radar Chart")
    print("="*100)
    
    score_features = ['length', 'entropy', 'structure_score', 'version_like_count', 'ip_count']
    
    normalized_scores = {}
    for f in score_features:
        values = np.array([d[f] for d in heterogeneity_data])
        min_val, max_val = np.min(values), np.max(values)
        if max_val > min_val:
            normalized_scores[f] = (values - min_val) / (max_val - min_val)
        else:
            normalized_scores[f] = np.zeros_like(values)
    
    composite_scores = np.zeros(len(heterogeneity_data))
    for f in score_features:
        composite_scores += normalized_scores[f]
    composite_scores /= len(score_features)
    
    q33 = np.percentile(composite_scores, 33)
    q66 = np.percentile(composite_scores, 66)
    
    levels = []
    for score in composite_scores:
        if score <= q33:
            levels.append('Low')
        elif score <= q66:
            levels.append('Medium')
        else:
            levels.append('High')
    
    print(f"\n异构性等级划分阈值:")
    print(f"  Low: score <= {q33:.4f}")
    print(f"  Medium: {q33:.4f} < score <= {q66:.4f}")
    print(f"  High: score > {q66:.4f}")
    
    level_counts = {'Low': levels.count('Low'), 'Medium': levels.count('Medium'), 'High': levels.count('High')}
    print(f"\n各等级样本数:")
    for level, count in level_counts.items():
        print(f"  {level}: {count} ({count/len(levels)*100:.1f}%)")
    
    radar_features = ['length', 'entropy', 'digit_ratio', 'structure_score', 
                     'version_like_count', 'ip_count', 'line_count']
    radar_labels = ['Length', 'Entropy', 'Digit\nRatio', 'Structure\nScore', 
                   'Version\nCount', 'IP\nCount', 'Line\nCount']
    
    level_stats = {}
    for level in ['Low', 'Medium', 'High']:
        indices = [i for i, l in enumerate(levels) if l == level]
        level_stats[level] = {}
        for f in radar_features:
            values = [heterogeneity_data[i][f] for i in indices]
            level_stats[level][f] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
    angles += angles[:1]
    
    colors = {'Low': '#3498db', 'Medium': '#f39c12', 'High': '#e74c3c'}
    
    for level in ['Low', 'Medium', 'High']:
        values = []
        for f in radar_features:
            all_values = [d[f] for d in heterogeneity_data]
            max_val = max(all_values)
            if max_val > 0:
                normalized = level_stats[level][f]['mean'] / max_val
            else:
                normalized = 0
            values.append(normalized)
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=f'{level} (n={level_counts[level]})', 
               color=colors[level], markersize=6)
        ax.fill(angles, values, alpha=0.25, color=colors[level])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('Banner Heterogeneity Level Comparison (Radar Chart)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)
    
    plot_path = os.path.join(output_dir, 'heterogeneity_radar.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n雷达图已保存: {plot_path}")
    
    radar_table = []
    for level in ['Low', 'Medium', 'High']:
        row = {'Level': level, 'Count': level_counts[level]}
        for f in radar_features:
            row[f'{f}_mean'] = level_stats[level][f]['mean']
            row[f'{f}_std'] = level_stats[level][f]['std']
        radar_table.append(row)
    
    df_radar = pd.DataFrame(radar_table)
    radar_path = os.path.join(output_dir, 'heterogeneity_level_stats.csv')
    df_radar.to_csv(radar_path, index=False)
    print(f"等级统计已保存: {radar_path}")
    
    return levels, level_stats

def experiment4_examples(banners_data, heterogeneity_data, levels, output_dir):
    print("\n" + "="*100)
    print("Experiment 4: Representative Sample Display")
    print("="*100)
    
    sample_indices = list(banners_data.keys())
    
    examples = {'Low': [], 'Medium': [], 'High': []}
    
    for level in ['Low', 'Medium', 'High']:
        level_indices = [i for i, l in enumerate(levels) if l == level]
        
        sorted_indices = sorted(level_indices, 
                               key=lambda i: heterogeneity_data[i]['length'], 
                               reverse=True)[:5]
        
        for idx in sorted_indices:
            sample_id = sample_indices[idx]
            banner_info = banners_data[sample_id]
            het = heterogeneity_data[idx]
            
            examples[level].append({
                'sample_id': sample_id,
                'banner': banner_info['banner'][:300] + ('...' if len(banner_info['banner']) > 300 else ''),
                'banner_full': banner_info['banner'],
                'label': banner_info['label'],
                'length': het['length'],
                'entropy': het['entropy'],
                'structure_score': het['structure_score'],
                'version_like_count': het['version_like_count'],
                'ip_count': het['ip_count']
            })
    
    for level in ['Low', 'Medium', 'High']:
        print(f"\n{'='*80}")
        print(f"{level} Heterogeneity Examples")
        print(f"{'='*80}")
        
        for i, ex in enumerate(examples[level][:3]):
            print(f"\n--- Example {i+1}: {ex['sample_id']} ---")
            print(f"Length: {ex['length']}, Entropy: {ex['entropy']:.2f}, Structure: {ex['structure_score']:.2f}")
            print(f"Version-like: {ex['version_like_count']}, IP Count: {ex['ip_count']}")
            print(f"Label: {ex['label']}")
            print(f"Banner Preview:\n{ex['banner'][:200]}...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    
    for idx, level in enumerate(['Low', 'Medium', 'High']):
        ax = axes[idx]
        ex = examples[level][0]
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        ax.add_patch(plt.Rectangle((0.2, 0.2), 9.6, 9.6, fill=True, 
                                    facecolor='#f8f9fa', edgecolor='#3498db', linewidth=2))
        
        ax.text(5, 9.5, f'{level} Heterogeneity Example', fontsize=13, fontweight='bold', 
                ha='center', va='top', color='#2c3e50')
        
        ax.text(0.5, 8.5, f"Sample: {ex['sample_id']}", fontsize=10, ha='left', va='top')
        ax.text(0.5, 7.8, f"Length: {ex['length']} | Entropy: {ex['entropy']:.2f} | Structure: {ex['structure_score']:.2f}", 
               fontsize=9, ha='left', va='top')
        ax.text(0.5, 7.2, f"Version-like: {ex['version_like_count']} | IP Count: {ex['ip_count']}", 
               fontsize=9, ha='left', va='top')
        
        ax.add_patch(plt.Rectangle((0.5, 0.5), 9, 6.4, fill=True, 
                                    facecolor='white', edgecolor='#dee2e6', linewidth=1))
        
        banner_lines = ex['banner_full'][:600].split('\n')[:10]
        y_pos = 6.5
        for line in banner_lines:
            if len(line) > 85:
                line = line[:82] + '...'
            ax.text(0.7, y_pos, line, fontsize=8, ha='left', va='top', 
                   fontfamily='monospace', color='#495057')
            y_pos -= 0.6
            if y_pos < 1:
                break
    
    plt.suptitle('Banner Heterogeneity Examples by Level', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'heterogeneity_examples.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n示例图已保存: {plot_path}")
    
    examples_path = os.path.join(output_dir, 'heterogeneity_examples.json')
    with open(examples_path, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False, default=str)
    print(f"示例数据已保存: {examples_path}")
    
    return examples

def main():
    output_dir = '/Users/lululu/Documents/code/Multi-Agent_version_identification_using_LLMs/test_on_linux/inference_analysis_0216'
    
    print("Loading data...")
    test_set_path = '/Users/lululu/Documents/code/Multi-Agent_version_identification_using_LLMs/data/test_set.jsonl'
    holdout_set_path = '/Users/lululu/Documents/code/Multi-Agent_version_identification_using_LLMs/data/holdout_set.jsonl'
    
    test_data = load_jsonl_data(test_set_path)
    holdout_data = load_jsonl_data(holdout_set_path)
    all_data = {**test_data, **holdout_data}
    
    print(f"Test set sample count: {len(test_data)}")
    print(f"Holdout set sample count: {len(holdout_data)}")
    print(f"Total sample count: {len(all_data)}")
    
    print("\nCalculating heterogeneity metrics...")
    heterogeneity_data = []
    sample_ids = []
    for sample_id, data in all_data.items():
        metrics = calculate_heterogeneity_metrics(data['banner'])
        heterogeneity_data.append(metrics)
        sample_ids.append(sample_id)
    
    print(f"Calculation completed, {len(heterogeneity_data)} samples in total")
    
    df_stats = experiment1_statistics(heterogeneity_data, output_dir)
    
    cluster_labels, df_cluster = experiment2_clustering(heterogeneity_data, output_dir)
    
    levels, level_stats = experiment3_radar_chart(heterogeneity_data, output_dir)
    
    examples = experiment4_examples(all_data, heterogeneity_data, levels, output_dir)
    
    print("\n" + "="*100)
    print("Banner异构性评估实验完成！")
    print("="*100)
    print(f"\n输出文件:")
    print(f"  - heterogeneity_statistics.csv")
    print(f"  - heterogeneity_distribution.png")
    print(f"  - heterogeneity_boxplots.png")
    print(f"  - clustering_elbow.png")
    print(f"  - clustering_tsne.png")
    print(f"  - clustering_statistics.csv")
    print(f"  - heterogeneity_radar.png")
    print(f"  - heterogeneity_level_stats.csv")
    print(f"  - heterogeneity_examples.png")
    print(f"  - heterogeneity_examples.json")

if __name__ == "__main__":
    main()
