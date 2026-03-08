import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def load_nmap_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    data.append(entry)
                except json.JSONDecodeError:
                    pass
    return data

def load_nsfirmid_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_nsfirmid_results(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    results.append(entry)
                except json.JSONDecodeError:
                    pass
    return results

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
    
    return sample_times, results

def calculate_nmap_stats(data):
    times = []
    for entry in data:
        if 'nmap_consume_time' in entry:
            times.append(float(entry['nmap_consume_time']) * 1000)
    
    q1 = np.percentile(times, 25) if times else 0
    q3 = np.percentile(times, 75) if times else 0
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_times = [t for t in times if lower_bound <= t <= upper_bound]
    
    return {
        'count': len(times),
        'mean': np.mean(times) if times else 0,
        'median': np.median(times) if times else 0,
        'std': np.std(times) if times else 0,
        'min': np.min(times) if times else 0,
        'max': np.max(times) if times else 0,
        'q1': q1,
        'q3': q3,
        'times': times,
        'filtered_times': filtered_times,
        'throughput': len(times) / (sum(times) / 1000) if times and sum(times) > 0 else 0
    }

def calculate_nsfirmid_stats(sample_times):
    q1 = np.percentile(sample_times, 25) if sample_times else 0
    q3 = np.percentile(sample_times, 75) if sample_times else 0
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_times = [t for t in sample_times if lower_bound <= t <= upper_bound]
    
    return {
        'count': len(sample_times),
        'mean': np.mean(sample_times) if sample_times else 0,
        'median': np.median(sample_times) if sample_times else 0,
        'std': np.std(sample_times) if sample_times else 0,
        'min': np.min(sample_times) if sample_times else 0,
        'max': np.max(sample_times) if sample_times else 0,
        'q1': q1,
        'q3': q3,
        'times': sample_times,
        'filtered_times': filtered_times,
        'throughput': len(sample_times) / (sum(sample_times) / 1000) if sample_times and sum(sample_times) > 0 else 0
    }

def calculate_accuracy_metrics(results):
    brand_tp = brand_fp = brand_fn = 0
    model_tp = model_fp = model_fn = 0
    version_tp = version_fp = version_fn = 0
    
    for result in results:
        flag = result.get('flag', [False, False, False])
        label = result.get('label', ['', '', ''])
        new_label = result.get('new_label', ['', '', ''])
        
        brand_match = flag[0] if len(flag) > 0 else False
        model_match = flag[1] if len(flag) > 1 else False
        version_match = flag[2] if len(flag) > 2 else False
        
        if brand_match:
            brand_tp += 1
        else:
            if label[0] and label[0] != 'np':
                brand_fn += 1
            if new_label[0]:
                brand_fp += 1
        
        if model_match:
            model_tp += 1
        else:
            if label[1] and label[1] != 'np':
                model_fn += 1
            if new_label[1]:
                model_fp += 1
        
        if version_match:
            version_tp += 1
        else:
            if label[2] and label[2] != 'nv':
                version_fn += 1
            if new_label[2]:
                version_fp += 1
    
    def calc_metrics(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1
    
    brand_p, brand_r, brand_f1 = calc_metrics(brand_tp, brand_fp, brand_fn)
    model_p, model_r, model_f1 = calc_metrics(model_tp, model_fp, model_fn)
    version_p, version_r, version_f1 = calc_metrics(version_tp, version_fp, version_fn)
    
    return {
        'brand': {'precision': brand_p, 'recall': brand_r, 'f1': brand_f1},
        'model': {'precision': model_p, 'recall': model_r, 'f1': model_f1},
        'version': {'precision': version_p, 'recall': version_r, 'f1': version_f1},
        'overall': {
            'precision': (brand_p + model_p + version_p) / 3,
            'recall': (brand_r + model_r + version_r) / 3,
            'f1': (brand_f1 + model_f1 + version_f1) / 3
        }
    }

def generate_summary_table(nmap_stats, nsfirmid_stats, accuracy_metrics, output_dir):
    table_data = {
        'Metrics': ['Sample Count', 'Average Time (ms)', 'Median Time (ms)', 'Max Time (ms)', 
                   'Throughput (samples/s)', 'Precision', 'Recall', 'F1 Score'],
        'NMAP': [
            nmap_stats['count'],
            f"{nmap_stats['mean']:.4f}",
            f"{nmap_stats['median']:.4f}",
            f"{nmap_stats['max']:.4f}",
            f"{nmap_stats['throughput']:.2f}",
            'N/A',
            'N/A',
            'N/A'
        ],
        'NS-FirmID': [
            nsfirmid_stats['count'],
            f"{nsfirmid_stats['mean']:.4f}",
            f"{nsfirmid_stats['median']:.4f}",
            f"{nsfirmid_stats['max']:.4f}",
            f"{nsfirmid_stats['throughput']:.2f}",
            f"{accuracy_metrics['overall']['precision']:.4f}",
            f"{accuracy_metrics['overall']['recall']:.4f}",
            f"{accuracy_metrics['overall']['f1']:.4f}"
        ]
    }
    
    table_path = os.path.join(output_dir, 'comparison_table.json')
    with open(table_path, 'w', encoding='utf-8') as f:
        json.dump(table_data, f, indent=2)
    
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"{'Metrics':<25} {'NMAP':<20} {'NS-FirmID':<20}")
    print("-"*80)
    for i, metric in enumerate(table_data['Metrics']):
        print(f"{metric:<25} {table_data['NMAP'][i]:<20} {table_data['NS-FirmID'][i]:<20}")
    print("="*80)
    
    return table_data, table_path

def generate_avg_time_plot(nmap_stats, nsfirmid_stats, output_dir):
    plt.figure(figsize=(10, 6))
    methods = ['NMAP', 'NS-FirmID']
    avg_times = [nmap_stats['mean'], nsfirmid_stats['mean']]
    bars = plt.bar(methods, avg_times, color=['#3498db', '#e74c3c'], alpha=0.8, width=0.5)
    plt.title('Average Processing Time Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Time (ms)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    for bar, val in zip(bars, avg_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.2f} ms', ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'avg_time_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_path

def generate_box_plot(nmap_stats, nsfirmid_stats, output_dir):
    plt.figure(figsize=(10, 6))
    box_data = [nmap_stats['filtered_times'], nsfirmid_stats['filtered_times']]
    plt.boxplot(box_data, tick_labels=['NMAP', 'NS-FirmID'], showfliers=False)
    plt.title('Processing Time Distribution (Box Plot, No Outliers)', fontsize=14, fontweight='bold')
    plt.ylabel('Time (ms)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'time_boxplot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_path

def generate_broken_axis_histogram(nmap_stats, nsfirmid_stats, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.subplots_adjust(wspace=0.05)
    
    nmap_filtered = nmap_stats['filtered_times']
    ns_filtered = nsfirmid_stats['filtered_times']
    
    if nmap_filtered:
        max_time_nmap = int(np.ceil(max(nmap_filtered)))
        bins_nmap = range(0, max_time_nmap + 1, 1)
        ax1.hist(nmap_filtered, bins=bins_nmap, alpha=0.7, color='#3498db', 
                edgecolor='black', label='NMAP')
        ax1.set_title('NMAP Time Distribution (1ms bins)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (ms)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_xlim(0, max_time_nmap + 1)
        ax1.set_xticks(range(0, max_time_nmap + 1, 5))
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper right', fontsize=10)
    
    if ns_filtered:
        min_time_ns = int(np.floor(min(ns_filtered)))
        max_time_ns = int(np.ceil(max(ns_filtered)))
        bins_ns = range(min_time_ns, max_time_ns + 1, 50)
        ax2.hist(ns_filtered, bins=bins_ns, alpha=0.7, color='#e74c3c', 
                edgecolor='black', label='NS-FirmID')
        ax2.set_title('NS-FirmID Time Distribution (50ms bins)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time (ms)', fontsize=12)
        ax2.set_xlim(min_time_ns - 50, max_time_ns + 50)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper right', fontsize=10)
    
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.yaxis.tick_left()
    ax2.yaxis.tick_right()
    
    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'time_distribution_broken_axis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_path

def generate_banner_time_plot(nmap_data, nsfirmid_stats, nsfirmid_results, output_dir):
    nmap_banner_times = []
    for entry in nmap_data:
        if 'banner' in entry and 'nmap_consume_time' in entry:
            nmap_banner_times.append({
                'banner_len': len(str(entry['banner'])),
                'time': float(entry['nmap_consume_time']) * 1000
            })
    
    ns_banner_times = []
    for i, result in enumerate(nsfirmid_results):
        ns_banner_times.append({
            'banner_len': result.get('banner_len', 0),
            'time': nsfirmid_stats['times'][i] if i < len(nsfirmid_stats['times']) else 0
        })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.subplots_adjust(wspace=0.05)
    
    nmap_q1 = np.percentile([d['time'] for d in nmap_banner_times], 25)
    nmap_q3 = np.percentile([d['time'] for d in nmap_banner_times], 75)
    nmap_iqr = nmap_q3 - nmap_q1
    nmap_lower = nmap_q1 - 1.5 * nmap_iqr
    nmap_upper = nmap_q3 + 1.5 * nmap_iqr
    nmap_filtered = [(d['banner_len'], d['time']) for d in nmap_banner_times 
                     if nmap_lower <= d['time'] <= nmap_upper]
    
    if nmap_filtered:
        banner_lens, times = zip(*nmap_filtered)
        ax1.scatter(banner_lens, times, alpha=0.6, s=30, color='#3498db')
        ax1.set_title('NMAP: Banner Length vs Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Banner Length (characters)', fontsize=12)
        ax1.set_ylabel('Time (ms)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        max_time = max(times)
        ax1.set_ylim(0, max_time + 5)
        ax1.set_yticks(range(0, int(max_time) + 10, 5))
    
    ns_q1 = np.percentile([d['time'] for d in ns_banner_times], 25)
    ns_q3 = np.percentile([d['time'] for d in ns_banner_times], 75)
    ns_iqr = ns_q3 - ns_q1
    ns_lower = ns_q1 - 1.5 * ns_iqr
    ns_upper = ns_q3 + 1.5 * ns_iqr
    ns_filtered = [(d['banner_len'], d['time']) for d in ns_banner_times 
                   if ns_lower <= d['time'] <= ns_upper]
    
    if ns_filtered:
        banner_lens, times = zip(*ns_filtered)
        ax2.scatter(banner_lens, times, alpha=0.6, s=30, color='#e74c3c')
        ax2.set_title('NS-FirmID: Banner Length vs Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Banner Length (characters)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        min_time = min(times)
        max_time = max(times)
        ax2.set_ylim(min_time - 50, max_time + 50)
        ax2.set_yticks(range(int(min_time), int(max_time) + 100, 50))
    
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.yaxis.tick_left()
    ax2.yaxis.tick_right()
    
    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'banner_time_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_path

def main():
    nmap_file = '/Users/lululu/Documents/code/Multi-Agent_version_identification_using_LLMs/test_on_linux/inference_analysis_0216/test_set_nmap.jsonl'
    nsfirmid_file = '/test_on_linux/inference_analysis_0216/test_set_for_plot_sft_qwen_0217.json'
    nsfirmid_results_file = '/test_on_linux/inference_analysis_0216/test_set_results_sft_qwen_0217.jsonl'
    output_dir = '/Users/lululu/Documents/code/Multi-Agent_version_identification_using_LLMs/test_on_linux/inference_analysis_0216'
    
    print("Loading NMAP data...")
    nmap_data = load_nmap_data(nmap_file)
    print(f"Loaded {len(nmap_data)} NMAP entries")
    
    print("\nLoading NS-FirmID data...")
    nsfirmid_data = load_nsfirmid_data(nsfirmid_file)
    print(f"Loaded NS-FirmID data with {len(nsfirmid_data.get('results', []))} results")
    
    print("\nLoading NS-FirmID results for accuracy calculation...")
    nsfirmid_results = load_nsfirmid_results(nsfirmid_results_file)
    print(f"Loaded {len(nsfirmid_results)} NS-FirmID results")
    
    print("\nEstimating NS-FirmID sample times...")
    nsfirmid_sample_times, nsfirmid_plot_results = estimate_sample_times(nsfirmid_data)
    
    print("\nCalculating statistics...")
    nmap_stats = calculate_nmap_stats(nmap_data)
    nsfirmid_stats = calculate_nsfirmid_stats(nsfirmid_sample_times)
    
    print("\nCalculating accuracy metrics...")
    accuracy_metrics = calculate_accuracy_metrics(nsfirmid_results)
    
    print("\nGenerating summary table...")
    table_data, table_path = generate_summary_table(nmap_stats, nsfirmid_stats, accuracy_metrics, output_dir)
    print(f"Table saved to: {table_path}")
    
    print("\nGenerating average time comparison plot...")
    plot_path = generate_avg_time_plot(nmap_stats, nsfirmid_stats, output_dir)
    print(f"Plot saved to: {plot_path}")
    
    print("\nGenerating box plot...")
    plot_path = generate_box_plot(nmap_stats, nsfirmid_stats, output_dir)
    print(f"Plot saved to: {plot_path}")
    
    print("\nGenerating broken-axis histogram...")
    plot_path = generate_broken_axis_histogram(nmap_stats, nsfirmid_stats, output_dir)
    print(f"Plot saved to: {plot_path}")
    
    print("\nGenerating banner-time comparison plot...")
    plot_path = generate_banner_time_plot(nmap_data, nsfirmid_stats, nsfirmid_plot_results, output_dir)
    print(f"Plot saved to: {plot_path}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
