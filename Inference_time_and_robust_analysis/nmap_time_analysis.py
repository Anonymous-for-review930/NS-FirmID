import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Read and parse JSONL file
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

# Calculate average processing time (convert to milliseconds)
def calculate_average_time(data):
    times = []
    for entry in data:
        if 'nmap_consume_time' in entry:
            # Convert to milliseconds
            times.append(float(entry['nmap_consume_time']) * 1000)
    
    if times:
        average_time = sum(times) / len(times)
        return average_time, times
    else:
        return 0, []

# Analyze relationship between banner length and processing time (convert time to milliseconds)
def analyze_banner_time_relation(data):
    banner_lengths = []
    process_times = []
    
    for entry in data:
        if 'banner' in entry and 'nmap_consume_time' in entry:
            banner_length = len(str(entry['banner']))
            # Convert to milliseconds
            process_time = float(entry['nmap_consume_time']) * 1000
            banner_lengths.append(banner_length)
            process_times.append(process_time)
    
    return banner_lengths, process_times

# Main function
def main():
    # File paths
    input_file = '/Users/lululu/Documents/code/Multi-Agent_version_identification_using_LLMs/test_on_linux/inference_analysis_0216/test_set_nmap.jsonl'
    output_dir = '/Users/lululu/Documents/code/Multi-Agent_version_identification_using_LLMs/test_on_linux/inference_analysis_0216'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {input_file}...")
    data = load_nmap_data(input_file)
    print(f"Loaded {len(data)} entries")
    
    # Calculate average processing time
    average_time, times = calculate_average_time(data)
    print(f"Average nmap processing time: {average_time:.4f} ms")
    print(f"Total samples with time data: {len(times)}")
    
    # Analyze relationship
    banner_lengths, process_times = analyze_banner_time_relation(data)
    print(f"Samples with both banner and time data: {len(banner_lengths)}")
    
    # Calculate basic statistics
    if banner_lengths and process_times:
        max_banner_length = max(banner_lengths)
        min_banner_length = min(banner_lengths)
        avg_banner_length = sum(banner_lengths) / len(banner_lengths)
        
        print(f"Banner length statistics:")
        print(f"  Min: {min_banner_length}")
        print(f"  Max: {max_banner_length}")
        print(f"  Avg: {avg_banner_length:.2f}")
        
        # Calculate outlier boundaries (using IQR method)
        q1 = np.percentile(process_times, 25)
        q3 = np.percentile(process_times, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filter outliers
        filtered_times = [t for t in process_times if lower_bound <= t <= upper_bound]
        filtered_banner_lengths = [banner_lengths[i] for i, t in enumerate(process_times) if lower_bound <= t <= upper_bound]
        
        print(f"\nOutlier filtering:")
        print(f"  Original samples: {len(process_times)}")
        print(f"  Filtered samples: {len(filtered_times)}")
        print(f"  Outliers removed: {len(process_times) - len(filtered_times)}")
        print(f"  Time range: {lower_bound:.2f} ms to {upper_bound:.2f} ms")
        
        # Generate time frequency distribution plot (1ms bins)
        plt.figure(figsize=(10, 6))
        max_time = int(np.ceil(max(filtered_times)))
        bins = range(0, max_time + 1, 1)  # 1ms per bin
        plt.hist(filtered_times, bins=bins, alpha=0.7, color='blue', edgecolor='black')
        plt.title('Nmap Processing Time Distribution (No Outliers)')
        plt.xlabel('Processing Time (ms)')
        plt.ylabel('Frequency')
        plt.xticks(range(0, max_time + 1, 5))  # Display one tick label every 5ms
        plt.grid(True, linestyle='--', alpha=0.7)
        
        hist_path = os.path.join(output_dir, 'time_distribution_histogram.png')
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Time distribution histogram saved to: {hist_path}")
        
        # Generate box plot (no outliers)
        plt.figure(figsize=(10, 6))
        plt.boxplot(filtered_times, showfliers=False)  # Don't display outliers
        plt.title('Nmap Processing Time Box Plot (No Outliers)')
        plt.ylabel('Processing Time (ms)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        box_path = os.path.join(output_dir, 'time_distribution_boxplot.png')
        plt.savefig(box_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Time distribution box plot saved to: {box_path}")
        
        # Generate banner length vs processing time plot (no outliers)
        plt.figure(figsize=(10, 6))
        plt.scatter(filtered_banner_lengths, filtered_times, alpha=0.6, s=50)
        plt.title('Banner Length vs Nmap Processing Time (No Outliers)')
        plt.xlabel('Banner Length (characters)')
        plt.ylabel('Processing Time (ms)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        scatter_path = os.path.join(output_dir, 'banner_time_relation.png')
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Banner time relation plot saved to: {scatter_path}")
        
        # Generate time statistics (using filtered data)
        time_stats = {
            'mean': np.mean(filtered_times),
            'median': np.median(filtered_times),
            'std': np.std(filtered_times),
            'min': np.min(filtered_times),
            'max': np.max(filtered_times),
            'q1': np.percentile(filtered_times, 25),
            'q3': np.percentile(filtered_times, 75)
        }
        
        print("\nProcessing time statistics (No Outliers):")
        for key, value in time_stats.items():
            print(f"  {key}: {value:.6f} ms")
        
        # Save relation data to file for further analysis (using filtered data)
        relation_data_path = os.path.join(output_dir, 'banner_time_relation_data.json')
        relation_data = {
            'banner_lengths': filtered_banner_lengths,
            'process_times': filtered_times,
            'units': {
                'banner_lengths': 'characters',
                'process_times': 'ms'
            },
            'outlier_filtering': {
                'original_samples': len(process_times),
                'filtered_samples': len(filtered_times),
                'outliers_removed': len(process_times) - len(filtered_times),
                'time_range': {
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
            }
        }
        with open(relation_data_path, 'w', encoding='utf-8') as f:
            json.dump(relation_data, f, indent=2)
        print(f"Relation data saved to: {relation_data_path}")
    else:
        print("Insufficient data for relation analysis")

if __name__ == "__main__":
    main()
