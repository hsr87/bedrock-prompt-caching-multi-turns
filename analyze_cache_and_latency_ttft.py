import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Set result folders and date prefix
folders = ['37_250627_ttft', '37_250627_1p_ttft']
date = '250630'

def calculate_generation_time_differences():
    """
    Calculate turn-by-turn averages of Generation Time (Latency - TTFT) for each folder
    and calculate percentage differences compared to the baseline folder
    """
    
    # Dictionary to store turn-by-turn averages for each folder
    folder_metrics = {}
    
    for folder in folders:
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        
        # Dictionary to store generation time values for each turn
        turn_gen_times = {}
        
        print(f"\nAnalyzing {folder} folder...")
        print(f"Found CSV files: {len(csv_files)}")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                for _, row in df.iterrows():
                    turn = row['turn']
                    latency = row['invocation_latency']
                    ttft = row['ttft']
                    generation_time = latency - ttft  # Calculate generation time
                    
                    if turn not in turn_gen_times:
                        turn_gen_times[turn] = []
                    
                    turn_gen_times[turn].append(generation_time)
                    
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
        
        # Calculate average for each turn
        turns = sorted(turn_gen_times.keys())
        gen_time_means = {}
        
        for turn in turns:
            gen_time_means[turn] = np.mean(turn_gen_times[turn])
        
        folder_metrics[folder] = {
            'gen_time_means': gen_time_means,
            'turns': turns
        }
    
    # Set the first folder as the baseline folder
    base_folder = folders[0]
    
    print(f"\n=== Turn-by-turn Average Percentage Differences in Generation Time (Latency - TTFT) (Baseline: {base_folder}) ===")
    
    base_metrics = folder_metrics[base_folder]['gen_time_means']
    turns = folder_metrics[base_folder]['turns']
    
    # Print header
    header = "Turn | " + " | ".join([f"{folder:>25s}" for folder in folders])
    print(header)
    print("-" * len(header))
    
    # Compare each turn
    for turn in turns:
        base_value = base_metrics[turn]
        row = f"{turn:4d} | {base_value:25.3f}s"
        
        for i, folder in enumerate(folders[1:], 1):
            if turn in folder_metrics[folder]['gen_time_means']:
                current_value = folder_metrics[folder]['gen_time_means'][turn]
                percent_diff = ((current_value - base_value) / base_value) * 100
                row += f" | {current_value:8.3f}s ({percent_diff:+6.1f}%)"
            else:
                row += f" | {'N/A':>25s}"
        
        print(row)
    
    # Overall average percentage differences
    print("\nOverall Average Generation Time Differences:")
    all_avg_diffs = {}
    for i, folder in enumerate(folders[1:], 1):
        all_diffs = []
        for turn in turns:
            if turn in folder_metrics[folder]['gen_time_means']:
                base_value = base_metrics[turn]
                current_value = folder_metrics[folder]['gen_time_means'][turn]
                percent_diff = ((current_value - base_value) / base_value) * 100
                all_diffs.append(percent_diff)
        
        if all_diffs:
            avg_diff = np.mean(all_diffs)
            all_avg_diffs[folder] = avg_diff
            print(f"  {folder} vs {base_folder}: {avg_diff:+.2f}%")
    
    return folder_metrics, all_avg_diffs

def calculate_percentage_differences():
    """
    Calculate turn-by-turn averages of latency, ttft, and Milliseconds per token for each folder
    and calculate percentage differences compared to the baseline folder
    """
    
    # Dictionary to store turn-by-turn averages for each folder
    folder_metrics = {}
    
    for folder in folders:
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        
        # Dictionary to store latency, ttft, and Milliseconds per token values for each turn
        turn_latencies = {}
        turn_ttfts = {}
        turn_tokens_per_sec = {}
        
        print(f"\nAnalyzing {folder} folder...")
        print(f"Found CSV files: {len(csv_files)}")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                for _, row in df.iterrows():
                    turn = row['turn']
                    latency = row['invocation_latency']
                    ttft = row['ttft']
                    output_tokens = row['output_tokens']
                    
                    generation_time = latency - ttft
                    
                    # Prevent division by zero
                    if generation_time > 0:
                        tokens_per_sec = generation_time / output_tokens
                    else:
                        tokens_per_sec = 0
                    
                    if turn not in turn_latencies:
                        turn_latencies[turn] = []
                        turn_ttfts[turn] = []
                        turn_tokens_per_sec[turn] = []
                    
                    turn_latencies[turn].append(latency)
                    turn_ttfts[turn].append(ttft)
                    turn_tokens_per_sec[turn].append(tokens_per_sec)
                    
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
        
        # Calculate average for each turn
        turns = sorted(turn_latencies.keys())
        latency_means = {}
        ttft_means = {}
        tokens_per_sec_means = {}
        
        for turn in turns:
            latency_means[turn] = np.mean(turn_latencies[turn])
            ttft_means[turn] = np.mean(turn_ttfts[turn])
            tokens_per_sec_means[turn] = np.mean(turn_tokens_per_sec[turn])
        
        folder_metrics[folder] = {
            'latency_means': latency_means,
            'ttft_means': ttft_means,
            'ms_per_tok_means': tokens_per_sec_means,
            'turns': turns
        }
    
    # Set the first folder as the baseline folder
    base_folder = folders[0]
    
    print(f"\n=== Turn-by-turn Average Percentage Differences (Baseline: {base_folder}) ===")
    
    # Compare each metric
    for metric_name, metric_key in [('Invocation Latency', 'latency_means'), 
                                     ('TTFT', 'ttft_means'),
                                     ('Milliseconds per token', 'ms_per_tok_means')]:
        print(f"\n--- {metric_name} ---")
        
        base_metrics = folder_metrics[base_folder][metric_key]
        turns = folder_metrics[base_folder]['turns']
        
        # Print header
        header = "Turn | " + " | ".join([f"{folder:>20s}" for folder in folders])
        print(header)
        print("-" * len(header))
        
        # Compare each turn
        for turn in turns:
            base_value = base_metrics[turn]
            if metric_name == 'Milliseconds per token':
                row = f"{turn:4d} | {base_value:20.2f} ms/tok"
            else:
                row = f"{turn:4d} | {base_value:20.3f}s"
            
            for i, folder in enumerate(folders[1:], 1):
                if turn in folder_metrics[folder][metric_key]:
                    current_value = folder_metrics[folder][metric_key][turn]
                    if base_value != 0:
                        percent_diff = ((current_value - base_value) / base_value) * 100
                    else:
                        percent_diff = 0
                    
                    if metric_name == 'Milliseconds per token':
                        row += f" | {current_value:8.2f} ms/tok ({percent_diff:+6.1f}%)"
                    else:
                        row += f" | {current_value:8.3f}s ({percent_diff:+6.1f}%)"
                else:
                    row += f" | {'N/A':>20s}"
            
            print(row)
        
        # Overall average percentage differences
        print("\nOverall Average Differences:")
        for i, folder in enumerate(folders[1:], 1):
            all_diffs = []
            for turn in turns:
                if turn in folder_metrics[folder][metric_key]:
                    base_value = base_metrics[turn]
                    current_value = folder_metrics[folder][metric_key][turn]
                    if base_value != 0:
                        percent_diff = ((current_value - base_value) / base_value) * 100
                        all_diffs.append(percent_diff)
            
            if all_diffs:
                avg_diff = np.mean(all_diffs)
                print(f"  {folder} vs {base_folder}: {avg_diff:+.2f}%")
    
    # Generate summary table
    print("\n=== Summary: Overall Average Percentage Differences ===")
    print(f"Baseline: {base_folder}")
    print("\nFolder Comparison          | Latency Diff | TTFT Diff | Generation Time Diff | Tokens/Sec Diff")
    print("-" * 95)
    
    for folder in folders[1:]:
        # Calculate each metric difference
        latency_diffs = []
        ttft_diffs = []
        gen_time_diffs = []
        tokens_per_sec_diffs = []
        
        for turn in turns:
            if turn in folder_metrics[folder]['latency_means']:
                base_lat = folder_metrics[base_folder]['latency_means'][turn]
                curr_lat = folder_metrics[folder]['latency_means'][turn]
                latency_diffs.append(((curr_lat - base_lat) / base_lat) * 100)
                
                base_ttft = folder_metrics[base_folder]['ttft_means'][turn]
                curr_ttft = folder_metrics[folder]['ttft_means'][turn]
                ttft_diffs.append(((curr_ttft - base_ttft) / base_ttft) * 100)
                
                # Calculate Generation Time differences
                base_gen_time = base_lat - base_ttft
                curr_gen_time = curr_lat - curr_ttft
                if base_gen_time != 0:
                    gen_time_diffs.append(((curr_gen_time - base_gen_time) / base_gen_time) * 100)
                
                # Calculate Milliseconds per token differences
                base_tokens_per_sec = folder_metrics[base_folder]['ms_per_tok_means'][turn]
                curr_tokens_per_sec = folder_metrics[folder]['ms_per_tok_means'][turn]
                if base_tokens_per_sec != 0:
                    tokens_per_sec_diffs.append(((curr_tokens_per_sec - base_tokens_per_sec) / base_tokens_per_sec) * 100)
        
        avg_lat_diff = np.mean(latency_diffs) if latency_diffs else 0
        avg_ttft_diff = np.mean(ttft_diffs) if ttft_diffs else 0
        avg_gen_time_diff = np.mean(gen_time_diffs) if gen_time_diffs else 0
        avg_tokens_per_sec_diff = np.mean(tokens_per_sec_diffs) if tokens_per_sec_diffs else 0
        
        print(f"{folder:25s} | {avg_lat_diff:+11.2f}% | {avg_ttft_diff:+9.2f}% | {avg_gen_time_diff:+18.2f}% | {avg_tokens_per_sec_diff:+14.2f}%")

def plot_comparison_metrics():
    """
    Read CSV files from folders and generate comparison graphs 
    for Invocation Latency, TTFT, and Milliseconds per token
    """
    
    # Create three subplots (Invocation Latency, TTFT, Milliseconds per token)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Color settings
    colors = ['blue', 'red', 'green']
    
    for idx, folder in enumerate(folders):
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        
        # Dictionary to store latency, ttft, and Milliseconds per token values for each turn
        turn_latencies = {}
        turn_ttfts = {}
        turn_tokens_per_sec = {}
        
        print(f"\nAnalyzing {folder} folder...")
        print(f"Found CSV files: {len(csv_files)}")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                for _, row in df.iterrows():
                    turn = row['turn']
                    latency = row['invocation_latency']
                    ttft = row['ttft']
                    output_tokens = row['output_tokens']
                    
                    generation_time = latency - ttft
                    
                    # Prevent division by zero
                    if generation_time > 0:
                        tokens_per_sec = generation_time / output_tokens 
                    else:
                        tokens_per_sec = 0
                    
                    if turn not in turn_latencies:
                        turn_latencies[turn] = []
                        turn_ttfts[turn] = []
                        turn_tokens_per_sec[turn] = []
                    
                    turn_latencies[turn].append(latency)
                    turn_ttfts[turn].append(ttft)
                    turn_tokens_per_sec[turn].append(tokens_per_sec)
                    
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
        
        # Calculate average and standard deviation for each turn
        turns = sorted(turn_latencies.keys())
        latency_means = []
        latency_stds = []
        ttft_means = []
        ttft_stds = []
        tokens_per_sec_means = []
        tokens_per_sec_stds = []
        
        for turn in turns:
            # Invocation Latency
            latencies = turn_latencies[turn]
            latency_means.append(np.mean(latencies))
            latency_stds.append(np.std(latencies))
            
            # TTFT
            ttfts = turn_ttfts[turn]
            ttft_means.append(np.mean(ttfts))
            ttft_stds.append(np.std(ttfts))
            
            # Milliseconds per token
            tokens_per_sec = turn_tokens_per_sec[turn]
            tokens_per_sec_means.append(np.mean(tokens_per_sec))
            tokens_per_sec_stds.append(np.std(tokens_per_sec))
            
            print(f"  Turn {turn}: {len(latencies)} data points")
        
        # Invocation Latency graph
        ax1.errorbar(turns, latency_means, yerr=latency_stds, 
                    marker='o', capsize=5, color=colors[idx],
                    label=folder, linewidth=2, markersize=6, alpha=0.8)
        
        # TTFT graph
        ax2.errorbar(turns, ttft_means, yerr=ttft_stds, 
                    marker='s', capsize=5, color=colors[idx],
                    label=folder, linewidth=2, markersize=6, alpha=0.8)
        
        # Milliseconds per token graph
        ax3.errorbar(turns, tokens_per_sec_means, yerr=tokens_per_sec_stds, 
                    marker='^', capsize=5, color=colors[idx],
                    label=folder, linewidth=2, markersize=6, alpha=0.8)
    
    # Invocation Latency graph settings
    ax1.set_xlabel('Turn', fontsize=12)
    ax1.set_ylabel('Invocation Latency (seconds)', fontsize=12)
    ax1.set_title('Invocation Latency Comparison', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, 11))
    
    # TTFT graph settings
    ax2.set_xlabel('Turn', fontsize=12)
    ax2.set_ylabel('TTFT (seconds)', fontsize=12)
    ax2.set_title('Time To First Token (TTFT) Comparison', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(1, 11))
    
    # Milliseconds per token graph settings
    ax3.set_xlabel('Turn', fontsize=12)
    ax3.set_ylabel('Milliseconds per token', fontsize=12)
    ax3.set_title('Milliseconds per output token Comparison', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(1, 11))
    
    # Overall graph title
    fig.suptitle('Performance Comparison', fontsize=16)
    
    # Save and display graph
    plt.tight_layout()
    plt.savefig(f'performance_comparison_{date}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nGraph saved as 'performance_comparison_{date}.png'.")
    
    # Print statistics summary
    print_statistics_summary(folders)

def print_statistics_summary(folders):
    """Print key statistics summary for each folder"""
    print("\n=== Statistics Summary ===")
    
    for folder in folders:
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        
        all_latencies = []
        all_ttfts = []
        all_tokens_per_sec = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                all_latencies.extend(df['invocation_latency'].values)
                all_ttfts.extend(df['ttft'].values)
                
                # Calculate Milliseconds per token
                for _, row in df.iterrows():
                    latency = row['invocation_latency']
                    ttft = row['ttft']
                    output_tokens = row['output_tokens']
                    generation_time = latency - ttft
                    
                    if generation_time > 0:
                        tokens_per_sec = output_tokens / generation_time
                        all_tokens_per_sec.append(tokens_per_sec)
                
            except Exception as e:
                continue
        
        print(f"\n{folder}:")
        print(f"  Invocation Latency - Mean: {np.mean(all_latencies):.3f}s, Std: {np.std(all_latencies):.3f}s")
        print(f"  TTFT - Mean: {np.mean(all_ttfts):.3f}s, Std: {np.std(all_ttfts):.3f}s")
        if all_tokens_per_sec:
            print(f"  Milliseconds per token - Mean: {np.mean(all_tokens_per_sec):.2f} tok/s, Std: {np.std(all_tokens_per_sec):.2f} tok/s")

def plot_cache_metrics():
    """Cache-related metrics comparison graph"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['blue', 'red', 'green']
    
    for idx, folder in enumerate(folders):
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        
        turn_cache_creation = {}
        turn_cache_read = {}
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                for _, row in df.iterrows():
                    turn = row['turn']
                    cache_creation = row['cache_creation_input_tokens']
                    cache_read = row['cache_read_input_tokens']
                    
                    if turn not in turn_cache_creation:
                        turn_cache_creation[turn] = []
                        turn_cache_read[turn] = []
                    
                    turn_cache_creation[turn].append(cache_creation)
                    turn_cache_read[turn].append(cache_read)
                    
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
        
        turns = sorted(turn_cache_creation.keys())
        cache_creation_means = [np.mean(turn_cache_creation[turn]) for turn in turns]
        cache_read_means = [np.mean(turn_cache_read[turn]) for turn in turns]
        
        # Cache Creation graph
        ax1.plot(turns, cache_creation_means, marker='o', color=colors[idx],
                label=folder, linewidth=2, markersize=6, alpha=0.8)
        
        # Cache Read graph
        ax2.plot(turns, cache_read_means, marker='s', color=colors[idx],
                label=folder, linewidth=2, markersize=6, alpha=0.8)
    
    # Cache Creation graph settings
    ax1.set_xlabel('Turn', fontsize=12)
    ax1.set_ylabel('Cache Creation Input Tokens', fontsize=12)
    ax1.set_title('Cache Creation Tokens by Turn', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, 11))
    
    # Cache Read graph settings
    ax2.set_xlabel('Turn', fontsize=12)
    ax2.set_ylabel('Cache Read Input Tokens', fontsize=12)
    ax2.set_title('Cache Read Tokens by Turn', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(1, 11))
    
    fig.suptitle('Cache Metrics Comparison', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f'cache_metrics_comparison_{date}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nCache metrics graph saved as 'cache_metrics_comparison_{date}.png'.")

def plot_generation_time_and_tokens_comparison():
    """
    Create comparison graphs for Generation Time (Latency - TTFT) and Milliseconds per token
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color settings
    colors = ['blue', 'red', 'green']
    
    for idx, folder in enumerate(folders):
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        
        # Dictionary to store generation time and Milliseconds per token values for each turn
        turn_gen_times = {}
        turn_tokens_per_sec = {}
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                for _, row in df.iterrows():
                    turn = row['turn']
                    latency = row['invocation_latency']
                    ttft = row['ttft']
                    output_tokens = row['output_tokens']
                    gen_time = latency - ttft
                    
                    # Prevent division by zero
                    if gen_time > 0:
                        tokens_per_sec = output_tokens / gen_time
                    else:
                        tokens_per_sec = 0
                    
                    if turn not in turn_gen_times:
                        turn_gen_times[turn] = []
                        turn_tokens_per_sec[turn] = []
                    
                    turn_gen_times[turn].append(gen_time)
                    turn_tokens_per_sec[turn].append(tokens_per_sec)
                    
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
        
        # Calculate average and standard deviation for each turn
        turns = sorted(turn_gen_times.keys())
        gen_time_means = []
        gen_time_stds = []
        tokens_per_sec_means = []
        tokens_per_sec_stds = []
        
        for turn in turns:
            gen_times = turn_gen_times[turn]
            gen_time_means.append(np.mean(gen_times))
            gen_time_stds.append(np.std(gen_times))
            
            tokens_per_sec = turn_tokens_per_sec[turn]
            tokens_per_sec_means.append(np.mean(tokens_per_sec))
            tokens_per_sec_stds.append(np.std(tokens_per_sec))
        
        # Generation Time graph
        ax1.errorbar(turns, gen_time_means, yerr=gen_time_stds, 
                    marker='D', capsize=5, color=colors[idx],
                    label=folder, linewidth=2, markersize=6, alpha=0.8)
        
        # Milliseconds per token graph
        ax2.errorbar(turns, tokens_per_sec_means, yerr=tokens_per_sec_stds, 
                    marker='^', capsize=5, color=colors[idx],
                    label=folder, linewidth=2, markersize=6, alpha=0.8)
    
    # Generation Time graph settings
    ax1.set_xlabel('Turn', fontsize=12)
    ax1.set_ylabel('Generation Time (seconds)', fontsize=12)
    ax1.set_title('Generation Time (Latency - TTFT) Comparison', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, 11))
    
    # Milliseconds per token graph settings
    ax2.set_xlabel('Turn', fontsize=12)
    ax2.set_ylabel('Milliseconds per token', fontsize=12)
    ax2.set_title('Milliseconds per token Comparison', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(1, 11))
    
    fig.suptitle('Generation Time and Milliseconds per token Comparison', fontsize=16)
    
    # Save and display graph
    plt.tight_layout()
    plt.savefig(f'generation_time_tokens_comparison_{date}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nGeneration Time & Milliseconds per token graph saved as 'generation_time_tokens_comparison_{date}.png'.")

def main():
    """Main function"""
    print(f"Current working directory: {os.getcwd()}")
    
    # Check folder existence
    for folder in folders:
        if not os.path.exists(folder):
            print(f"Warning: {folder} folder not found.")
            return
    
    # 0. Calculate percentage differences (including Milliseconds per token)
    calculate_percentage_differences()
    
    # 0-1. Calculate Generation Time differences
    calculate_generation_time_differences()
    
    # 1. Invocation Latency, TTFT, and Milliseconds per token comparison graphs
    plot_comparison_metrics()
    
    # 2. Cache metrics comparison graphs
    plot_cache_metrics()
    
    # 3. Generation Time and Milliseconds per token comparison graphs
    plot_generation_time_and_tokens_comparison()

if __name__ == "__main__":
    main()