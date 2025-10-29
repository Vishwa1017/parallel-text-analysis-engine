import json
import matplotlib.pyplot as plt
import os
from typing import List
from utils.benchMarkResults import BenchmarkResults

class ResultsVisualizer:
    """Generate graphs and visualizations for benchmark results"""
    
    @staticmethod
    def save_results_to_json(results: List[BenchmarkResults], filename: str = "results/sequential_baseline.json"):
        """Save benchmark results to JSON file"""
        os.makedirs("results", exist_ok=True)
        
        data = {
            "results": [
                {
                    "dataset_size": r.dataset_size,
                    "num_files": r.num_files,
                    "total_tokens": r.total_tokens,
                    "unique_words": r.unique_words,
                    "word_freq_time": r.word_freq_time,
                    "tfidf_time": r.tfidf_time,
                    "total_time": r.total_time,
                    "throughput": r.throughput,
                    "avg_tokens_per_file": r.avg_tokens_per_file
                }
                for r in results
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {filename}")
    
    @staticmethod
    def plot_execution_time(results: List[BenchmarkResults]):
        """Plot execution time comparison across dataset sizes"""
        sizes = [r.dataset_size for r in results]
        word_freq_times = [r.word_freq_time for r in results]
        tfidf_times = [r.tfidf_time for r in results]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = range(len(sizes))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], word_freq_times, width, label='Word Frequency', color='#3498db')
        ax.bar([i + width/2 for i in x], tfidf_times, width, label='TF-IDF Computation', color='#e74c3c')
        
        ax.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Sequential Baseline: Execution Time by Operation', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sizes)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/execution_time.png", dpi=300)
        print("Graph saved: results/execution_time.png")
        plt.close()
    
    @staticmethod
    def plot_throughput(results: List[BenchmarkResults]):
        """Plot throughput (tokens/sec) across dataset sizes"""
        sizes = [r.dataset_size for r in results]
        throughputs = [r.throughput for r in results]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(sizes, throughputs, marker='o', linewidth=2, markersize=10, color='#2ecc71')
        ax.fill_between(range(len(sizes)), throughputs, alpha=0.3, color='#2ecc71')
        
        ax.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Throughput (tokens/sec)', fontsize=12, fontweight='bold')
        ax.set_title('Sequential Baseline: Processing Throughput', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on points
        for i, (size, throughput) in enumerate(zip(sizes, throughputs)):
            ax.text(i, throughput + max(throughputs)*0.02, f'{throughput:.2f}', 
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/throughput.png", dpi=300)
        print("Graph saved: results/throughput.png")
        plt.close()
    
    @staticmethod
    def plot_dataset_metrics(results: List[BenchmarkResults]):
        """Plot dataset metrics (files, tokens, unique words)"""
        sizes = [r.dataset_size for r in results]
        num_files = [r.num_files for r in results]
        total_tokens = [r.total_tokens for r in results]
        unique_words = [r.unique_words for r in results]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Number of files
        axes[0].bar(sizes, num_files, color='#9b59b6')
        axes[0].set_title('Number of Files', fontweight='bold')
        axes[0].set_ylabel('Count', fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(num_files):
            axes[0].text(i, v + max(num_files)*0.02, str(v), ha='center', fontweight='bold')
        
        # Total tokens
        axes[1].bar(sizes, total_tokens, color='#f39c12')
        axes[1].set_title('Total Tokens Processed', fontweight='bold')
        axes[1].set_ylabel('Count', fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(total_tokens):
            axes[1].text(i, v + max(total_tokens)*0.02, f'{v:,}', ha='center', fontweight='bold')
        
        # Unique words
        axes[2].bar(sizes, unique_words, color='#1abc9c')
        axes[2].set_title('Unique Words Found', fontweight='bold')
        axes[2].set_ylabel('Count', fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)
        for i, v in enumerate(unique_words):
            axes[2].text(i, v + max(unique_words)*0.02, str(v), ha='center', fontweight='bold')
        
        fig.suptitle('Sequential Baseline: Dataset Metrics', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/dataset_metrics.png", dpi=300, bbox_inches='tight')
        print("Graph saved: results/dataset_metrics.png")
        plt.close()
    
    @staticmethod
    def plot_total_execution_time(results: List[BenchmarkResults]):
        """Plot total execution time across dataset sizes"""
        sizes = [r.dataset_size for r in results]
        total_times = [r.total_time for r in results]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(sizes, total_times, color=['#3498db', '#e74c3c', '#2ecc71'], edgecolor='black', linewidth=2)
        
        ax.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Execution Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Sequential Baseline: Total Execution Time', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, time in zip(bars, total_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time:.4f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/total_execution_time.png", dpi=300)
        print("Graph saved: results/total_execution_time.png")
        plt.close()
    
    @staticmethod
    def generate_all_visualizations(results: List[BenchmarkResults]):
        """Generate all visualizations"""
        print("\nGenerating visualizations...")
        ResultsVisualizer.save_results_to_json(results)
        ResultsVisualizer.plot_execution_time(results)
        ResultsVisualizer.plot_throughput(results)
        ResultsVisualizer.plot_dataset_metrics(results)
        ResultsVisualizer.plot_total_execution_time(results)
        print("All visualizations completed!\n")