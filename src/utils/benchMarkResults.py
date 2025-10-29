from dataclasses import dataclass
from typing import List

@dataclass
class BenchmarkResults:
    """Stores comprehensive benchmark metrics"""
    dataset_size: str
    num_files: int
    total_tokens: int
    unique_words: int
    execution_time: float
    throughput: float  # tokens per second
    avg_tokens_per_file: float
    word_freq_time: float
    tfidf_time: float
    total_time: float
    
    def __str__(self):
        return f"""
╔════════════════════════════════════════════════════╗
║         SEQUENTIAL BASELINE RESULTS - {self.dataset_size.upper()}         ║
╠════════════════════════════════════════════════════╣
║ Dataset Size: {self.dataset_size.ljust(40)} ║
║ Number of Files: {str(self.num_files).ljust(36)} ║
║ Total Tokens Processed: {str(self.total_tokens).ljust(30)} ║
║ Unique Words Found: {str(self.unique_words).ljust(34)} ║
║ Avg Tokens Per File: {str(self.avg_tokens_per_file).ljust(33)} ║
╠════════════════════════════════════════════════════╣
║ TIMING METRICS:                                    ║
║ Word Frequency Time: {f'{self.word_freq_time:.4f}s'.ljust(27)} ║
║ TF-IDF Computation Time: {f'{self.tfidf_time:.4f}s'.ljust(21)} ║
║ Total Execution Time: {f'{self.total_time:.4f}s'.ljust(25)} ║
║ Throughput: {f'{self.throughput:.2f} tokens/sec'.ljust(36)} ║
╚════════════════════════════════════════════════════╝
        """

class Benchmarker:
    """Helper class to measure and record benchmark metrics"""
    
    @staticmethod
    def measure(
        dataset_size: str,
        num_files: int,
        total_tokens: int,
        unique_words: int,
        word_freq_time: float,
        tfidf_time: float
    ) -> BenchmarkResults:
        """
        Create benchmark results with all metrics
        """
        total_time = word_freq_time + tfidf_time
        throughput = total_tokens / total_time if total_time > 0 else 0
        avg_tokens_per_file = total_tokens / num_files if num_files > 0 else 0
        
        return BenchmarkResults(
            dataset_size=dataset_size,
            num_files=num_files,
            total_tokens=total_tokens,
            unique_words=unique_words,
            execution_time=total_time,
            throughput=throughput,
            avg_tokens_per_file=avg_tokens_per_file,
            word_freq_time=word_freq_time,
            tfidf_time=tfidf_time,
            total_time=total_time
        )