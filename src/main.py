import os
import time
import shutil
from utils.dataSetGenrator import DatasetGenerator
from utils.file_loader import load_text_files
from sequential.word_count_seq import compute_word_frequencies
from utils.TFIDF_Impl import TFIDFComputer
from utils.benchMarkResults import Benchmarker
from utils.resultVisualiser import ResultsVisualizer
from utils.logger import log_header, log_info, log_success

if __name__ == "__main__":
    log_header("Sequential Baseline — Multiple Dataset Sizes")
    
    # Generate datasets
    log_info("Generating datasets...")
    DatasetGenerator.create_datasets()
    
    dataset_sizes = ['small', 'medium', 'large']
    results = []
    
    for size in dataset_sizes:
        folder = f"datasets/test/{size}"
        log_header(f"Processing {size.upper()} dataset")
        
        # Load documents
        docs = load_text_files(folder)
        num_files = len(docs)
        
        if not docs:
            log_info(f"No documents found in {folder}")
            continue
        
        # Count total tokens
        if isinstance(docs, list):
            total_tokens = sum(len(content.split()) for content in docs)
        else:
            total_tokens = sum(len(content.split()) for content in docs.values())
        
        # Benchmark word frequency
        start_time = time.time()
        word_freqs = compute_word_frequencies(docs)
        word_freq_time = time.time() - start_time
        
        # Get unique words count
        unique_words = len(word_freqs)
        
        # Benchmark TF-IDF
        start_time = time.time()
        tfidf_scores = TFIDFComputer.compute_tfidf(docs, num_files)
        tfidf_time = time.time() - start_time
        
        # Record results with correct parameters
        benchmark = Benchmarker.measure(
            dataset_size=size,
            num_files=num_files,
            total_tokens=total_tokens,
            unique_words=unique_words,
            word_freq_time=word_freq_time,
            tfidf_time=tfidf_time
        )
        results.append(benchmark)
        
        log_success(f"Completed {size} dataset analysis")
        print(benchmark)
        
        log_info("Top 10 frequent words:")
        for word, count in word_freqs.most_common(10):
            print(f"  {word}: {count}")
        
        log_info("Top 10 TF-IDF terms:")
        tfidf_list = list(tfidf_scores.items())[:10]
        for term, score in tfidf_list:
            print(f"  {term}: {score:.6f}")
    
    # Summary
    log_header("Sequential Baseline Summary")
    for result in results:
        print(result)
    
    # Generate visualizations
    ResultsVisualizer.generate_all_visualizations(results)
    
    # Delete datasets folder after completion
    log_info("Cleaning up datasets...")
    time.sleep(1)
    
    if os.path.exists("datasets"):
        try:
            shutil.rmtree("datasets")
            log_success("Datasets folder deleted successfully")
        except OSError as e:
            log_info(f"Warning: Could not delete datasets folder: {e}")