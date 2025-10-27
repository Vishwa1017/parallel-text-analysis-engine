import os
from utils.file_loader import load_text_files
from sequential.word_count_seq import compute_word_frequencies
from utils.logger import log_header, log_info, log_success

if __name__ == "__main__":
    log_header("Sequential Pipeline — Word Frequency Stage")

    folder = "datasets/test"
    #Create dir if not exists
    if not os.path.exists(folder):
        os.makedirs(folder)
        log_info(f"Created directory: {folder}")
        
    docs = load_text_files(folder)

    if not docs:
        log_info("No documents found. Add .txt files and re-run.")
    else:
        freqs = compute_word_frequencies(docs)
        log_success("Word frequency analysis completed!")

        log_info("Top 10 most frequent words:")
        for word, count in freqs.most_common(10):
            print(f"{word}: {count}")
