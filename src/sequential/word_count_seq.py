from collections import Counter
from utils.text_preprocess import clean_and_tokenize
from utils.logger import log_header, log_info, log_success

def compute_word_frequencies(documents: list[str]) -> Counter:
    """
    Compute total word frequencies across all documents sequentially.
    Returns a Counter object mapping word → frequency.
    """
    log_header("Step 3 — Sequential Word Frequency Analysis")

    total_counter = Counter()
    for i, doc in enumerate(documents, start=1):
        tokens = clean_and_tokenize(doc)
        #log_info(f"Doc {i}: {len(tokens)} tokens processed")
        total_counter.update(tokens)

    log_success(f"Total unique words found: {len(total_counter)}")
    return total_counter
