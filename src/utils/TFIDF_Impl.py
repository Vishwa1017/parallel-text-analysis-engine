import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Union

class TFIDFComputer:
    """Sequential TF-IDF computation"""
    
    @staticmethod
    def compute_tfidf(docs: Union[Dict[str, str], List[str]], num_docs: int) -> Dict[str, float]:
        """
        Compute TF-IDF scores for all terms
        Args:
            docs: Dictionary of {filename: content} or List of content strings
            num_docs: Total number of documents
        Returns:
            Dictionary of {term: tfidf_score}
        """
        # Convert list to dict if needed
        if isinstance(docs, list):
            docs = {f"doc_{i}": content for i, content in enumerate(docs)}
        
        # Step 1: Compute Document Frequency (DF)
        df = defaultdict(int)
        for content in docs.values():
            unique_terms = set(content.lower().split())
            for term in unique_terms:
                df[term] += 1
        
        # Step 2: Compute IDF
        idf = {}
        for term, count in df.items():
            idf[term] = math.log(num_docs / (1 + count))
        
        # Step 3: Compute TF and TF-IDF
        tfidf_scores = defaultdict(float)
        for content in docs.values():
            tf = Counter(content.lower().split())
            total_terms = len(content.lower().split())
            
            for term, count in tf.items():
                tf_score = count / total_terms if total_terms > 0 else 0
                tfidf_scores[term] += tf_score * idf.get(term, 0)
        
        # Average TF-IDF across documents
        for term in tfidf_scores:
            tfidf_scores[term] /= len(docs)
        
        return dict(sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True))