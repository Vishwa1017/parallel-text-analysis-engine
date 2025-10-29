import os
from pathlib import Path

class DatasetGenerator:
    """Generate synthetic datasets of varying sizes for testing"""
    
    @staticmethod
    def create_datasets():
        """Create small, medium, and large datasets"""
        datasets = {
            'small': 10,      # 10 files
            'medium': 100,    # 100 files
            'large': 1000     # 1000 files
        }
        
        for size, num_files in datasets.items():
            folder = f"datasets/test/{size}"
            os.makedirs(folder, exist_ok=True)
            
            for i in range(num_files):
                filepath = f"{folder}/file_{i}.txt"
                if not os.path.exists(filepath):
                    with open(filepath, 'w') as f:
                        # Generate sample text
                        f.write(DatasetGenerator._generate_sample_text(i))
    
    @staticmethod
    def _generate_sample_text(file_id):
        """Generate sample text content"""
        words = ["python", "data", "analysis", "text", "processing", "parallel", 
                 "distributed", "algorithm", "performance", "optimization"] * 100
        return " ".join(words[file_id:file_id+500])