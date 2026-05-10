import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class HistoricalMemoryLayer:
    """
    Historical Memory Layer using Retrieval-Augmented Generation (RAG) concepts.
    Stores successfully resolved past tickets.
    When a new ambiguous ticket arrives, it retrieves the K nearest historical tickets.
    This can be used to dynamically boost confidence or suggest resolutions.
    """
    def __init__(self, data_path: str = None):
        if data_path is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base, 'data', 'processed', 'train.csv')
            
        self.data_path = data_path
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.memory_df = None
        self.is_ready = False
        
        self._load_memory()

    def _load_memory(self):
        try:
            if not os.path.exists(self.data_path):
                logger.warning(f"[HistoricalMemory] Data file not found at {self.data_path}")
                return

            self.memory_df = pd.read_csv(self.data_path)
            
            # Ensure required columns exist
            if 'text' not in self.memory_df.columns or 'category' not in self.memory_df.columns:
                logger.warning("[HistoricalMemory] Required columns ('text', 'category') missing.")
                return

            # Fit TF-IDF and Nearest Neighbors
            logger.info(f"[HistoricalMemory] Indexing {len(self.memory_df)} historical tickets...")
            X = self.vectorizer.fit_transform(self.memory_df['text'].fillna(''))
            self.nn_model.fit(X)
            
            self.is_ready = True
            logger.info("[HistoricalMemory] Indexing complete.")
            
        except Exception as e:
            logger.error(f"[HistoricalMemory] Failed to load memory: {e}")

    def retrieve_similar(self, query_text: str, k: int = 3) -> List[Dict]:
        """
        Retrieve top K similar historical tickets.
        """
        if not self.is_ready:
            return []

        # Vectorize query
        X_query = self.vectorizer.transform([query_text])
        
        # Search
        distances, indices = self.nn_model.kneighbors(X_query, n_neighbors=k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # Cosine distance to similarity score
            similarity = 1.0 - dist
            row = self.memory_df.iloc[idx]
            
            results.append({
                'text': row['text'],
                'category': row['category'],
                'similarity': round(similarity, 4)
            })
            
        return results

    def compute_historical_boost(self, query_text: str, candidate_category: str, k: int = 5) -> float:
        """
        Calculate a confidence boost if the most similar past tickets 
        were resolved in the same candidate category.
        """
        if not self.is_ready:
            return 0.0
            
        similar_tickets = self.retrieve_similar(query_text, k=k)
        if not similar_tickets:
            return 0.0
            
        # Count how many of the top-k match the candidate category, weighted by similarity
        boost = 0.0
        total_weight = 0.0
        
        for t in similar_tickets:
            weight = t['similarity']
            total_weight += weight
            if t['category'] == candidate_category:
                boost += weight
                
        if total_weight == 0:
            return 0.0
            
        match_ratio = boost / total_weight
        
        # Max boost is 0.15 (15%)
        return round(match_ratio * 0.15, 4)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    memory = HistoricalMemoryLayer()
    
    test_queries = [
        "My invoice from last month is incorrect, please fix the billing.",
        "The API keeps returning 500 errors since last Tuesday's update.",
        "How do I add another user to our account?"
    ]
    
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        results = memory.retrieve_similar(q, k=2)
        for r in results:
            print(f" -> [{r['category']}] (sim: {r['similarity']:.2f}) {r['text']}")
            
        boost = memory.compute_historical_boost(q, "billing")
        print(f"Historical boost for 'billing': +{boost}")
