import os
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import json

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class SemanticSearch:
    def __init__(self):
        self.embedding_cache = {}
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using OpenAI API"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        
        embedding = np.array(response.data[0].embedding)
        self.embedding_cache[text] = embedding
        return embedding
    
    def calculate_similarities(self, query_embedding: np.ndarray, text_embeddings: List[np.ndarray]) -> np.ndarray:
        """Calculate cosine similarities between query and text embeddings"""
        query_embedding = query_embedding.reshape(1, -1)
        text_embeddings_matrix = np.vstack(text_embeddings)
        
        return cosine_similarity(query_embedding, text_embeddings_matrix)[0]
    
    def create_image_embeddings(self, image_descriptions: Dict[str, Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Create embeddings for image descriptions"""
        embeddings = {}
        
        for url, info in image_descriptions.items():
            # Create a comprehensive description text for embedding
            description_text = f"Room type: {info.get('room_type', '')}, "
            description_text += f"View: {info.get('view_type', '')}, "
            description_text += f"Features: {', '.join(info.get('features', []))}, "
            description_text += f"Max capacity: {info.get('max_capacity', 0)}, "
            description_text += f"Description: {info.get('description', '')}"
            
            # Get embedding for this text
            embedding = self.get_embedding(description_text)
            embeddings[url] = embedding
            
        return embeddings
    
    def semantic_search(self, query: str, image_descriptions: Dict[str, Dict[str, Any]], 
                        threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        Perform semantic search on image descriptions based on query
        Returns a list of (image_url, similarity_score) tuples above threshold
        """
        # Create query embedding
        query_embedding = self.get_embedding(query)
        
        # Create embeddings for all image descriptions
        image_embeddings = self.create_image_embeddings(image_descriptions)
        
        # Calculate similarities
        similarities = self.calculate_similarities(
            query_embedding, 
            list(image_embeddings.values())
        )
        
        # Create (url, score) pairs and filter by threshold
        url_score_pairs = list(zip(image_embeddings.keys(), similarities))
        filtered_results = [(url, score) for url, score in url_score_pairs if score >= threshold]
        
        # Sort by similarity score (descending)
        return sorted(filtered_results, key=lambda x: x[1], reverse=True)

# Example usage
if __name__ == "__main__":
    # Example image descriptions
    image_descriptions = {
        "image1.jpg": {
            "room_type": "double",
            "max_capacity": 2,
            "view_type": "sea view",
            "features": ["balcony", "air conditioning", "TV"],
            "description": "Spacious double room with a beautiful sea view and a private balcony."
        },
        "image2.jpg": {
            "room_type": "triple",
            "max_capacity": 3,
            "view_type": "city view",
            "features": ["desk", "minibar", "air conditioning"],
            "description": "Comfortable triple room with a desk and city views."
        }
    }
    
    search = SemanticSearch()
    results = search.semantic_search("double room with sea view", image_descriptions)
    
    print("Search results:")
    for url, score in results:
        print(f"{url}: {score:.4f}") 