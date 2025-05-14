from typing import List, Dict, Any, Tuple
import json
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class HybridSearch:
    """
    Hybrid search that combines keyword-based and vector-based semantic search
    """
    
    def __init__(self, keyword_weight: float = 0.7, semantic_weight: float = 0.3):
        """
        Initialize hybrid search
        
        Args:
            keyword_weight: Weight for keyword search results (0.0 to 1.0)
            semantic_weight: Weight for semantic search results (0.0 to 1.0)
        """
        self.embedding_cache = {}
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        
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
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def keyword_search(self, 
                      query_json: Dict[str, Any], 
                      vision_results: Dict[str, Dict[str, Any]],
                      min_score: float = 0.3) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Perform keyword-based search comparing structured fields
        
        Args:
            query_json: JSON output from QueryAgent
            vision_results: Dictionary mapping image URLs to VisionAgent JSON results
            min_score: Minimum score threshold (0.0 to 1.0)
            
        Returns:
            List of tuples with (image_url, score, match_details)
        """
        # Initialize results
        results = []
        
        # Extract query elements
        query_room_type = query_json.get("room_type", "").lower()
        query_max_capacity = query_json.get("max_capacity", 0)
        query_view_type = query_json.get("view_type", "").lower()
        query_features = [feature.lower() for feature in query_json.get("features", [])]
        query_description = query_json.get("description", "")
        
        # Score each image
        for url, vision_json in vision_results.items():
            score = 0.0
            max_score = 0.0
            match_details = {
                "query_description": query_description
            }
            
            # Room type match
            if query_room_type and query_room_type != "any":
                max_score += 1.0
                vision_room_type = vision_json.get("room_type", "").lower()
                room_match = query_room_type in vision_room_type
                
                if room_match:
                    score += 1.0
                    
                match_details["room_type"] = {
                    "query": query_room_type,
                    "vision": vision_room_type,
                    "match": room_match
                }
            
            # Max capacity match
            if query_max_capacity > 0:
                max_score += 1.0
                vision_max_capacity = vision_json.get("max_capacity", 0)
                capacity_match = vision_max_capacity >= query_max_capacity
                
                if capacity_match:
                    score += 1.0
                    
                match_details["max_capacity"] = {
                    "query": query_max_capacity,
                    "vision": vision_max_capacity,
                    "match": capacity_match
                }
            
            # View type match
            if query_view_type and query_view_type not in ["any", "standard"]:
                max_score += 1.0
                vision_view_type = vision_json.get("view_type", "").lower()
                view_match = query_view_type in vision_view_type
                
                if view_match:
                    score += 1.0
                    
                match_details["view_type"] = {
                    "query": query_view_type,
                    "vision": vision_view_type,
                    "match": view_match
                }
            
            # Features match
            if query_features:
                max_score += len(query_features)
                vision_features = [feature.lower() for feature in vision_json.get("features", [])]
                
                matched_features = []
                for feature in query_features:
                    for vision_feature in vision_features:
                        if feature in vision_feature:
                            matched_features.append(feature)
                            score += 1.0
                            break
                            
                match_details["features"] = {
                    "query": query_features,
                    "vision": vision_features,
                    "matches": matched_features
                }
            
            # Calculate normalized score
            normalized_score = score / max_score if max_score > 0 else 0.0
            
            # Add to results if score meets minimum threshold
            if normalized_score >= min_score:
                results.append((url, normalized_score, match_details))
        
        # Sort by score (descending)
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def semantic_search(self, query_json: Dict[str, Any], vision_results: Dict[str, Dict[str, Any]]) -> List[Tuple[str, float]]:
        """
        Perform semantic search using description fields
        
        Args:
            query_json: JSON output from QueryAgent
            vision_results: Dictionary mapping image URLs to VisionAgent JSON results
            
        Returns:
            List of tuples with (image_url, similarity_score)
        """
        query_description = query_json.get("description", "")
        
        if not query_description:
            return []
        
        # Get query embedding
        query_embedding = self.get_embedding(query_description)
        
        # Get embeddings for vision descriptions
        results = []
        for url, vision_json in vision_results.items():
            vision_description = vision_json.get("description", "")
            
            if not vision_description:
                continue
                
            vision_embedding = self.get_embedding(vision_description)
            similarity = self.cosine_similarity(query_embedding, vision_embedding)
            
            results.append((url, similarity))
        
        # Sort by similarity (descending)
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def hybrid_search(self, 
                     query_json: Dict[str, Any], 
                     vision_results: Dict[str, Dict[str, Any]],
                     keyword_min_score: float = 0.3,
                     semantic_min_score: float = 0.5,
                     max_results: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Perform hybrid search (keyword + semantic)
        
        Args:
            query_json: JSON output from QueryAgent
            vision_results: Dictionary mapping image URLs to VisionAgent JSON results
            keyword_min_score: Minimum score for keyword search
            semantic_min_score: Minimum score for semantic search
            max_results: Maximum number of results to return
            
        Returns:
            List of tuples with (image_url, combined_score, match_details)
        """
        # Perform keyword search
        keyword_results = self.keyword_search(query_json, vision_results, min_score=keyword_min_score)
        
        # Convert keyword results to dictionary for easy lookup
        keyword_scores = {url: score for url, score, _ in keyword_results}
        keyword_details = {url: details for url, _, details in keyword_results}
        
        # Perform semantic search
        semantic_results = self.semantic_search(query_json, vision_results)
        
        # Convert semantic results to dictionary for easy lookup
        semantic_scores = {url: score for url, score in semantic_results if score >= semantic_min_score}
        
        # Combine results
        combined_results = {}
        
        # Process all URLs that appear in either keyword or semantic results
        all_urls = set(keyword_scores.keys()) | set(semantic_scores.keys())
        
        for url in all_urls:
            keyword_score = keyword_scores.get(url, 0.0)
            semantic_score = semantic_scores.get(url, 0.0)
            
            # Calculate combined score
            combined_score = (self.keyword_weight * keyword_score) + (self.semantic_weight * semantic_score)
            
            # Get details
            details = keyword_details.get(url, {})
            
            # Add semantic score to details
            details["semantic_score"] = semantic_score
            details["keyword_score"] = keyword_score
            details["combined_score"] = combined_score
            
            combined_results[url] = (url, combined_score, details)
        
        # Sort by combined score (descending) and limit results
        sorted_results = sorted(combined_results.values(), key=lambda x: x[1], reverse=True)
        return sorted_results[:max_results]
    
    def format_search_results(self, search_results: List[Tuple[str, float, Dict[str, Any]]], vision_results: Dict[str, Dict[str, Any]]) -> str:
        """
        Format search results as a readable string
        
        Args:
            search_results: Results from the hybrid_search method
            vision_results: Dictionary mapping image URLs to VisionAgent JSON results
            
        Returns:
            Formatted string with search results
        """
        if not search_results:
            return "No matching rooms found."
        
        output = f"Found {len(search_results)} matching rooms:\n\n"
        
        for i, (url, combined_score, details) in enumerate(search_results, 1):
            output += f"Match #{i}: {url}\n"
            output += f"Combined Score: {combined_score:.2f}\n"
            output += f"Keyword Score: {details.get('keyword_score', 0.0):.2f}\n"
            output += f"Semantic Score: {details.get('semantic_score', 0.0):.2f}\n"
            
            # Room type
            if "room_type" in details:
                room_info = details["room_type"]
                match_text = "✓" if room_info["match"] else "✗"
                output += f"Room Type: {match_text} Query: '{room_info['query']}', Found: '{room_info['vision']}'\n"
            
            # Max capacity
            if "max_capacity" in details:
                capacity_info = details["max_capacity"]
                match_text = "✓" if capacity_info["match"] else "✗"
                output += f"Capacity: {match_text} Query: {capacity_info['query']}, Found: {capacity_info['vision']}\n"
            
            # View type
            if "view_type" in details:
                view_info = details["view_type"]
                match_text = "✓" if view_info["match"] else "✗"
                output += f"View: {match_text} Query: '{view_info['query']}', Found: '{view_info['vision']}'\n"
            
            # Features
            if "features" in details:
                feature_info = details["features"]
                match_text = f"({len(feature_info['matches'])}/{len(feature_info['query'])})"
                output += f"Features {match_text}:\n"
                
                for query_feature in feature_info["query"]:
                    match_text = "✓" if query_feature in feature_info["matches"] else "✗"
                    output += f"  {match_text} {query_feature}\n"
                
                output += f"Found in room: {', '.join(feature_info['vision'])}\n"
            
            # Descriptions (for semantic matching)
            vision_json = vision_results.get(url, {})
            query_description = details.get("query_description", "N/A")
            vision_description = vision_json.get("description", "N/A")
            
            output += f"Query Description: \"{query_description}\"\n"
            output += f"Room Description: \"{vision_description}\"\n"
            
            output += "\n"
        
        return output

# Example usage
if __name__ == "__main__":
    # Example query JSON (from QueryAgent)
    query_json = {
        "room_type": "double",
        "max_capacity": 2,
        "view_type": "sea view",
        "features": ["balcony", "air conditioning"],
        "description": "Double room with sea view, balcony, and air conditioning."
    }
    
    # Example vision JSONs (from VisionAgent)
    vision_results = {
        "image1.jpg": {
            "room_type": "double",
            "max_capacity": 2,
            "view_type": "sea view",
            "features": ["private balcony", "air conditioning", "flat-screen TV"],
            "description": "Spacious double room with a beautiful sea view and a private balcony. The room has air conditioning, a flat-screen TV, and a comfortable queen-sized bed."
        },
        "image2.jpg": {
            "room_type": "double",
            "max_capacity": 2,
            "view_type": "garden view",
            "features": ["balcony", "air conditioning", "desk"],
            "description": "Double room overlooking the garden with a balcony and desk. Well-equipped with air conditioning and a working area."
        },
        "image3.jpg": {
            "room_type": "single",
            "max_capacity": 1,
            "view_type": "sea view",
            "features": ["balcony", "air conditioning"],
            "description": "Comfortable single room with sea view and essential amenities including a balcony and air conditioning."
        },
        "image4.jpg": {
            "room_type": "double",
            "max_capacity": 3,
            "view_type": "mountain view",
            "features": ["balcony", "air conditioning", "sofa bed"],
            "description": "Spacious room with a sofa bed and mountain views. Features a balcony and air conditioning."
        },
        "image5.jpg": {
            "room_type": "suite",
            "max_capacity": 4,
            "view_type": "sea view",
            "features": ["balcony", "air conditioning", "kitchenette", "living area"],
            "description": "Luxury suite with a separate living area and kitchenette. Offers sea views from a large balcony and air conditioning throughout."
        }
    }
    
    # Create search instance
    search = HybridSearch(keyword_weight=0.6, semantic_weight=0.4)
    
    # Perform hybrid search
    results = search.hybrid_search(query_json, vision_results)
    
    # Format and display results
    formatted_results = search.format_search_results(results, vision_results)
    print(formatted_results) 