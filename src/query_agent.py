import os
import json
from openai import OpenAI
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class QueryAgent:
    """
    This agent analyzes user text queries and converts them 
    into a JSON format expected by the vision_agent.
    """
    
    def __init__(self):
        self.model = "gpt-4-turbo"  # or another model can be used
        
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Processes the user query and converts it to a structured JSON format
        
        Args:
            user_query: The textual query from the user
            
        Returns:
            Query structured in JSON format
        """
        try:
            # Call LLM with system instructions and user input
            response = client.chat.completions.create(
    model=self.model,
    messages=[
        {
            "role": "system",
            "content": """
            You are a hotel‐room search assistant.  
            Return exactly one JSON object with these fields and allowed values:

            {
            "room_type":    "single | double | twin | suite | family room | studio | luxury suite | \"\"",
            "max_capacity": integer or null,
            "view_type":    "sea | city | garden | mountain | pool | street | none | \"\"",
            "features":     ["any feature explicitly mentioned by the user", …],
            "description":  "concise summary (<70 words) using only requested fields"
            }

            **Rules:**
            - **ONLY** include fields the user **explicitly** mentioned.  
            - If the user omits a field, set defaults:  
            • room_type="standard"  
            • max_capacity=2  
            • view_type="standard"  
            - For every field the user did not mention at all, you may also use `""` (strings), `null` (integers), or `[]` (lists) instead of defaults.  
            - The **features** array must list **only** the exact amenities the user named; do not invent extras.  
            - The **description** should be under 50 words, mention exactly the desired features and avoid any embellishments, but should also accommodate the user's potential wishes.  
            - Return **only** the JSON object—no extra text.
            """
        },
        {
            "role": "user",
            "content": user_query
        }
    ],
    response_format={"type": "json_object"}
            )
            
            # Parse the response from the LLM
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"Error processing query: {e}")
            # Return a default structure in case of error
            return {
                "room_type": "any",
                "max_capacity": 2,
                "view_type": "any",
                "features": [],
                "description": f"Error processing query: {str(e)}"
            }

    def explain_query(self, query_json: Dict[str, Any]) -> str:
        """
        Converts the JSON format query into a human-readable text
        
        Args:
            query_json: Structured query
            
        Returns:
            Human-readable explanation of the query
        """
        room_type = query_json.get("room_type", "not specified")
        max_capacity = query_json.get("max_capacity", 0)
        view_type = query_json.get("view_type", "not specified")
        features = query_json.get("features", [])
        features_text = ", ".join(features) if features else "not specified"
        
        explanation = f"Search criteria:\n"
        explanation += f"- Room type: {room_type}\n"
        explanation += f"- Maximum capacity: {max_capacity} people\n"
        explanation += f"- View: {view_type}\n"
        explanation += f"- Features: {features_text}\n"
        explanation += f"\nDescription: {query_json.get('description', '')}"
        
        return explanation

# Test usage
if __name__ == "__main__":
    test_queries = [
        "I'm looking for a double room with a sea view.",
        "I want a room with a balcony and air conditioning, with a city view.",
        "Do you have a triple room with a desk?",
        "I'm looking for a room that can accommodate up to 4 people."
    ]
    
    agent = QueryAgent()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*50}")
        print(f"Test #{i}: Original Query: \"{query}\"")
        
        # Process the user query
        result = agent.process_query(query)
        
        # Show results
        print("\nJSON output:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Human-readable explanation
        print("\nHuman-readable explanation:")
        print(agent.explain_query(result)) 