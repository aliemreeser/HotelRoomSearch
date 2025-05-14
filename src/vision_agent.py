import os
import requests
import base64
from openai import OpenAI
from typing import List, Dict, Any
import json
from dotenv import load_dotenv
import urllib3
import pathlib

# SSL uyarılarını devre dışı bırak
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class VisionAgent:
    def __init__(self):
        self.image_descriptions = {}
        self.embedding_cache = {}
        
    def download_image(self, image_url: str) -> bytes:
        """Download image from URL and return as bytes"""
        response = requests.get(image_url, verify=False)
        if response.status_code != 200:
            raise Exception(f"Failed to download image: {response.status_code}")
        return response.content
    
    def read_local_image(self, image_path: str) -> bytes:
        """Read image from local file and return as bytes"""
        with open(image_path, "rb") as f:
            return f.read()
    
    def encode_image_to_base64(self, image_content: bytes) -> str:
        """Encode image bytes to base64 string"""
        return base64.b64encode(image_content).decode('utf-8')
    
    def analyze_image(self, image_path_or_url: str, is_local: bool = False) -> Dict[str, Any]:
        """
        Analyze image using OpenAI's GPT-4o-mini vision model and return detailed description
        
        Args:
            image_path_or_url: Path to local file or URL to image
            is_local: If True, image_path_or_url is a local file path
        """
        try:
            # Get image content
            if is_local:
                image_content = self.read_local_image(image_path_or_url)
            else:
                image_content = self.download_image(image_path_or_url)
            
            # Encode image
            base64_image = self.encode_image_to_base64(image_content)
            
            # Call OpenAI API with GPT-4o-mini
            response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": """
            You are an AI specialized in analyzing hotel room images.
            Return exactly one JSON object with these fields and possible values:

            {
            "room_type":    "single | double | twin | suite | family room | studio | luxury suite | \"\"",
            "max_capacity": integer or null,
            "view_type":    "sea | city | garden | mountain | pool | none | \"\"",
            "features":     ["any visible feature as a string", …],
            "description":  "A brief paragraph describing the room and visible features."
            }

            **Rules:**
            - **List every feature** you can **visually confirm** in the image; do **not** restrict to a predefined list.
            - Fill only fields you can actually see.
            - If you cannot confirm a field, use `""` for strings, `[]` for lists, and `null` for integers.
            - The **description** must mention room type, capacity, view (if any), and summarize the visible features in one or two sentences.
            - **Do not** guess or invent anything not visible.
            - **Return only** the JSON object, no extra text.
            """
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this hotel room image and provide a detailed description."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    response_format={"type": "json_object"}
            )
            
            # Parse the response
            description = json.loads(response.choices[0].message.content)
            
            # Cache the result
            self.image_descriptions[image_path_or_url] = description
            
            return description
            
        except Exception as e:
            print(f"Error analyzing image {image_path_or_url}: {e}")
            return {
                "room_type": "unknown",
                "max_capacity": 0,
                "view_type": "unknown",
                "features": [],
                "description": f"Failed to analyze image: {str(e)}"
            }
    
    def analyze_images(self, image_urls: List[str]) -> Dict[str, Dict[str, Any]]:
        """Analyze multiple images and return their descriptions"""
        results = {}
        for url in image_urls:
            results[url] = self.analyze_image(url)
        return results

# Example usage
if __name__ == "__main__":
    # Using the correct URL
    test_image_url = "https://static.obilet.com.s3.eu-central-1.amazonaws.com/CaseStudy/HotelImages/1.jpg"
    
    agent = VisionAgent()
    
    # URL test
    try:
        print(f"\n{'='*50}")
        print(f"Analyzing test image: {test_image_url}")
        
        # Download and save the image (for display purposes)
        try:
            image_content = agent.download_image(test_image_url)
            temp_image_path = "temp_image.jpg"
            with open(temp_image_path, "wb") as f:
                f.write(image_content)
            print(f"Image successfully downloaded and saved: {temp_image_path}")
            print("You can open temp_image.jpg to view the image.")
        except Exception as e:
            print(f"Error downloading image: {e}")
            exit(1)  # End the program if image cannot be downloaded
        
        # Analyze the image
        result = agent.analyze_image(test_image_url)
        
        print("\nAnalysis result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        print("\nKey features:")
        print(f"- Room type: {result.get('room_type', 'Unknown')}")
        print(f"- Maximum capacity: {result.get('max_capacity', 'Unknown')}")
        print(f"- View: {result.get('view_type', 'Unknown')}")
        print(f"- Features: {', '.join(result.get('features', ['Unknown']))}")
        
    except Exception as e:
        print(f"Error during test: {e}") 