# OBilet Hotel Room Image Filtering System

A sophisticated AI-powered system for filtering and searching hotel room images based on user preferences. This system uses visual AI and natural language processing to analyze hotel room images and match them with user queries.

## System Overview

The OBilet Hotel Room Image Filtering System consists of three main components:

1. **VisionAgent**: Analyzes hotel room images using OpenAI's GPT-4o to extract detailed information including:
   - Room type (single, double, suite, etc.)
   - Maximum capacity
   - View type (sea, city, garden, etc.)
   - Room features (balcony, desk, TV, etc.)
   - Descriptive summary

2. **QueryAgent**: Processes natural language queries from users and converts them into a structured JSON format that can be used for search.

3. **HybridSearch**: Combines two search approaches for optimal results:
   - Keyword-based matching for specific attributes
   - Semantic search using embedding vectors for context-aware matching

## Features

- **Visual Analysis**: AI-based detection of room attributes directly from images
- **Natural Language Queries**: Search with everyday language (e.g., "Double room with sea view")
- **Hybrid Search**: Combined keyword and semantic search for better results
- **Analysis Caching**: Saves analyzed results to prevent redundant API calls
- **Rate Limit Handling**: Built-in delays between API calls to prevent rate limiting
- **Web Interface**: User-friendly interface for searching and viewing results
- **Visual Result Indicators**: Color-coded indicators showing which search criteria match

## Technical Architecture

```
OBilet2/
├── data/                   # Stores analysis results
├── src/
│   ├── __init__.py
│   ├── main.py             # Main system integration
│   ├── vision_agent.py     # Image analysis with GPT-4o
│   ├── query_agent.py      # Natural language query processing
│   ├── hybrid_search.py    # Combined search algorithm
│   ├── web_app.py          # Flask web application
│   └── templates/
│       └── index.html      # Web interface template
└── requirements.txt        # Python dependencies
```

## Requirements

- Python 3.8+
- OpenAI API key
- Flask for web interface
- Dependencies listed in requirements.txt

## Installation

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Command Line Interface

Run the main application with various options:

```bash
# Analyze all images (with 4-second delay between images to avoid rate limits)
PYTHONPATH=. python src/main.py --analyze --wait 4.0

# Re-analyze all images (force refresh)
PYTHONPATH=. python src/main.py --analyze --force --wait 4.0

# Search with a specific query
PYTHONPATH=. python src/main.py --query "Double room with sea view"

# Interactive search mode
PYTHONPATH=. python src/main.py
```

### Web Interface

Run the Flask web application:

```bash
PYTHONPATH=. python src/web_app.py
```

Then open a browser and navigate to:
- http://127.0.0.1:8080 (local access)
- http://[your-ip]:8080 (network access)

The web interface allows you to:
1. Enter natural language search queries
2. View search results with detailed matching information
3. Analyze or re-analyze all hotel images

## How It Works

1. **Image Analysis Process**:
   - The system downloads hotel room images from URLs
   - Images are encoded and sent to OpenAI's GPT-4o model
   - The model analyzes visual content and returns structured data
   - Results are cached to avoid redundant analysis

2. **Query Processing**:
   - User queries in natural language are processed by GPT-4-turbo
   - The model extracts key search criteria (room type, capacity, view, features)
   - A structured JSON representation is created

3. **Search Algorithm**:
   - Keyword matching compares structured fields (60% of the score by default)
   - Semantic matching uses embedding vectors to compare descriptions (40% of the score by default)
   - Results are ranked by combined score and filtered by minimum thresholds

4. **Result Presentation**:
   - Results display room details with scores
   - Color indicators show which criteria matched (green) or didn't match (red)
   - Combined score represents the overall match quality

## Example Queries

- "Double rooms with a sea view"
- "Rooms with a balcony and air conditioning, with a city view"
- "Triple rooms with a desk"
- "Rooms that can accommodate 4 people"
- "Luxury rooms with a bathtub and view"

## API Endpoints

The web application provides two main API endpoints:

- **POST /search**: Search for hotel rooms with a given query
  - Param: `query` - Natural language search query
  - Returns: Matching room results with scores and details

- **POST /analyze**: Analyze all hotel room images
  - Param: `force` - Whether to force re-analysis of all images
  - Returns: Status and count of analyzed images

## Technical Notes

- The system uses GPT-4o for image analysis and GPT-4-turbo for query processing
- Rate limiting is handled by introducing configurable wait times between API calls
- The hybrid search algorithm can be tuned by adjusting weights and thresholds
- Analysis results are stored in JSON format for persistence between sessions

## Future Improvements

Potential enhancements for the system:

1. Support for additional room attributes (bathroom type, amenities, accessibility)
2. Enhanced UI with filters, sorting options, and pagination
3. Performance optimizations for larger datasets
4. Integration with booking systems
5. Multi-language support for queries and results
6. Advanced filtering options in the web interface 