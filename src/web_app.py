import os
import json
from flask import Flask, render_template, request, jsonify
from src.main import OBiletHotelSearch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)

# Initialize search system
search_system = OBiletHotelSearch()

# Data directory for storing analysis results
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Load analyzed images at startup
search_system.load_analyzed_images()

# Generate image URLs for OBilet (1.jpg to 25.jpg)
base_url = "https://static.obilet.com.s3.eu-central-1.amazonaws.com/CaseStudy/HotelImages/"
hotel_images = [f"{base_url}{i}.jpg" for i in range(1, 26)]

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze hotel room images"""
    force_reanalyze = request.form.get('force', 'false') == 'true'
    
    # Start image analysis
    results = search_system.analyze_images(hotel_images, force_reanalyze=force_reanalyze)
    
    return jsonify({
        'status': 'success',
        'message': f'Analysis completed: {len(results)} images.',
        'count': len(results)
    })

@app.route('/search', methods=['POST'])
def search():
    """Search for hotel rooms based on user query"""
    user_query = request.form.get('query', '')
    
    if not user_query:
        return jsonify({
            'status': 'error',
            'message': 'Query cannot be empty.'
        })
    
    # Process query and get results
    query_json = search_system.query_agent.process_query(user_query)
    search_results = search_system.hybrid_search.hybrid_search(
        query_json, 
        search_system.analyzed_images,
        keyword_min_score=0.3,
        semantic_min_score=0.5,
        max_results=5
    )
    
    # Format results for web display
    formatted_results = []
    for i, (url, score, details) in enumerate(search_results, 1):
        vision_json = search_system.analyzed_images.get(url, {})
        
        result = {
            'rank': i,
            'url': url,
            'image_url': url,
            'combined_score': round(score * 100),
            'keyword_score': round(details.get('keyword_score', 0.0) * 100),
            'semantic_score': round(details.get('semantic_score', 0.0) * 100),
            'room_type': vision_json.get('room_type', 'Unknown'),
            'max_capacity': vision_json.get('max_capacity', 0),
            'view_type': vision_json.get('view_type', 'Unknown'),
            'features': vision_json.get('features', []),
            'description': vision_json.get('description', ''),
            'query_description': details.get('query_description', ''),
            'matches': {}
        }
        
        # Add match details
        if 'room_type' in details:
            result['matches']['room_type'] = details['room_type']['match']
            
        if 'max_capacity' in details:
            result['matches']['max_capacity'] = details['max_capacity']['match']
            
        if 'view_type' in details:
            result['matches']['view_type'] = details['view_type']['match']
            
        if 'features' in details:
            result['matches']['features'] = {
                'query': details['features']['query'],
                'matches': details['features']['matches']
            }
        
        formatted_results.append(result)
    
    return jsonify({
        'status': 'success',
        'query': user_query,
        'structured_query': query_json,
        'results': formatted_results,
        'count': len(formatted_results)
    })

def create_templates_directory():
    """Create the templates directory if it doesn't exist"""
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create index.html template
    index_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OBilet Hotel Room Search</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            .result-card {
                transition: transform 0.3s;
                margin-bottom: 20px;
            }
            .result-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 20px rgba(0,0,0,0.1);
            }
            .match-indicator {
                font-weight: bold;
            }
            .match-true {
                color: green;
            }
            .match-false {
                color: red;
            }
            .feature-list {
                display: flex;
                flex-wrap: wrap;
                gap: 5px;
            }
            .feature-item {
                background-color: #f8f9fa;
                padding: 2px 8px;
                border-radius: 10px;
                font-size: 0.9em;
            }
            .score-bar {
                height: 5px;
                border-radius: 2px;
                margin-bottom: 10px;
            }
            .loader {
                display: none;
                border: 5px solid #f3f3f3;
                border-top: 5px solid #3498db;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 2s linear infinite;
                margin: 20px auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container mt-5">
            <div class="row justify-content-center">
                <div class="col-md-10">
                    <div class="card shadow">
                        <div class="card-header bg-primary text-white">
                            <h1 class="h3 mb-0">OBilet Hotel Room Search System</h1>
                        </div>
                        <div class="card-body">
                            <div class="row mb-4">
                                <div class="col-md-8">
                                    <div class="mb-3">
                                        <label for="queryInput" class="form-label">Search Query:</label>
                                        <input type="text" class="form-control" id="queryInput" 
                                               placeholder="Ex: Double room with sea view">
                                    </div>
                                </div>
                                <div class="col-md-4 d-flex align-items-end">
                                    <button class="btn btn-primary me-2" id="searchBtn">Search</button>
                                    <button class="btn btn-secondary" id="analyzeBtn">Analyze Images</button>
                                </div>
                            </div>
                            
                            <div id="loader" class="loader"></div>
                            
                            <div id="statusBox" class="alert alert-info mb-4" style="display:none;"></div>
                            
                            <div id="resultsContainer">
                                <h2 class="h4 mb-3" id="resultsHeader" style="display:none;">Search Results</h2>
                                <div id="resultsContent" class="row"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const queryInput = document.getElementById('queryInput');
                const searchBtn = document.getElementById('searchBtn');
                const analyzeBtn = document.getElementById('analyzeBtn');
                const statusBox = document.getElementById('statusBox');
                const resultsHeader = document.getElementById('resultsHeader');
                const resultsContent = document.getElementById('resultsContent');
                const loader = document.getElementById('loader');
                
                // Search button click handler
                searchBtn.addEventListener('click', function() {
                    const query = queryInput.value.trim();
                    if (!query) {
                        showStatus('error', 'Please enter a search query.');
                        return;
                    }
                    
                    performSearch(query);
                });
                
                // Analyze button click handler
                analyzeBtn.addEventListener('click', function() {
                    if (confirm('Do you want to analyze all images? This may take some time.')) {
                        analyzeImages(false);
                    }
                });
                
                // Enter key press in query input
                queryInput.addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        searchBtn.click();
                    }
                });
                
                // Perform search
                function performSearch(query) {
                    showLoader(true);
                    showStatus('info', 'Searching...');
                    
                    const formData = new FormData();
                    formData.append('query', query);
                    
                    fetch('/search', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        showLoader(false);
                        
                        if (data.status === 'success') {
                            if (data.count > 0) {
                                showStatus('success', `${data.count} results found for "${query}".`);
                                displayResults(data.results, data.structured_query);
                            } else {
                                showStatus('warning', `No results found for "${query}".`);
                                resultsHeader.style.display = 'none';
                                resultsContent.innerHTML = '';
                            }
                        } else {
                            showStatus('error', data.message || 'An error occurred during search.');
                        }
                    })
                    .catch(error => {
                        showLoader(false);
                        showStatus('error', 'Connection error: ' + error.message);
                    });
                }
                
                // Analyze images
                function analyzeImages(force = false) {
                    showLoader(true);
                    showStatus('info', 'Analyzing images. This may take a while...');
                    
                    const formData = new FormData();
                    formData.append('force', force ? 'true' : 'false');
                    
                    fetch('/analyze', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        showLoader(false);
                        
                        if (data.status === 'success') {
                            showStatus('success', data.message);
                        } else {
                            showStatus('error', data.message || 'An error occurred during analysis.');
                        }
                    })
                    .catch(error => {
                        showLoader(false);
                        showStatus('error', 'Connection error: ' + error.message);
                    });
                }
                
                // Display search results
                function displayResults(results, structuredQuery) {
                    resultsHeader.style.display = 'block';
                    resultsContent.innerHTML = '';
                    
                    results.forEach(result => {
                        const matchRoom = result.matches.room_type ? 'match-true' : 'match-false';
                        const matchCapacity = result.matches.max_capacity ? 'match-true' : 'match-false';
                        const matchView = result.matches.view_type ? 'match-true' : 'match-false';
                        
                        let featuresHtml = '';
                        if (result.features && result.features.length > 0) {
                            featuresHtml = '<div class="feature-list mt-2">';
                            result.features.forEach(feature => {
                                const isMatched = result.matches.features && 
                                                 result.matches.features.matches.includes(feature.toLowerCase());
                                featuresHtml += `<span class="feature-item ${isMatched ? 'match-true' : ''}">${feature}</span>`;
                            });
                            featuresHtml += '</div>';
                        }
                        
                        const resultHtml = `
                            <div class="col-md-6">
                                <div class="card result-card h-100">
                                    <img src="${result.image_url}" class="card-img-top" alt="Room image" style="height: 200px; object-fit: cover;">
                                    <div class="card-body">
                                        <div class="d-flex justify-content-between align-items-start mb-2">
                                            <h5 class="card-title mb-0">#${result.rank} ${result.room_type}</h5>
                                            <span class="badge bg-primary">${result.combined_score}%</span>
                                        </div>
                                        
                                        <div class="score-bar w-100 bg-success" style="opacity: ${result.combined_score/100}"></div>
                                        
                                        <ul class="list-group list-group-flush mb-3">
                                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                                <span class="match-indicator ${matchRoom}">Room Type:</span>
                                                <span>${result.room_type}</span>
                                            </li>
                                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                                <span class="match-indicator ${matchCapacity}">Capacity:</span>
                                                <span>${result.max_capacity} person(s)</span>
                                            </li>
                                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                                <span class="match-indicator ${matchView}">View:</span>
                                                <span>${result.view_type}</span>
                                            </li>
                                        </ul>
                                        
                                        <div class="mb-3">
                                            <strong>Features:</strong>
                                            ${featuresHtml}
                                        </div>
                                        
                                        <p class="card-text small">${result.description}</p>
                                    </div>
                                    <div class="card-footer">
                                        <div class="small text-muted">
                                            <div>Keyword Score: ${result.keyword_score}%</div>
                                            <div>Semantic Score: ${result.semantic_score}%</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        `;
                        
                        resultsContent.innerHTML += resultHtml;
                    });
                }
                
                // Show status message
                function showStatus(type, message) {
                    statusBox.style.display = 'block';
                    statusBox.textContent = message;
                    
                    statusBox.className = 'alert mb-4';
                    switch (type) {
                        case 'success':
                            statusBox.classList.add('alert-success');
                            break;
                        case 'error':
                            statusBox.classList.add('alert-danger');
                            break;
                        case 'warning':
                            statusBox.classList.add('alert-warning');
                            break;
                        default:
                            statusBox.classList.add('alert-info');
                    }
                    
                    // Auto-hide success messages after 5 seconds
                    if (type === 'success') {
                        setTimeout(() => {
                            statusBox.style.display = 'none';
                        }, 5000);
                    }
                }
                
                // Show/hide loader
                function showLoader(show) {
                    loader.style.display = show ? 'block' : 'none';
                }
            });
        </script>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    
    # Write index.html template to the templates directory
    with open(os.path.join(templates_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(index_html)

if __name__ == '__main__':
    # Create templates directory and index.html
    create_templates_directory()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=8080) 