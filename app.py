import os
import numpy as np
import pandas as pd
import faiss
import re
import glob
from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Configuration
RESULTS_DIR = 'Results'
EMBEDDINGS_PATH = os.path.join(RESULTS_DIR, 'embeddings.npy')
MAPPING_PATH = os.path.join(RESULTS_DIR, 'chunk_mapping.csv')
INDEX_PATH = os.path.join(RESULTS_DIR, 'patent_index.faiss')
IMAGES_DIR = os.path.join(RESULTS_DIR, 'Patent_Images')

# Initialize the model and load the data
model = None
faiss_index = None
chunk_mapping = None

def load_resources():
    """Load all the necessary resources for the search."""
    global model, faiss_index, chunk_mapping
    
    try:
        # Load the sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Loaded sentence transformer model")
        
        # Load the FAISS index
        faiss_index = faiss.read_index(INDEX_PATH)
        print(f"✅ Loaded FAISS index with {faiss_index.ntotal} vectors")
        
        # Load the chunk mapping
        chunk_mapping = pd.read_csv(MAPPING_PATH)
        print(f"✅ Loaded mapping data with {len(chunk_mapping)} entries")
        
        return True
    except Exception as e:
        print(f"❌ Error loading resources: {str(e)}")
        return False

def search_patents(query, k=10):
    """Search the patent database for the query."""
    if model is None or faiss_index is None or chunk_mapping is None:
        return {"error": "Search resources not loaded"}
    
    try:
        # Generate embedding for the query
        query_embedding = model.encode([query])[0].reshape(1, -1).astype('float32')
        
        # Search the FAISS index
        distances, indices = faiss_index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunk_mapping):
                patent_id = chunk_mapping.iloc[idx]['patent_id']
                chunk_idx = chunk_mapping.iloc[idx]['chunk_idx']
                
                # Get the text chunk
                chunk_text = get_chunk_text(patent_id, chunk_idx)
                
                results.append({
                    'patent_id': patent_id,
                    'chunk_idx': int(chunk_idx),
                    'distance': float(distances[0][i]),
                    'similarity': float(1.0 / (1.0 + distances[0][i])),
                    'text': chunk_text
                })
        
        return results
    except Exception as e:
        print(f"Error in search: {str(e)}")
        return {"error": str(e)}

def get_chunk_text(patent_id, chunk_idx):
    """Get the text for a specific chunk."""
    try:
        # Try to load from the chunks file
        chunk_file = os.path.join(RESULTS_DIR, f"{patent_id}_chunks.txt")
        
        if os.path.exists(chunk_file):
            with open(chunk_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Split by the separator
            chunks = content.split('='*80)
            
            # Get the specific chunk
            if int(chunk_idx) < len(chunks) - 1:  # -1 because the last split might be empty
                chunk_text = chunks[int(chunk_idx)].strip()
                # Remove the "CHUNK X:" prefix if present
                if chunk_text.startswith(f"CHUNK {int(chunk_idx)+1}:"):
                    chunk_text = chunk_text[len(f"CHUNK {int(chunk_idx)+1}:"):].strip()
                return chunk_text
        
        # Fallback: try to load from the full text file
        text_file = os.path.join(RESULTS_DIR, f"{patent_id}.txt")
        if os.path.exists(text_file):
            with open(text_file, 'r', encoding='utf-8') as f:
                return f"[Full text from {patent_id}]"
        
        return f"[Chunk text not found for {patent_id}, chunk {chunk_idx}]"
    except Exception as e:
        return f"[Error retrieving chunk text: {str(e)}]"

def get_patent_images(patent_id):
    """Get images associated with a patent."""
    images = []
    
    # Look for the patent image directory
    patent_image_dir = os.path.join(IMAGES_DIR, patent_id)
    
    if os.path.exists(patent_image_dir):
        # Get all PNG files
        image_files = glob.glob(os.path.join(patent_image_dir, "*.png"))
        
        # Sort the images by page number
        def extract_page_num(path):
            match = re.search(r'page_(\d+)\.png', path)
            if match:
                return int(match.group(1))
            return 0
            
        image_files.sort(key=extract_page_num)
        
        # Create image data with page numbers
        for img_path in image_files:
            page_num = extract_page_num(img_path)
            rel_path = os.path.relpath(img_path, start=os.getcwd())
            filename = os.path.basename(img_path)
            
            images.append({
                'page': page_num,
                'path': rel_path,
                'filename': filename,
                'url': f'/patent_images/{patent_id}/{filename}'
            })
    
    return images

def get_patent_metadata(patent_id):
    """Extract metadata from the patent ID."""
    # Extract info from the patent ID
    title = patent_id.replace('_', ' ')
    
    # If it's a US patent number, format it nicely
    if patent_id.startswith('US') and any(c.isdigit() for c in patent_id):
        parts = patent_id.split('_')
        patent_number = parts[0]
        title = ' '.join(parts[1:]) if len(parts) > 1 else patent_number
    
    return {
        'id': patent_id,
        'title': title,
        'url': f"https://patents.google.com/patent/{patent_id.split('_')[0]}"
    }

def clean_patent_text(text):
    """Clean the patent text to remove figure and sheet references."""
    # Remove U.S. Patent header lines
    text = re.sub(r'U\.S\. Patent\s+[A-Za-z0-9\.]+\s+Sheet \d+ of \d+\s+US \d+,\d+ [A-Z]\d+', '', text)
    
    # Remove figure captions
    text = re.sub(r'FIG\. \d+[A-Za-z]?( shows| is| illustrates)?', '[FIGURE]', text)
    
    # Clean up page markers
    text = re.sub(r'Page \d+\s*\n', '', text)
    
    return text

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handle search requests."""
    query = request.form.get('query', '')
    
    if not query:
        return jsonify({"error": "No query provided"})
    
    results = search_patents(query)
    
    # Add metadata to results
    for result in results:
        metadata = get_patent_metadata(result['patent_id'])
        result.update(metadata)
    
    return jsonify(results)

@app.route('/patent/<patent_id>')
def patent_detail(patent_id):
    """Show details for a specific patent."""
    # Get the text file
    text_file = os.path.join(RESULTS_DIR, f"{patent_id}.txt")
    text_content = ""
    
    if os.path.exists(text_file):
        with open(text_file, 'r', encoding='utf-8') as f:
            text_content = f.read()
            
        # Clean the text content
        text_content = clean_patent_text(text_content)
    
    # Get patent images
    images = get_patent_images(patent_id)
    
    metadata = get_patent_metadata(patent_id)
    
    return render_template('patent.html', 
                          patent_id=patent_id,
                          metadata=metadata,
                          content=text_content,
                          images=images)

@app.route('/patent_images/<patent_id>/<filename>')
def patent_image(patent_id, filename):
    """Serve patent images."""
    return send_from_directory(os.path.join(IMAGES_DIR, patent_id), filename)

@app.route('/api/patent/<patent_id>/chunks')
def get_patent_chunks(patent_id):
    """API endpoint to get chunks for a patent."""
    chunks = []
    chunk_file = os.path.join(RESULTS_DIR, f"{patent_id}_chunks.txt")
    
    if os.path.exists(chunk_file):
        with open(chunk_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Split by the separator
        raw_chunks = content.split('='*80)
        
        # Process each chunk
        for i, chunk in enumerate(raw_chunks):
            if chunk.strip():
                # Remove the "CHUNK X:" prefix if present
                chunk_text = chunk.strip()
                if chunk_text.startswith(f"CHUNK {i+1}:"):
                    chunk_text = chunk_text[len(f"CHUNK {i+1}:"):].strip()
                
                # Clean the chunk text
                chunk_text = clean_patent_text(chunk_text)
                
                chunks.append({
                    'id': i,
                    'text': chunk_text
                })
    
    return jsonify(chunks)

if __name__ == '__main__':
    # Load resources before starting the app
    if load_resources():
        print("🚀 Starting Patent Search app")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load necessary resources. Make sure the vector database exists.") 