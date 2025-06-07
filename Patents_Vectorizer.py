import os
import sys
import re
import logging
import traceback
import textwrap
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for required dependencies before importing them
dependencies_ok = True

try:
    from pdf2image import convert_from_path
except ImportError:
    logger.error("pdf2image library not found. Please install with: pip install pdf2image")
    dependencies_ok = False

try:
    from PIL import Image
    import pytesseract
except ImportError:
    logger.error("PIL or pytesseract libraries not found. Please install with: pip install pillow pytesseract")
    dependencies_ok = False

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.error("sentence-transformers library not found. Please install with: pip install sentence-transformers")
    dependencies_ok = False

try:
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
except ImportError:
    logger.error("scikit-learn or matplotlib not found. Please install with: pip install scikit-learn matplotlib")
    dependencies_ok = False

try:
    import faiss
except ImportError:
    logger.error("FAISS library not found. Please install with: pip install faiss-cpu")
    dependencies_ok = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize
except ImportError:
    logger.error("NLTK library not found. Please install with: pip install nltk")
    dependencies_ok = False

def check_tesseract():
    """Check if Tesseract OCR is installed and available."""
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        logger.error("""
Tesseract OCR not found. Please install:
- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
- Linux: sudo apt install tesseract-ocr
- macOS: brew install tesseract
Then ensure it's in your PATH environment variable.""")
        return False

def check_poppler():
    """Check if Poppler is installed and available for pdf2image."""
    try:
        # Try a simple conversion to see if poppler is available
        convert_from_path("dummy", first_page=1, last_page=1)
    except FileNotFoundError:
        # This is expected since "dummy" doesn't exist
        # But if we get here, poppler is likely available
        return True
    except Exception as e:
        if "poppler" in str(e).lower():
            logger.error("""
Poppler not found. Please install:
- Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/
  Extract to a folder (e.g., C:\\Program Files\\poppler)
  Add bin directory to PATH environment variable
- Linux: sudo apt install poppler-utils
- macOS: brew install poppler""")
            return False
        return True
    return True

def download_nltk_data():
    """Download required NLTK data."""
    try:
        nltk.download('punkt', quiet=True)
        logger.info("NLTK punkt data downloaded successfully")
        return True
    except Exception as e:
        logger.warning(f"Could not download NLTK punkt data: {str(e)}")
        return False

class PatentOCRVectorizer:
    """
    Complete Patent OCR and Vectorization System
    
    This class handles the entire pipeline from PDF files to a searchable vector database:
    1. Convert PDF pages to images
    2. Apply OCR to extract text
    3. Clean and chunk the text
    4. Generate vector embeddings
    5. Create FAISS index for similarity search
    6. Save the vectorized database
    """
    
    def __init__(self, pdf_dir='Patents', output_dir='Results', images_dir='Results/Patent_Images'):
        self.pdf_dir = pdf_dir
        self.output_dir = output_dir
        self.images_dir = images_dir
        
        # Initialize the sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Data storage
        self.patent_texts = {}
        self.patent_chunks = {}
        self.patent_embeddings = {}
        self.faiss_index = None
        self.chunk_mapping = []
        
        # Create output directories if they don't exist
        for directory in [output_dir, images_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
    
    def convert_pdf_to_images(self, pdf_path, dpi=300):
        """Convert PDF to images using pdf2image."""
        file_name = os.path.basename(pdf_path)
        file_base = os.path.splitext(file_name)[0]
        
        # Create directory for this patent's images
        patent_img_dir = os.path.join(self.images_dir, file_base)
        if not os.path.exists(patent_img_dir):
            os.makedirs(patent_img_dir)
        
        logger.info(f"Converting PDF to images: {file_name}")
        
        try:
            # Convert PDF to images
            pages = convert_from_path(pdf_path, dpi=dpi)
            
            # Save each page as an image
            image_paths = []
            for i, page in enumerate(pages):
                img_path = os.path.join(patent_img_dir, f"page_{i+1}.png")
                page.save(img_path, 'PNG')
                image_paths.append(img_path)
            
            logger.info(f"  Converted {len(pages)} pages to images")
            return image_paths, patent_img_dir
        
        except Exception as e:
            logger.error(f"  Error converting PDF to images: {str(e)}")
            traceback.print_exc()
            return [], patent_img_dir
    
    def apply_ocr_to_images(self, image_paths, lang='eng'):
        """Apply OCR to extract text from images."""
        all_text = ""
        
        logger.info(f"Applying OCR to {len(image_paths)} images")
        
        for i, img_path in enumerate(tqdm(image_paths, desc="OCR Processing")):
            try:
                # Open the image
                img = Image.open(img_path)
                
                # Apply OCR with enhanced configuration for patents
                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?()[]{}"-/ '
                text = pytesseract.image_to_string(img, lang=lang, config=custom_config)
                
                # Add page number for reference
                all_text += f"\n[Page {i+1}]\n{text}\n"
                
            except Exception as e:
                logger.error(f"  Error processing image {img_path}: {str(e)}")
                all_text += f"\n[Page {i+1}]\nOCR ERROR: {str(e)}\n"
        
        # Clean up the extracted text
        all_text = self._clean_ocr_text(all_text)
        
        return all_text
    
    def _clean_ocr_text(self, text):
        """Clean the OCR extracted text."""
        # Save original length for comparison
        original_length = len(text)
        
        # Remove multiple blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove line numbers (typically appearing at start of line)
        text = re.sub(r'^\s*\d+\s*', '', text, flags=re.MULTILINE)
        
        # Remove page numbering artifacts
        text = re.sub(r'\b\d+\s*\|\s*\w+', '', text)
        
        # Remove headers/footers that typically appear on patent pages
        text = re.sub(r'^US\s+\d+\s+[A-Z]\d+\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'U\.S\. Patent\s+[A-Za-z0-9\.]+\s+Sheet \d+ of \d+\s+US \d+,\d+ [A-Z]\d+', '', text)
        
        # Handle figure references
        text = re.sub(r'FIG\. \d+[A-Za-z]?( shows| is| illustrates)?', '[FIGURE]', text)
        
        # Clean up page markers
        text = re.sub(r'Page \d+\s*\n', '', text)
        
        # Keep paragraph numbers but remove brackets (like [0046])
        text = re.sub(r'\[\s*(\d+)\s*\]', r'\1. ', text)
        
        # Remove multiple spaces but preserve paragraph structure
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove common OCR artifacts
        text = re.sub(r'[|}{~`]', '', text)  # Remove common misrecognized characters
        text = re.sub(r'\s+([.,:;!?])', r'\1', text)  # Fix spacing before punctuation
        
        # Check if we've removed too much content
        cleaned_length = len(text)
        if cleaned_length < original_length * 0.3:  # If we've removed more than 70%
            logger.warning(f"Aggressive cleaning removed {original_length - cleaned_length} chars ({(original_length - cleaned_length) / original_length:.1%})")
        
        return text.strip()
    
    def chunk_text(self, text, chunk_size=1000, overlap=200):
        """Split the text into chunks with overlap while respecting sentence boundaries."""
        # If text is too short, return as single chunk
        if len(text) < chunk_size:
            return [text] if text.strip() else []
        
        # Remove paragraph numbers in brackets before tokenizing into sentences
        cleaned_text = re.sub(r'\[\s*\d+\s*\]', '', text)
        
        # Extract sentences from the text
        sentences = []
        
        try:
            # Try NLTK first for better sentence boundary detection
            sentences = sent_tokenize(cleaned_text)
            logger.debug(f"NLTK tokenizer found {len(sentences)} sentences")
        except Exception as e:
            logger.warning(f"NLTK tokenization failed: {str(e)}")
            
            # Fallback to custom sentence splitting with regex
            sentences = []
            # Split on periods, question marks, and exclamation points followed by space or newline
            sentence_parts = re.split(r'(?<=[.!?])(?=\s|$)', cleaned_text)
            for part in sentence_parts:
                part = part.strip()
                if len(part) > 10:  # Avoid very short fragments
                    sentences.append(part)
            logger.debug(f"Custom tokenizer found {len(sentences)} sentences")
        
        # If still no sentences, try paragraph splitting
        if not sentences:
            logger.warning("No sentences found, using text paragraphs instead")
            # Fallback to paragraph splitting
            sentences = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]
            
            # If still empty, use line-by-line
            if not sentences:
                sentences = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
                
            # Last resort: just use the whole text as one chunk
            if not sentences:
                sentences = [cleaned_text]
        
        # Group sentences into chunks while respecting sentence boundaries
        chunks = []
        current_chunk_sentences = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed the chunk size and we already have content
            if current_length + sentence_length > chunk_size and current_chunk_sentences:
                # Save current chunk
                chunk_text = ' '.join(current_chunk_sentences)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                if overlap > 0:
                    # Calculate how many sentences to keep for overlap
                    overlap_sentences = []
                    overlap_length = 0
                    
                    # Add sentences from end of previous chunk for overlap
                    for prev_sentence in reversed(current_chunk_sentences):
                        if overlap_length + len(prev_sentence) > overlap:
                            break
                        overlap_sentences.insert(0, prev_sentence)
                        overlap_length += len(prev_sentence) + 1  # +1 for space
                    
                    # Start new chunk with overlap sentences
                    current_chunk_sentences = overlap_sentences
                    current_length = overlap_length
                else:
                    # No overlap
                    current_chunk_sentences = []
                    current_length = 0
            
            # Handle case where a single sentence is longer than chunk_size
            if sentence_length > chunk_size and not current_chunk_sentences:
                # Just add this long sentence as its own chunk
                chunks.append(sentence)
                current_chunk_sentences = []
                current_length = 0
                continue
            
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_length += sentence_length + 1  # +1 for space
        
        # Add the last chunk if not empty
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append(chunk_text)
        
        return chunks
    
    def generate_embeddings(self):
        """Generate embeddings for each patent chunk."""
        all_chunks = []
        chunk_to_patent = []
        
        # Collect all chunks and track their source
        for patent_id, chunks in self.patent_chunks.items():
            for chunk_idx, chunk in enumerate(chunks):
                # Only include chunks with meaningful content
                if len(chunk) > 50:  # Minimum chunk size
                    all_chunks.append(chunk)
                    chunk_to_patent.append((patent_id, chunk_idx))
        
        if not all_chunks:
            logger.error("No valid chunks found to generate embeddings")
            return
            
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks")
        
        # Generate embeddings in batches
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(all_chunks), batch_size), desc="Generating embeddings"):
            batch = all_chunks[i:i+batch_size]
            try:
                batch_embeddings = self.model.encode(batch)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i}: {str(e)}")
                # Add zeros for failed batch to maintain alignment
                zero_embedding = np.zeros((len(batch), self.model.get_sentence_embedding_dimension()))
                embeddings.extend(zero_embedding)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Store mapping for retrieval
        self.chunk_mapping = chunk_to_patent
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings_array)
        
        # Save embeddings and mapping
        np.save(os.path.join(self.output_dir, 'embeddings.npy'), embeddings_array)
        pd.DataFrame(self.chunk_mapping, columns=['patent_id', 'chunk_idx']).to_csv(
            os.path.join(self.output_dir, 'chunk_mapping.csv'), index=False
        )
        
        # Save faiss index
        faiss.write_index(self.faiss_index, os.path.join(self.output_dir, 'patent_index.faiss'))
        
        logger.info(f"Generated embeddings for {len(all_chunks)} chunks from {len(self.patent_chunks)} patents")
    
    def cluster_embeddings(self, n_clusters=15):
        """Cluster the embeddings and visualize."""
        if self.faiss_index is None:
            logger.error("No embeddings found. Run generate_embeddings first.")
            return
        
        # Get embeddings from FAISS index
        try:
            # This might not work with all FAISS versions
            raw_index = faiss.downcast_index(self.faiss_index)
            if isinstance(raw_index, faiss.IndexFlat):
                embeddings = raw_index.xb.reshape(raw_index.ntotal, -1)
            else:
                logger.error("Unsupported index type for clustering")
                return
        except Exception as e:
            logger.error(f"Error extracting index data: {str(e)}")
            return
            
        logger.info(f"Clustering {len(embeddings)} embeddings into {n_clusters} clusters")
        
        # Run K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Reduce dimensionality for visualization
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            reduced_embeddings[:, 0], 
            reduced_embeddings[:, 1], 
            c=cluster_labels, 
            cmap='viridis', 
            alpha=0.7
        )
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'Patent Chunks Clustered into {n_clusters} Groups')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cluster_visualization.png'), dpi=300)
        plt.close()
        
        # Save cluster assignments
        cluster_df = pd.DataFrame({
            'patent_id': [m[0] for m in self.chunk_mapping],
            'chunk_idx': [m[1] for m in self.chunk_mapping],
            'cluster': cluster_labels
        })
        cluster_df.to_csv(os.path.join(self.output_dir, 'cluster_assignments.csv'), index=False)
        
        # Analyze clusters
        self._analyze_clusters(cluster_df, n_clusters)
        
        logger.info(f"Clustered {len(embeddings)} embeddings into {n_clusters} clusters")
    
    def _analyze_clusters(self, cluster_df, n_clusters):
        """Analyze the clusters and create a summary."""
        analysis_file = os.path.join(self.output_dir, 'cluster_analysis.txt')
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write(f"Patent Chunk Cluster Analysis\n")
            f.write(f"==============================\n\n")
            
            for cluster_id in range(n_clusters):
                cluster_rows = cluster_df[cluster_df['cluster'] == cluster_id]
                f.write(f"Cluster {cluster_id}: {len(cluster_rows)} chunks\n")
                f.write(f"------------------------------------------------\n")
                
                # Count patents in this cluster
                patents_in_cluster = cluster_rows['patent_id'].unique()
                f.write(f"Patents: {len(patents_in_cluster)}\n")
                
                # Sample some chunks from this cluster
                sample_size = min(5, len(cluster_rows))
                if sample_size > 0:
                    sample_rows = cluster_rows.sample(sample_size, random_state=42)
                    
                    f.write("\nSample chunks from this cluster:\n\n")
                    
                    for _, row in sample_rows.iterrows():
                        patent_id = row['patent_id']
                        chunk_idx = row['chunk_idx']
                        
                        # Make sure the patent and chunk exist
                        if patent_id in self.patent_chunks and chunk_idx < len(self.patent_chunks[patent_id]):
                            chunk_text = self.patent_chunks[patent_id][chunk_idx]
                            
                            f.write(f"Patent: {patent_id}\n")
                            f.write(f"Chunk: {chunk_idx+1}\n")
                            # Get first 300 characters as a preview
                            preview = chunk_text[:300] + '...' if len(chunk_text) > 300 else chunk_text
                            preview = textwrap.fill(preview, width=80)
                            f.write(f"{preview}\n\n")
                        else:
                            f.write(f"Patent: {patent_id} (Chunk {chunk_idx+1} not found)\n\n")
                
                f.write(f"\n{'='*50}\n\n")
    
    def process_single_patent(self, pdf_path, dpi=300):
        """Process a single patent PDF with OCR and vectorization."""
        # Extract file name without extension
        file_name = os.path.basename(pdf_path)
        file_base = os.path.splitext(file_name)[0]
        
        logger.info(f"Processing patent: {file_name}")
        
        # Convert PDF to images
        image_paths, img_dir = self.convert_pdf_to_images(pdf_path, dpi)
        
        if not image_paths:
            logger.error(f"  No images generated for {file_name}")
            return False
        
        # Apply OCR to extract text
        text = self.apply_ocr_to_images(image_paths)
        
        # Check if we got meaningful text
        if len(text.strip()) < 100:  # Arbitrary minimum size
            logger.warning(f"  Extracted text is suspiciously short ({len(text)} chars)")
            return False
        
        # Store the text
        self.patent_texts[file_base] = text
        
        # Save the full extracted text
        text_file = os.path.join(self.output_dir, f"{file_base}.txt")
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        logger.info(f"  Saved text ({len(text)} chars) to {text_file}")
        
        # Create chunks
        chunks = self.chunk_text(text)
        
        if chunks:
            self.patent_chunks[file_base] = chunks
            
            # Save chunks to file
            chunk_file = os.path.join(self.output_dir, f"{file_base}_chunks.txt")
            with open(chunk_file, 'w', encoding='utf-8') as f:
                for i, chunk in enumerate(chunks):
                    f.write(f"CHUNK {i+1}:\n")
                    wrapped_text = textwrap.fill(chunk, width=100)
                    f.write(f"{wrapped_text}\n\n{'='*80}\n\n")
            
            logger.info(f"  Created {len(chunks)} chunks for {file_base}")
        else:
            logger.warning(f"  No chunks created for {file_base}")
            self.patent_chunks[file_base] = []
        
        return True
    
    def process_all_patents(self, dpi=300):
        """Process all PDFs in the input directory with complete OCR and vectorization pipeline."""
        
        # Check required system dependencies first
        if not check_tesseract() or not check_poppler():
            logger.error("Missing required system dependencies. Processing aborted.")
            return False

        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.error(f"No PDF files found in {self.pdf_dir}")
            return False
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        successful = 0
        failed = 0
        
        # Process each PDF
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_dir, pdf_file)
            
            try:
                result = self.process_single_patent(pdf_path, dpi)
                if result:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"  Error processing {pdf_file}: {str(e)}")
                traceback.print_exc()
                failed += 1
        
        logger.info(f"PDF Processing complete. Success: {successful}, Failed: {failed}")
        
        if successful == 0:
            logger.error("No patents were processed successfully. Cannot create vector database.")
            return False
        
        # Generate embeddings for all processed patents
        logger.info("Generating vector embeddings...")
        try:
            self.generate_embeddings()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            traceback.print_exc()
            return False
        
        # Create clusters and visualizations
        logger.info("Creating clusters and visualizations...")
        try:
            self.cluster_embeddings(n_clusters=min(15, max(3, successful // 2)))
        except Exception as e:
            logger.error(f"Error during clustering: {str(e)}")
            logger.info("Continuing despite clustering error...")
        
        logger.info(f"Vector database creation complete! Processed {successful} patents.")
        logger.info(f"Database saved to: {self.output_dir}")
        logger.info(f"Images saved to: {self.images_dir}")
        
        return True
    
    def search_patents(self, query, k=5):
        """Search the vectorial database for similar patents."""
        if self.faiss_index is None:
            logger.error("No index found. Run process_all_patents first.")
            return []
        
        logger.info(f"Searching for: '{query}'")
        
        # Generate embedding for the query
        try:
            query_embedding = self.model.encode([query])[0].reshape(1, -1).astype('float32')
        except Exception as e:
            logger.error(f"Error encoding query: {str(e)}")
            return []
        
        # Search the FAISS index
        try:
            distances, indices = self.faiss_index.search(query_embedding, k)
        except Exception as e:
            logger.error(f"Error searching FAISS index: {str(e)}")
            return []
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunk_mapping):
                patent_id, chunk_idx = self.chunk_mapping[idx]
                
                # Make sure the patent and chunk exist
                if patent_id in self.patent_chunks and chunk_idx < len(self.patent_chunks[patent_id]):
                    chunk_text = self.patent_chunks[patent_id][chunk_idx]
                    
                    results.append({
                        'patent_id': patent_id,
                        'chunk_idx': chunk_idx,
                        'distance': distances[0][i],
                        'similarity': 1.0 / (1.0 + distances[0][i]),  # Convert distance to similarity
                        'text': chunk_text
                    })
        
        return results

def main():
    """Main function to run the complete patent OCR and vectorization pipeline."""
    
    # Check dependencies first
    if not dependencies_ok:
        logger.error("Required Python libraries are missing. Please install them first.")
        logger.error("Run: pip install pdf2image pillow pytesseract sentence-transformers faiss-cpu scikit-learn matplotlib nltk")
        sys.exit(1)
    
    # Download NLTK data
    download_nltk_data()
    
    print("="*60)
    print("🔬 Patent OCR Vectorization System 🔬")
    print("="*60)
    print("📋 This system will:")
    print("1. 📄 Convert PDF pages to images")
    print("2. 🔍 Apply OCR to extract text")
    print("3. 🧹 Clean and chunk the text")
    print("4. 🧠 Generate vector embeddings")
    print("5. 🔎 Create searchable FAISS database")
    print("="*60)
    
    # Initialize the vectorizer
    vectorizer = PatentOCRVectorizer(
        pdf_dir='Patents',
        output_dir='Results',
        images_dir='Results/Patent_Images'
    )
    
    # Process all patents
    success = vectorizer.process_all_patents(dpi=300)
    
    if success:
        print("="*60)
        print("✅ SUCCESS! Patent vectorial database created.")
        print("="*60)
        print("🔍 You can now search using:")
        print("  results = vectorizer.search_patents('your query here')")
        print("  for result in results:")
        print("      print(f'Patent: {result[\"patent_id\"]}, Similarity: {result[\"similarity\"]:.3f}')")
        print("      print(result['text'][:200] + '...')")
        print("="*60)
        
        # Example search to demonstrate functionality
        logger.info("Running example search for 'energy storage'...")
        try:
            example_results = vectorizer.search_patents("energy storage", k=3)
            if example_results:
                print(f"[!] Found {len(example_results)} results:")
                for i, result in enumerate(example_results):
                    print(f"{i+1}. Patent: {result['patent_id']}, Similarity: {result['similarity']:.3f}")
                    print(f"   Preview: {result['text'][:150]}...")
            else:
                print("[!] No results found for example search.")
        except Exception as e:
            logger.error(f"Error in example search: {str(e)}")
        
    else:
        print("❌ Failed to create patent vectorial database.")
        sys.exit(1)

if __name__ == "__main__":
    main()