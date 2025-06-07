#!/usr/bin/env python
"""
Complete Patent Processing Script

This script handles the entire patent processing pipeline:
1. Extract PNG images from all PDFs in the Patents folder
2. Apply OCR to extract text from the images
3. Clean and chunk the text
4. Create vector embeddings for semantic search
"""

import os
import sys
import re
import glob
import time
import logging
import traceback
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check dependencies
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
    import faiss
except ImportError:
    logger.error("FAISS library not found. Please install with: pip install faiss-cpu")
    dependencies_ok = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    nltk.download('punkt', quiet=True)
except ImportError:
    logger.error("NLTK library not found. Please install with: pip install nltk")
    dependencies_ok = False

def print_header(title):
    """Print a nice header in the terminal."""
    print("\n" + "=" * 70)
    print(f"{title}")
    print("=" * 70)

def check_tesseract():
    """Check if Tesseract OCR is installed and available."""
    try:
        version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract OCR is installed (version: {version})")
        return True
    except Exception as e:
        logger.error(f"""
Tesseract OCR not found or error: {str(e)}
Please install:
- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
- Linux: sudo apt install tesseract-ocr
- macOS: brew install tesseract
Then ensure it's in your PATH environment variable.""")
        return False

def check_poppler():
    """Check if Poppler is installed and available for pdf2image."""
    try:
        # Create a minimal valid PDF in memory to test poppler
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pdf') as f:
            f.write(b"%PDF-1.0\n1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n2 0 obj\n<</Type/Pages/Kids[3 0 R]/Count 1>>\nendobj\n3 0 obj\n<</Type/Page/MediaBox[0 0 3 3]>>\nendobj\nxref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\ntrailer\n<</Size 4/Root 1 0 R>>\nstartxref\n149\n%EOF\n")
            f.flush()
            try:
                # Try to convert the first page only - this will verify poppler works
                convert_from_path(f.name, first_page=1, last_page=1)
                logger.info("Poppler is installed and working")
                return True
            except Exception as e:
                if "poppler" in str(e).lower():
                    logger.error(f"""
Poppler error: {str(e)}
Please install:
- Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/
  Extract to a folder (e.g., C:\\Program Files\\poppler)
  Add bin directory to PATH environment variable
- Linux: sudo apt install poppler-utils
- macOS: brew install poppler""")
                    return False
                else:
                    logger.info(f"Poppler seems to be working (non-poppler error: {str(e)})")
                    return True
    except Exception as e:
        logger.error(f"Error checking poppler: {str(e)}")
        return False

def safe_ocr(image_path, lang='eng'):
    """Apply OCR with additional error handling and retries."""
    # Open the image
    try:
        img = Image.open(image_path)
    except Exception as e:
        logger.error(f"Error opening image {image_path}: {str(e)}")
        return f"[ERROR: Could not open image - {str(e)}]"

    # Try OCR with multiple approaches
    for attempt in range(3):
        try:
            # Different approaches for each attempt
            if attempt == 0:
                # First attempt: Basic settings
                text = pytesseract.image_to_string(img, lang=lang)
            elif attempt == 1:
                # Second attempt: Convert to grayscale
                gray_img = img.convert('L')
                text = pytesseract.image_to_string(gray_img, lang=lang)
            else:
                # Third attempt: Resize image
                resized_img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
                text = pytesseract.image_to_string(resized_img, lang=lang)
            
            # If we got here, OCR worked
            return text
        
        except Exception as e:
            logger.warning(f"OCR attempt {attempt+1} failed for {image_path}: {str(e)}")
            # Wait a moment before retry
            time.sleep(0.5)
    
    # If all attempts failed
    return f"[OCR FAILED AFTER MULTIPLE ATTEMPTS]"

def extract_images_from_pdf(pdf_path, output_dir, dpi=300):
    """Extract images from a PDF file with improved error handling."""
    file_name = os.path.basename(pdf_path)
    file_base = os.path.splitext(file_name)[0]
    
    # Create directory for this patent's images
    patent_img_dir = os.path.join(output_dir, file_base)
    if not os.path.exists(patent_img_dir):
        os.makedirs(patent_img_dir)
    
    logger.info(f"Converting PDF to images: {file_name}")
    
    try:
        # Try to convert PDF to images
        # Set thread_count=1 to avoid threading issues
        pages = convert_from_path(pdf_path, dpi=dpi, thread_count=1)
        
        # Save each page as an image
        image_paths = []
        for i, page in enumerate(pages):
            img_path = os.path.join(patent_img_dir, f"page_{i+1}.png")
            
            # Make sure we don't try to save to an image that's in use
            if os.path.exists(img_path):
                try:
                    # Try to remove existing file if it exists
                    os.remove(img_path)
                except Exception:
                    # If we can't remove it, use a different filename
                    img_path = os.path.join(patent_img_dir, f"page_{i+1}_{int(time.time())}.png")
            
            # Save the image
            page.save(img_path, 'PNG')
            image_paths.append(img_path)
        
        logger.info(f"  Converted {len(pages)} pages to images")
        return image_paths, patent_img_dir
    
    except Exception as e:
        logger.error(f"  Error converting PDF to images: {str(e)}")
        traceback.print_exc()
        return [], patent_img_dir

def apply_ocr_to_images(image_paths, lang='eng'):
    """Apply OCR to extract text from images."""
    all_text = ""
    
    logger.info(f"Applying OCR to {len(image_paths)} images")
    
    for i, img_path in enumerate(tqdm(image_paths, desc="OCR Processing")):
        # Use our safer OCR function
        text = safe_ocr(img_path, lang)
        
        # Add page number for reference
        all_text += f"\n[Page {i+1}]\n{text}\n"
    
    return all_text

def clean_ocr_text(text):
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

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split the text into chunks with overlap while respecting sentence boundaries."""
    # If text is too short, return as single chunk
    if len(text) < chunk_size:
        return [text] if text.strip() else []
    
    # Try NLTK for sentence boundary detection
    try:
        sentences = sent_tokenize(text)
        logger.debug(f"NLTK tokenizer found {len(sentences)} sentences")
    except Exception as e:
        logger.warning(f"NLTK tokenization failed: {str(e)}")
        
        # Fallback to simple paragraph splitting
        sentences = []
        for para in text.split('\n\n'):
            if para.strip():
                sentences.append(para.strip())
        
        logger.debug(f"Fallback tokenizer found {len(sentences)} paragraphs")
    
    # If still no sentences, use basic line splitting
    if not sentences:
        sentences = [line.strip() for line in text.split('\n') if line.strip()]
        logger.debug(f"Last resort tokenizer found {len(sentences)} lines")
    
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
            continue
        
        # Add sentence to current chunk
        current_chunk_sentences.append(sentence)
        current_length += sentence_length + 1  # +1 for space
    
    # Add the last chunk if not empty
    if current_chunk_sentences:
        chunk_text = ' '.join(current_chunk_sentences)
        chunks.append(chunk_text)
    
    return chunks

def generate_embeddings(patent_chunks, output_dir):
    """Generate embeddings for all chunks and create FAISS index."""
    # Initialize the model
    logger.info("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    all_chunks = []
    chunk_to_patent = []
    
    # Collect all chunks
    for patent_id, chunks in patent_chunks.items():
        for chunk_idx, chunk in enumerate(chunks):
            if len(chunk) > 50:  # Minimum chunk size
                all_chunks.append(chunk)
                chunk_to_patent.append((patent_id, chunk_idx))
    
    if not all_chunks:
        logger.error("No valid chunks found to generate embeddings")
        return False
    
    logger.info(f"Generating embeddings for {len(all_chunks)} chunks")
    
    # Generate embeddings in batches
    batch_size = 32
    embeddings = []
    
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Generating embeddings"):
        batch = all_chunks[i:i+batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    dimension = embeddings_array.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings_array)
    
    # Save embeddings and mapping
    np.save(os.path.join(output_dir, 'embeddings.npy'), embeddings_array)
    pd.DataFrame(chunk_to_patent, columns=['patent_id', 'chunk_idx']).to_csv(
        os.path.join(output_dir, 'chunk_mapping.csv'), index=False
    )
    
    # Save faiss index
    faiss.write_index(faiss_index, os.path.join(output_dir, 'patent_index.faiss'))
    
    logger.info(f"Generated embeddings database with {len(all_chunks)} chunks from {len(patent_chunks)} patents")
    return True

def process_patent(pdf_path, output_dir, images_dir, dpi=300):
    """Process a single patent: extract images, apply OCR, chunk text."""
    file_name = os.path.basename(pdf_path)
    file_base = os.path.splitext(file_name)[0]
    
    logger.info(f"Processing patent: {file_name}")
    
    # Extract images from PDF
    image_paths, img_dir = extract_images_from_pdf(pdf_path, images_dir, dpi)
    
    if not image_paths:
        logger.error(f"No images generated for {file_name}")
        return None, None
    
    # Apply OCR to extract text
    text = apply_ocr_to_images(image_paths)
    
    # Clean the extracted text
    text = clean_ocr_text(text)
    
    # Check if we got meaningful text
    if len(text.strip()) < 100:  # Arbitrary minimum size
        logger.warning(f"Extracted text is suspiciously short ({len(text)} chars)")
        if len(text.strip()) < 50:
            logger.error(f"Text too short to be useful, skipping this patent")
            return None, None
    
    # Save the full extracted text
    text_file = os.path.join(output_dir, f"{file_base}.txt")
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    logger.info(f"Saved text ({len(text)} chars) to {text_file}")
    
    # Create chunks
    chunks = chunk_text(text)
    
    if chunks:
        # Save chunks to file
        chunk_file = os.path.join(output_dir, f"{file_base}_chunks.txt")
        with open(chunk_file, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                f.write(f"CHUNK {i+1}:\n")
                f.write(f"{chunk}\n\n{'='*80}\n\n")
        
        logger.info(f"Created {len(chunks)} chunks for {file_base}")
    else:
        logger.warning(f"No chunks created for {file_base}")
        chunks = []
    
    return file_base, chunks

def process_all_patents(pdf_dir='Patents', output_dir='Results', images_dir=None, dpi=300, limit=None):
    """Process all patents in the input directory."""
    # Setup directories
    if images_dir is None:
        images_dir = os.path.join(output_dir, 'Patent_Images')
    
    # Create output directories if they don't exist
    for directory in [output_dir, images_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
    
    # Find PDF files
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_dir}")
        return False
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Apply limit if specified
    if limit and limit > 0:
        pdf_files = pdf_files[:limit]
        logger.info(f"Limited processing to {limit} files")
    
    patent_chunks = {}
    successful = 0
    failed = 0
    
    # Process each PDF
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        
        try:
            patent_id, chunks = process_patent(pdf_path, output_dir, images_dir, dpi)
            
            if patent_id:
                patent_chunks[patent_id] = chunks
                successful += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {str(e)}")
            traceback.print_exc()
            failed += 1
    
    logger.info(f"PDF Processing complete. Success: {successful}, Failed: {failed}")
    
    if successful == 0:
        logger.error("No patents were processed successfully. Cannot create vector database.")
        return False
    
    # Generate embeddings for all processed patents
    logger.info("Generating vector embeddings...")
    try:
        generate_embeddings(patent_chunks, output_dir)
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        traceback.print_exc()
        return False
    
    logger.info(f"Vector database creation complete! Processed {successful} patents.")
    logger.info(f"Database saved to: {output_dir}")
    logger.info(f"Images saved to: {images_dir}")
    
    return True

def clean_and_restart():
    """Clean previous results and restart processing."""
    print_header("🧹 Cleaning Previous Results")
    
    results_dir = 'Results'
    
    # Make sure the directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        logger.info(f"Created directory: {results_dir}")
        return True
    
    # Ask for confirmation
    print("This will delete all previously processed results.")
    confirm = input("Are you sure you want to continue? (y/n): ")
    
    if confirm.lower() != 'y':
        logger.info("Operation cancelled.")
        return False
    
    # Delete vector database files
    db_files = ['embeddings.npy', 'chunk_mapping.csv', 'patent_index.faiss']
    for file in db_files:
        file_path = os.path.join(results_dir, file)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Deleted: {file}")
            except Exception as e:
                logger.error(f"Error deleting {file}: {str(e)}")
    
    # Delete text and chunk files
    for file in os.listdir(results_dir):
        if file.endswith('.txt'):
            file_path = os.path.join(results_dir, file)
            try:
                os.remove(file_path)
                logger.info(f"Deleted: {file}")
            except Exception as e:
                logger.error(f"Error deleting {file}: {str(e)}")
    
    # Handle the images directory
    images_dir = os.path.join(results_dir, 'Patent_Images')
    if os.path.exists(images_dir):
        clean_images = input("Do you want to delete all extracted images too? (y/n): ")
        
        if clean_images.lower() == 'y':
            try:
                shutil.rmtree(images_dir)
                os.makedirs(images_dir)
                logger.info(f"Cleaned and recreated: {images_dir}")
            except Exception as e:
                logger.error(f"Error cleaning images directory: {str(e)}")
    
    logger.info("Cleanup complete.")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Process patents to create a searchable vector database.')
    parser.add_argument('--pdf-dir', default='Patents', help='Directory containing patent PDFs')
    parser.add_argument('--output-dir', default='Results', help='Directory to save processed results')
    parser.add_argument('--images-dir', default=None, help='Directory to save extracted images (default: Results/Patent_Images)')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for image conversion (default: 300)')
    parser.add_argument('--limit', type=int, default=0, help='Limit the number of patents to process (default: process all)')
    parser.add_argument('--clean', action='store_true', help='Clean previous results before processing')
    
    args = parser.parse_args()
    
    print_header("🔬 Patent Processing System 🔬")
    print("This system will:")
    print("1. 📄 Convert PDF pages to images")
    print("2. 🔍 Apply OCR to extract text")
    print("3. 🧹 Clean and chunk the text")
    print("4. 🧠 Generate vector embeddings")
    print("5. 🔎 Create searchable FAISS database")
    
    # Check dependencies
    if not dependencies_ok:
        logger.error("Required Python libraries are missing. Please install them first.")
        return 1
    
    # Check system dependencies
    if not check_tesseract() or not check_poppler():
        logger.error("Missing required system dependencies. Please install them and try again.")
        return 1
    
    # Clean previous results if requested
    if args.clean:
        if not clean_and_restart():
            return 1
    
    # Process all patents
    success = process_all_patents(
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        images_dir=args.images_dir,
        dpi=args.dpi,
        limit=args.limit
    )
    
    if success:
        print_header("✅ SUCCESS! Patent vectorial database created.")
        return 0
    else:
        logger.error("Failed to create patent vectorial database.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 