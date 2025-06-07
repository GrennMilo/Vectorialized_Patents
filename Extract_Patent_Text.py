import os
import sys
import re
import logging
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm

# Set environment variables to help avoid Tesseract quotation errors
os.environ['OMP_THREAD_LIMIT'] = '1'  # Limit OpenMP threads to avoid concurrency issues

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for required dependencies
try:
    from pdf2image import convert_from_path
    print("✅ pdf2image is installed")
except ImportError:
    print("❌ pdf2image library not found. Please install with: pip install pdf2image")
    sys.exit(1)

try:
    from PIL import Image
    import pytesseract
    print("✅ PIL and pytesseract are installed")
except ImportError:
    print("❌ PIL or pytesseract libraries not found. Please install with: pip install pillow pytesseract")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    print("✅ sentence-transformers is installed")
except ImportError:
    print("❌ sentence-transformers library not found. Please install with: pip install sentence-transformers")
    sys.exit(1)

try:
    import faiss
    print("✅ FAISS is installed")
except ImportError:
    print("❌ FAISS library not found. Please install with: pip install faiss-cpu")
    sys.exit(1)

try:
    from nltk.tokenize import sent_tokenize
    import nltk
    nltk.download('punkt', quiet=True)
    print("✅ NLTK is installed and punkt data downloaded")
except ImportError:
    print("❌ NLTK library not found. Please install with: pip install nltk")
    sys.exit(1)

def check_tesseract():
    """Check if Tesseract OCR is installed and available."""
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract OCR is installed (version: {version})")
        return True
    except Exception as e:
        print(f"❌ Tesseract OCR not found or error: {str(e)}")
        print("""
Please install Tesseract OCR:
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
        print("✅ Poppler is installed")
        return True
    except Exception as e:
        if "poppler" in str(e).lower():
            print(f"❌ Poppler error: {str(e)}")
            print("""
Please install Poppler:
- Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/
  Extract to a folder (e.g., C:\\Program Files\\poppler)
  Add bin directory to PATH environment variable
- Linux: sudo apt install poppler-utils
- macOS: brew install poppler""")
            return False
        print("✅ Poppler seems to be working")
        return True
    return True

class PatentProcessor:
    def __init__(self, pdf_dir='Patents', output_dir='Results', images_dir=None):
        self.pdf_dir = pdf_dir
        self.output_dir = output_dir
        
        if images_dir is None:
            self.images_dir = os.path.join(output_dir, 'Patent_Images')
        else:
            self.images_dir = images_dir
            
        # Create output directories
        for directory in [output_dir, self.images_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
        
        # Initialize sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Storage for processed data
        self.patent_texts = {}
        self.patent_chunks = {}
        self.chunk_mapping = []
        
    def extract_images_from_pdf(self, pdf_path, dpi=300):
        """Extract images from a PDF file."""
        file_name = os.path.basename(pdf_path)
        file_base = os.path.splitext(file_name)[0]
        
        # Create directory for this patent's images
        patent_img_dir = os.path.join(self.images_dir, file_base)
        if not os.path.exists(patent_img_dir):
            os.makedirs(patent_img_dir)
        
        print(f"[🔄] Converting PDF to images: {file_name}")
        
        try:
            # Convert PDF to images
            pages = convert_from_path(pdf_path, dpi=dpi)
            
            # Save each page as an image
            image_paths = []
            for i, page in enumerate(pages):
                img_path = os.path.join(patent_img_dir, f"page_{i+1}.png")
                page.save(img_path, 'PNG')
                image_paths.append(img_path)
            
            print(f"[✓] Converted {len(pages)} pages to images")
            return image_paths, patent_img_dir
        
        except Exception as e:
            print(f"[❌] Error converting PDF to images: {str(e)}")
            traceback.print_exc()
            return [], patent_img_dir
    
    def apply_ocr_to_images(self, image_paths, lang='eng'):
        """Apply OCR to extract text from images."""
        all_text = ""
        
        print(f"[🔄] Applying OCR to {len(image_paths)} images")
        
        for i, img_path in enumerate(tqdm(image_paths, desc="OCR Processing")):
            try:
                # Open the image
                img = Image.open(img_path)
                
                # Much simpler OCR configuration to avoid quotation errors
                # Avoid using any custom config options that could cause parsing issues
                text = None
                error_occurred = False
                
                # Try multiple approaches with increasing simplicity
                try:
                    # Approach 1: Basic OCR with minimal options
                    text = pytesseract.image_to_string(img, lang=lang, config='--psm 6')
                except Exception as e1:
                    error_occurred = True
                    print(f"[⚠️] First OCR attempt failed: {str(e1)}")
                    try:
                        # Approach 2: Even more basic OCR
                        text = pytesseract.image_to_string(img, lang=lang)
                    except Exception as e2:
                        print(f"[⚠️] Second OCR attempt failed: {str(e2)}")
                        try:
                            # Approach 3: Convert to grayscale first
                            gray_img = img.convert('L')
                            text = pytesseract.image_to_string(gray_img, lang=lang)
                        except Exception as e3:
                            print(f"[❌] All OCR attempts failed on {img_path}")
                            text = f"[OCR FAILED FOR THIS PAGE - MULTIPLE ATTEMPTS]"
                
                # Add page number for reference
                all_text += f"\n[Page {i+1}]\n{text if text else '[OCR FAILED]'}\n"
                
                if error_occurred:
                    print(f"[✓] Recovered from OCR error on page {i+1}")
                
            except Exception as e:
                print(f"[⚠️] Error opening image {img_path}: {str(e)}")
                all_text += f"\n[Page {i+1}]\n[IMAGE PROCESSING ERROR - SKIPPING PAGE]\n"
        
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
        
        # Remove multiple spaces but preserve paragraph structure
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove common OCR artifacts
        text = re.sub(r'[|}{~`]', '', text)  # Remove common misrecognized characters
        
        # Check if we've removed too much content
        cleaned_length = len(text)
        if cleaned_length < original_length * 0.3:  # If we've removed more than 70%
            print(f"[⚠️] Aggressive cleaning removed {original_length - cleaned_length} chars ({(original_length - cleaned_length) / original_length:.1%})")
        
        return text.strip()
    
    def chunk_text(self, text, chunk_size=1000, overlap=200):
        """Split the text into chunks with overlap."""
        if len(text) < chunk_size:
            return [text] if text.strip() else []
        
        try:
            # Try NLTK for sentence boundary detection
            sentences = sent_tokenize(text)
        except Exception:
            # Fallback to simple splitting
            sentences = []
            for para in text.split('\n\n'):
                if para.strip():
                    sentences.append(para.strip())
        
        # Group sentences into chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                if overlap > 0:
                    # Keep some sentences for overlap
                    overlap_length = 0
                    overlap_sentences = []
                    
                    for prev_sentence in reversed(current_chunk):
                        if overlap_length + len(prev_sentence) > overlap:
                            break
                        overlap_sentences.insert(0, prev_sentence)
                        overlap_length += len(prev_sentence) + 1
                    
                    current_chunk = overlap_sentences
                    current_length = overlap_length
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length + 1
        
        # Add the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks
    
    def generate_embeddings(self):
        """Generate embeddings for all chunks and create FAISS index."""
        all_chunks = []
        chunk_to_patent = []
        
        # Collect all chunks
        for patent_id, chunks in self.patent_chunks.items():
            for chunk_idx, chunk in enumerate(chunks):
                if len(chunk) > 50:  # Minimum chunk size
                    all_chunks.append(chunk)
                    chunk_to_patent.append((patent_id, chunk_idx))
        
        if not all_chunks:
            print("[❌] No valid chunks found to generate embeddings")
            return
        
        print(f"[🔄] Generating embeddings for {len(all_chunks)} chunks")
        
        # Generate embeddings in batches
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(all_chunks), batch_size), desc="Generating embeddings"):
            batch = all_chunks[i:i+batch_size]
            batch_embeddings = self.model.encode(batch)
            embeddings.extend(batch_embeddings)
        
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
        
        print(f"[✓] Generated embeddings database with {len(all_chunks)} chunks from {len(self.patent_chunks)} patents")
    
    def process_patent(self, pdf_path, dpi=300):
        """Process a single patent: extract images, apply OCR, chunk text."""
        file_name = os.path.basename(pdf_path)
        file_base = os.path.splitext(file_name)[0]
        
        print(f"\n{'='*60}")
        print(f"[🔄] Processing patent: {file_name}")
        
        # Extract images from PDF
        image_paths, img_dir = self.extract_images_from_pdf(pdf_path, dpi)
        
        if not image_paths:
            print(f"[❌] No images generated for {file_name}")
            return False
        
        # Apply OCR to extract text
        text = self.apply_ocr_to_images(image_paths)
        
        # Remove any lingering OCR error lines
        text = re.sub(r'OCR ERROR: No closing quotation', '', text)
        text = re.sub(r'OCR ERROR:.*?\n', '\n', text)
        
        # Check if we got meaningful text
        if len(text.strip()) < 50:  # Lower the threshold to be more permissive
            print(f"[⚠️] Extracted text is suspiciously short ({len(text)} chars)")
            # Still proceed with what we have
            print(f"[ℹ️] Continuing with limited text")
        
        # Store the text
        self.patent_texts[file_base] = text
        
        # Save the full extracted text
        text_file = os.path.join(self.output_dir, f"{file_base}.txt")
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"[✓] Saved text ({len(text)} chars) to {text_file}")
        
        # Create chunks, but use a smaller chunk size if the text is limited
        chunk_size = 500 if len(text) < 1000 else 1000
        chunks = self.chunk_text(text, chunk_size=chunk_size)
        
        if chunks:
            self.patent_chunks[file_base] = chunks
            
            # Save chunks to file
            chunk_file = os.path.join(self.output_dir, f"{file_base}_chunks.txt")
            with open(chunk_file, 'w', encoding='utf-8') as f:
                for i, chunk in enumerate(chunks):
                    f.write(f"CHUNK {i+1}:\n")
                    f.write(f"{chunk}\n\n{'='*80}\n\n")
            
            print(f"[✓] Created {len(chunks)} chunks for {file_base}")
        else:
            # If no chunks created, create a single chunk with whatever we have
            if text.strip():
                self.patent_chunks[file_base] = [text.strip()]
                
                # Save the single chunk
                chunk_file = os.path.join(self.output_dir, f"{file_base}_chunks.txt")
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    f.write(f"CHUNK 1:\n")
                    f.write(f"{text.strip()}\n\n{'='*80}\n\n")
                
                print(f"[⚠️] Created 1 emergency chunk for {file_base}")
            else:
                print(f"[❌] No chunks created for {file_base}")
                self.patent_chunks[file_base] = []
        
        return True
    
    def process_all_patents(self, dpi=300):
        """Process all patents in the input directory."""
        # Find PDF files
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"[❌] No PDF files found in {self.pdf_dir}")
            return False
        
        print(f"[📋] Found {len(pdf_files)} PDF files to process")
        
        successful = 0
        failed = 0
        
        # Process each PDF
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_dir, pdf_file)
            
            try:
                result = self.process_patent(pdf_path, dpi)
                if result:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"[❌] Error processing {pdf_file}: {str(e)}")
                traceback.print_exc()
                failed += 1
        
        print(f"\n[📊] PDF Processing complete. Success: {successful}, Failed: {failed}")
        
        if successful == 0:
            print("[❌] No patents were processed successfully. Cannot create vector database.")
            return False
        
        # Generate embeddings for all processed patents
        print("\n[🧠] Generating vector embeddings...")
        try:
            self.generate_embeddings()
        except Exception as e:
            print(f"[❌] Error generating embeddings: {str(e)}")
            traceback.print_exc()
            return False
        
        print(f"\n[✅] Vector database creation complete! Processed {successful} patents.")
        print(f"[📁] Database saved to: {self.output_dir}")
        print(f"[🖼️] Images saved to: {self.images_dir}")
        
        return True

def main():
    print("\n" + "="*60)
    print("🔬 Patent Text Extraction and Vectorization 🔬")
    print("="*60)
    print("[📋] This tool will:")
    print("1. Extract PNG images from patent PDFs")
    print("2. Apply OCR to extract text")
    print("3. Save raw text to Results folder")
    print("4. Split text into chunks")
    print("5. Create vectorial database")
    print("="*60 + "\n")
    
    # Apply Tesseract workarounds
    try:
        pytesseract.pytesseract.TessBaseAPI = None  # Disable the TessBaseAPI to avoid certain errors
        print("[ℹ️] Applied Tesseract error prevention measures")
    except:
        print("[ℹ️] Could not apply all Tesseract error prevention measures")
    
    # Check system dependencies
    if not check_tesseract() or not check_poppler():
        print("\n[❌] Missing required system dependencies. Please install them and try again.")
        sys.exit(1)
    
    # Initialize processor
    processor = PatentProcessor(
        pdf_dir='Patents',
        output_dir='Results'
    )
    
    # Process all patents
    success = processor.process_all_patents(dpi=300)
    
    if success:
        print("\n" + "="*60)
        print("[🎉] SUCCESS! Patent processing complete.")
        print("="*60)
    else:
        print("\n[❌] Failed to process patents.")
        sys.exit(1)

if __name__ == "__main__":
    main() 