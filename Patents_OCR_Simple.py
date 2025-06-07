import os
import sys
import logging
import traceback
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for required dependencies before importing them
dependencies_ok = True

try:
    from pdf2image import convert_from_path
    print("✅ pdf2image is installed")
except ImportError:
    logger.error("pdf2image library not found. Please install with: pip install pdf2image")
    dependencies_ok = False

try:
    from PIL import Image
    import pytesseract
    print("✅ PIL and pytesseract are installed")
except ImportError:
    logger.error("PIL or pytesseract libraries not found. Please install with: pip install pillow pytesseract")
    dependencies_ok = False

def check_tesseract():
    """Check if Tesseract OCR is installed and available."""
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract OCR is installed (version: {version})")
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
        # Try a simple conversion to see if poppler is available
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pdf') as f:
            f.write(b"%PDF-1.0\n1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n2 0 obj\n<</Type/Pages/Kids[3 0 R]/Count 1>>\nendobj\n3 0 obj\n<</Type/Page/MediaBox[0 0 3 3]>>\nendobj\nxref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\ntrailer\n<</Size 4/Root 1 0 R>>\nstartxref\n149\n%EOF\n")
            f.flush()
            try:
                convert_from_path(f.name, first_page=1, last_page=1)
                print("✅ Poppler is installed and working")
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
                    print(f"✅ Poppler seems to be working (non-poppler error: {str(e)})")
                    return True
    except Exception as e:
        logger.error(f"Error checking poppler: {str(e)}")
        return False

def extract_images_from_pdf(pdf_path, output_dir):
    """Extract images from a PDF file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    file_name = os.path.basename(pdf_path)
    file_base = os.path.splitext(file_name)[0]
    
    # Create directory for this patent's images
    patent_img_dir = os.path.join(output_dir, file_base)
    if not os.path.exists(patent_img_dir):
        os.makedirs(patent_img_dir)
    
    print(f"Converting PDF to images: {file_name}")
    
    try:
        # Convert PDF to images
        pages = convert_from_path(pdf_path, dpi=300)
        
        # Save each page as an image
        image_paths = []
        for i, page in enumerate(pages):
            img_path = os.path.join(patent_img_dir, f"page_{i+1}.png")
            page.save(img_path, 'PNG')
            image_paths.append(img_path)
        
        print(f"  Converted {len(pages)} pages to images in {patent_img_dir}")
        return image_paths, patent_img_dir
    
    except Exception as e:
        logger.error(f"  Error converting PDF to images: {str(e)}")
        traceback.print_exc()
        return [], patent_img_dir

def apply_ocr_to_image(image_path, output_dir):
    """Apply OCR to a single image and save the text."""
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Apply OCR
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(img, lang='eng', config=custom_config)
        
        # Get filename for saving text
        base_name = os.path.basename(image_path)
        base_name_no_ext = os.path.splitext(base_name)[0]
        
        # Save the text
        text_path = os.path.join(output_dir, f"{base_name_no_ext}.txt")
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return text_path, len(text)
    except Exception as e:
        logger.error(f"Error in OCR for {image_path}: {str(e)}")
        return None, 0

def main():
    # Check dependencies
    if not dependencies_ok:
        logger.error("Required Python libraries are missing. Please install them first.")
        sys.exit(1)
    
    # Check system dependencies
    print("\nChecking system dependencies:")
    tesseract_ok = check_tesseract()
    poppler_ok = check_poppler()
    
    if not tesseract_ok or not poppler_ok:
        logger.error("Missing required system dependencies. Please install them and try again.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("🔬 Simple Patent OCR Test 🔬")
    print("="*60)
    
    # Define directories
    pdf_dir = 'Patents'
    output_dir = 'Results'
    images_dir = os.path.join(output_dir, 'Patent_Images')
    
    # Create output directories if they don't exist
    for directory in [output_dir, images_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    # Find PDF files
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_dir}")
        sys.exit(1)
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process a limited number of PDFs for testing
    test_limit = 1  # Process only the first PDF for testing
    processed_pdfs = 0
    
    for pdf_file in pdf_files[:test_limit]:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        
        try:
            # Extract images from PDF
            image_paths, img_dir = extract_images_from_pdf(pdf_path, images_dir)
            
            if not image_paths:
                print(f"  No images generated for {pdf_file}")
                continue
            
            # Create text output directory
            text_dir = os.path.join(output_dir, os.path.splitext(pdf_file)[0])
            if not os.path.exists(text_dir):
                os.makedirs(text_dir)
            
            # Apply OCR to the first few images (for testing)
            ocr_limit = min(3, len(image_paths))  # Process up to 3 images for testing
            print(f"Applying OCR to {ocr_limit} of {len(image_paths)} images")
            
            for i, img_path in enumerate(image_paths[:ocr_limit]):
                print(f"  Processing image {i+1}/{ocr_limit}: {os.path.basename(img_path)}")
                text_path, text_length = apply_ocr_to_image(img_path, text_dir)
                
                if text_path:
                    print(f"    OCR complete: Extracted {text_length} characters to {text_path}")
                else:
                    print(f"    OCR failed for this image")
            
            processed_pdfs += 1
            print(f"Completed processing {pdf_file} ({processed_pdfs}/{test_limit})")
            
        except Exception as e:
            logger.error(f"  Error processing {pdf_file}: {str(e)}")
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"Processing complete! Processed {processed_pdfs} patents.")
    print("="*60)

if __name__ == "__main__":
    main() 