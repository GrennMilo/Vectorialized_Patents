import os
import sys
import argparse
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Import our component extractor
from patent_component_extractor import PatentComponentExtractor

# Use the existing OCR tools from PatentsVectorizer
try:
    from Patents_Vectorizer import PatentOCRVectorizer
except ImportError:
    print("Error: Patents_Vectorizer.py not found in the current directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_single_patent(pdf_path, output_dir, ocr_only=False):
    """
    Process a single patent PDF file:
    1. Convert PDF to images
    2. Apply OCR to extract text
    3. Extract patent components (title, abstract, claims, etc.)
    4. Save extracted components to files
    
    Args:
        pdf_path: Path to the patent PDF file
        output_dir: Directory to save the output files
        ocr_only: If True, only perform OCR without component extraction
        
    Returns:
        True if processing was successful, False otherwise
    """
    try:
        # Extract file name without extension
        file_name = os.path.basename(pdf_path)
        file_base = os.path.splitext(file_name)[0]
        
        logger.info(f"Processing patent: {file_name}")
        
        # Create OCR vectorizer (we'll only use its OCR functionality)
        vectorizer = PatentOCRVectorizer(
            pdf_dir=os.path.dirname(pdf_path),
            output_dir=output_dir,
            images_dir=os.path.join(output_dir, "Patent_Images")
        )
        
        # Convert PDF to images and apply OCR
        image_paths, img_dir = vectorizer.convert_pdf_to_images(pdf_path)
        
        if not image_paths:
            logger.error(f"No images generated for {file_name}")
            return False
        
        # Apply OCR to extract text
        text = vectorizer.apply_ocr_to_images(image_paths)
        
        # Check if we got meaningful text
        if len(text.strip()) < 100:  # Arbitrary minimum size
            logger.warning(f"Extracted text is suspiciously short ({len(text)} chars)")
            return False
        
        # Save the full extracted text
        text_file = os.path.join(output_dir, f"{file_base}.txt")
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        logger.info(f"Saved OCR text ({len(text)} chars) to {text_file}")
        
        # If only OCR is requested, we're done
        if ocr_only:
            return True
        
        # Extract patent components
        component_dir = os.path.join(output_dir, "Components", file_base)
        os.makedirs(component_dir, exist_ok=True)
        
        extractor = PatentComponentExtractor()
        components = extractor.extract_components(text)
        
        # Save components to files
        extractor.save_components_to_files(components, component_dir, file_base)
        
        # Create a summary file
        summary_file = os.path.join(component_dir, f"{file_base}_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Patent: {file_base}\n")
            f.write(f"Number: {components['patent_number']}\n")
            f.write(f"Title: {components['title']}\n\n")
            f.write(f"Abstract:\n{components['abstract']}\n\n")
            f.write(f"Number of Claims: {len(components['claims'])}\n")
            f.write(f"Number of Figures: {len(components['figures'])}\n")
            f.write(f"Body Text: {len(components['body_text'])} characters\n")
        
        logger.info(f"Component extraction complete for {file_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def process_patents(input_dir, output_dir, max_workers=None, ocr_only=False, limit=None):
    """
    Process all patent PDFs in the input directory.
    
    Args:
        input_dir: Directory containing patent PDF files
        output_dir: Directory to save the output files
        max_workers: Maximum number of worker processes (None = auto)
        ocr_only: If True, only perform OCR without component extraction
        limit: Maximum number of patents to process (None = all)
        
    Returns:
        Tuple of (successful_count, failed_count)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PDF files in the input directory
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        logger.error(f"No PDF files found in {input_dir}")
        return 0, 0
    
    # Limit the number of files if requested
    if limit and limit < len(pdf_files):
        pdf_files = pdf_files[:limit]
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    successful = 0
    failed = 0
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_patent, 
                           os.path.join(input_dir, pdf_file), 
                           output_dir,
                           ocr_only): pdf_file 
            for pdf_file in pdf_files
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing patents"):
            pdf_file = futures[future]
            try:
                result = future.result()
                if result:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
                failed += 1
    
    logger.info(f"Processing complete. Success: {successful}, Failed: {failed}")
    return successful, failed

def main():
    """Main function to run the patent processing pipeline."""
    parser = argparse.ArgumentParser(description="Process patent PDFs with OCR and component extraction")
    parser.add_argument("--input", "-i", default="Patents", help="Directory containing patent PDF files")
    parser.add_argument("--output", "-o", default="Results", help="Directory to save the output files")
    parser.add_argument("--workers", "-w", type=int, default=None, help="Maximum number of worker processes")
    parser.add_argument("--ocr-only", action="store_true", help="Only perform OCR without component extraction")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Maximum number of patents to process")
    
    args = parser.parse_args()
    
    print("="*60)
    print("🔎 Patent OCR & Component Extraction 🔍")
    print("="*60)
    print(f"📂 Input directory: {args.input}")
    print(f"📂 Output directory: {args.output}")
    if args.ocr_only:
        print("🔍 Mode: OCR only (no component extraction)")
    else:
        print("🔍 Mode: Full OCR and component extraction")
    if args.limit:
        print(f"⚠️ Processing limited to {args.limit} patents")
    print("="*60)
    
    # Check if the input directory exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Input directory '{args.input}' does not exist")
        return 1
    
    # Process the patents
    successful, failed = process_patents(
        args.input, 
        args.output, 
        max_workers=args.workers,
        ocr_only=args.ocr_only,
        limit=args.limit
    )
    
    print("="*60)
    print(f"✅ Successfully processed: {successful} patents")
    if failed > 0:
        print(f"❌ Failed to process: {failed} patents")
    print(f"📂 Results saved to: {args.output}")
    print("="*60)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 