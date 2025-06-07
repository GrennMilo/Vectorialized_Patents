#!/usr/bin/env python
"""
Patent Processing Script

Process a single patent PDF file, extracting text and images for the web application.
"""

import os
import sys
import argparse
import logging

# Import our tools
from patent_component_extractor import PatentComponentExtractor
from Patents_Vectorizer import PatentOCRVectorizer, check_tesseract, check_poppler, download_nltk_data

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment():
    """Check and setup the environment for patent processing."""
    if not check_tesseract():
        print("❌ Tesseract OCR is not properly installed or configured.")
        return False
        
    if not check_poppler():
        print("❌ Poppler is not properly installed or configured.")
        return False
    
    # Download NLTK data if needed
    download_nltk_data()
    
    return True

def process_patent(pdf_path, output_dir, skip_ocr=False):
    """
    Process a single patent PDF file:
    1. Convert PDF to images (unless skip_ocr is True)
    2. Apply OCR to extract text (unless skip_ocr is True)
    3. Extract patent components (title, abstract, claims, etc.)
    4. Save extracted components to files
    
    Args:
        pdf_path: Path to the patent PDF file
        output_dir: Directory to save the output files
        skip_ocr: If True, assume OCR text already exists and skip OCR step
        
    Returns:
        Dictionary of extracted components if successful, None otherwise
    """
    try:
        # Extract file name without extension
        file_name = os.path.basename(pdf_path)
        file_base = os.path.splitext(file_name)[0]
        
        text_file = os.path.join(output_dir, f"{file_base}.txt")
        
        # If we need to perform OCR
        if not skip_ocr or not os.path.exists(text_file):
            logger.info(f"Running OCR on patent: {file_name}")
            
            # Create OCR vectorizer
            vectorizer = PatentOCRVectorizer(
                pdf_dir=os.path.dirname(pdf_path),
                output_dir=output_dir,
                images_dir=os.path.join(output_dir, "Patent_Images")
            )
            
            # Convert PDF to images and apply OCR
            image_paths, img_dir = vectorizer.convert_pdf_to_images(pdf_path)
            
            if not image_paths:
                logger.error(f"No images generated for {file_name}")
                return None
            
            # Apply OCR to extract text
            text = vectorizer.apply_ocr_to_images(image_paths)
            
            # Check if we got meaningful text
            if len(text.strip()) < 100:  # Arbitrary minimum size
                logger.warning(f"Extracted text is suspiciously short ({len(text)} chars)")
                return None
            
            # Save the full extracted text
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            logger.info(f"Saved OCR text ({len(text)} chars) to {text_file}")
        else:
            # Load existing OCR text
            logger.info(f"Using existing OCR text from {text_file}")
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
        
        # Extract patent components
        component_dir = os.path.join(output_dir, "Components", file_base)
        os.makedirs(component_dir, exist_ok=True)
        
        extractor = PatentComponentExtractor()
        components = extractor.extract_components(text)
        
        # Save components to files
        extractor.save_components_to_files(components, component_dir, file_base)
        
        # Create a summary file with all components
        summary_file = os.path.join(component_dir, f"{file_base}_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Patent: {file_base}\n")
            f.write(f"Number: {components['patent_number']}\n")
            f.write(f"Title: {components['title']}\n\n")
            f.write(f"Abstract:\n{components['abstract']}\n\n")
            f.write(f"Claims ({len(components['claims'])}):\n")
            for i, claim in enumerate(components['claims'], 1):
                f.write(f"  {i}. {claim[:100]}...\n")
            f.write(f"\nFigures ({len(components['figures'])}):\n")
            for i, figure in enumerate(components['figures'], 1):
                f.write(f"  {i}. {figure[:100]}...\n")
            f.write(f"\nBody Text: {len(components['body_text'])} characters\n")
        
        logger.info(f"Component extraction complete for {file_name}")
        logger.info(f"Results saved to {component_dir}")
        
        # Print summary
        print("\n" + "="*60)
        print(f"📄 Patent: {file_base}")
        print(f"🔢 Number: {components['patent_number']}")
        print(f"📝 Title: {components['title']}")
        print(f"📋 Abstract: {components['abstract'][:100]}...")
        print(f"📑 Claims: {len(components['claims'])}")
        print(f"🖼️ Figures: {len(components['figures'])}")
        print(f"📚 Body Text: {len(components['body_text'])} characters")
        print("="*60)
        
        return components
        
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to process a single patent."""
    parser = argparse.ArgumentParser(description="Extract components from a patent PDF")
    parser.add_argument("pdf_file", help="Path to the patent PDF file")
    parser.add_argument("--output", "-o", default="Results", help="Directory to save the output files")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR if text file already exists")
    
    args = parser.parse_args()
    
    # Check if the PDF file exists
    if not os.path.exists(args.pdf_file):
        print(f"❌ Error: File '{args.pdf_file}' does not exist")
        return 1
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    print("="*60)
    print("🔎 Patent Component Extraction 🔍")
    print("="*60)
    print(f"📄 Processing: {args.pdf_file}")
    print(f"📂 Output directory: {args.output}")
    if args.skip_ocr:
        print("⏩ Skipping OCR if text file exists")
    print("="*60)
    
    # Process the patent
    components = process_patent(args.pdf_file, args.output, args.skip_ocr)
    
    if components is None:
        print("\n❌ Failed to process patent")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 