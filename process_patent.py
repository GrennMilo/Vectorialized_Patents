#!/usr/bin/env python
"""
Patent Processing Script

Process a single patent PDF file, extracting text and images for the web application.
"""

import os
import sys
import argparse
from Patents_Vectorizer import PatentOCRVectorizer, check_tesseract, check_poppler, download_nltk_data

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

def process_patent(pdf_path, dpi=300):
    """Process a single patent PDF file."""
    # Ensure the PDF file exists
    if not os.path.isfile(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return False
    
    # Check if the file is a PDF
    if not pdf_path.lower().endswith('.pdf'):
        print(f"❌ File is not a PDF: {pdf_path}")
        return False
    
    print(f"🔍 Processing patent: {os.path.basename(pdf_path)}")
    
    # Initialize the vectorizer
    vectorizer = PatentOCRVectorizer(
        pdf_dir=os.path.dirname(pdf_path),
        output_dir='Results',
        images_dir='Results/Patent_Images'
    )
    
    # Process the patent
    success = vectorizer.process_single_patent(pdf_path, dpi)
    
    if success:
        print("✅ Patent processing completed successfully!")
        
        # Extract the patent ID (filename without extension)
        patent_id = os.path.splitext(os.path.basename(pdf_path))[0]
        
        print(f"📄 Text saved to: Results/{patent_id}.txt")
        print(f"📄 Chunks saved to: Results/{patent_id}_chunks.txt")
        print(f"🖼️ Images saved to: Results/Patent_Images/{patent_id}/")
        
        # Check if we need to regenerate embeddings
        regenerate = input("Do you want to regenerate the vector database with this patent? (y/n): ")
        if regenerate.lower() == 'y':
            print("🧮 Regenerating vector database...")
            vectorizer.generate_embeddings()
            print("✅ Vector database updated!")
    else:
        print("❌ Patent processing failed.")
    
    return success

def main():
    """Main function to process a patent from command line."""
    parser = argparse.ArgumentParser(description='Process a single patent PDF file.')
    parser.add_argument('pdf_path', help='Path to the PDF file to process')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for image conversion (default: 300)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔬 Patent Processing Tool 🔬")
    print("=" * 60)
    
    # Setup the environment
    if not setup_environment():
        print("❌ Environment setup failed. Please fix the issues above.")
        return 1
    
    # Process the patent
    if process_patent(args.pdf_path, args.dpi):
        print("\n🎉 Patent ready for search! You can now use the web application to search this patent.")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main()) 