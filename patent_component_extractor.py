"""
Patent Component Extractor

This module extracts structured components from patent OCR text, including:
- Patent number
- Title
- Abstract
- Claims
- Figures
- Body text
"""

import re
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PatentComponentExtractor:
    """
    Extract structured components from patent OCR text
    """
    
    def __init__(self):
        """Initialize the component extractor with regex patterns."""
        self.patterns = {
            'patent_number': r'US\s*([0-9,]{6,})\s*[A-Z][0-9]',
            'title': r'(?:\(54\)|^54\))\s+(.*?)(?=\(|$)',
            'abstract': r'(?:ABSTRACT|^\(57\)\s+ABSTRACT)[\s\n]+(.+?)(?=\n\s*(?:[0-9]+\s+Claims|BRIEF DESCRIPTION|DETAILED DESCRIPTION|BACKGROUND|\[FIGURE\]|FIG\.)|\n\s*$)',
            'claims_start': r'(?:^|\n)(?:What is claimed is:|I claim:|We claim:|Claims?:)(?:\s*\n|\s+)',
            'figures': r'(?:FIG\.?\s*[0-9]+[A-Za-z]*|^\[FIGURE\]).*?(?=FIG\.?\s*[0-9]+[A-Za-z]*|\[FIGURE\]|\n\n|$)',
        }
        
        # Additional patterns for more advanced extraction
        self.secondary_patterns = {
            'patent_number_alt': r'United States Patent\s+([0-9,]+)',
            'title_alt': r'(?:Title:?|Title of Invention:?)\s+([^\n]+)',
            'abstract_alt': r'(?:Abstract|Summary)[\s\n]+(.+?)(?=\n\s*Introduction|\n\s*Background|\n\s*Description|\n\s*$)',
            'body_start': r'(?:DETAILED DESCRIPTION|DESCRIPTION OF (?:THE )?(?:PREFERRED )?EMBODIMENTS?)',
        }
    
    def extract_components(self, text, patent_id=None):
        """
        Extract components from patent text
        
        Args:
            text (str): The OCR-extracted patent text
            patent_id (str, optional): Patent ID for reference
            
        Returns:
            dict: Dictionary of extracted components
        """
        components = {
            'patent_number': None,
            'title': None,
            'abstract': None,
            'claims': [],
            'figures': [],
            'body': None,
        }
        
        # Extract patent number
        components['patent_number'] = self._extract_patent_number(text)
        
        # Extract title
        components['title'] = self._extract_title(text)
        
        # Extract abstract
        components['abstract'] = self._extract_abstract(text)
        
        # Extract claims
        components['claims'] = self._extract_claims(text)
        
        # Extract figures
        components['figures'] = self._extract_figures(text)
        
        # Extract body text
        components['body'] = self._extract_body(text)
        
        # Log extraction results
        patent_ref = patent_id if patent_id else components['patent_number'] or "Unknown Patent"
        logger.info(f"Extracted components from {patent_ref}:")
        for component, value in components.items():
            if isinstance(value, list):
                logger.info(f"  - {component}: {len(value)} items extracted")
            elif value:
                preview = value[:50] + "..." if len(value) > 50 else value
                logger.info(f"  - {component}: {preview}")
            else:
                logger.info(f"  - {component}: Not found")
        
        return components
    
    def _extract_patent_number(self, text):
        """Extract the patent number from text."""
        # Try primary pattern
        match = re.search(self.patterns['patent_number'], text)
        if match:
            return match.group(1).replace(',', '')
        
        # Try alternative pattern
        match = re.search(self.secondary_patterns['patent_number_alt'], text)
        if match:
            return match.group(1).replace(',', '')
        
        return None
    
    def _extract_title(self, text):
        """Extract the patent title from text."""
        # Try primary pattern
        match = re.search(self.patterns['title'], text)
        if match:
            title = match.group(1).strip()
            # Clean up the title
            title = re.sub(r'\s+', ' ', title)
            return title
        
        # Try alternative pattern
        match = re.search(self.secondary_patterns['title_alt'], text)
        if match:
            title = match.group(1).strip()
            # Clean up the title
            title = re.sub(r'\s+', ' ', title)
            return title
        
        # Try to find title at the beginning of the document
        lines = text.split('\n')
        for i in range(min(10, len(lines))):
            line = lines[i].strip()
            if len(line) > 10 and len(line) < 200 and line.isupper():
                return line
        
        return None
    
    def _extract_abstract(self, text):
        """Extract the patent abstract from text."""
        # Try primary pattern
        match = re.search(self.patterns['abstract'], text, re.DOTALL)
        if match:
            abstract = match.group(1).strip()
            # Clean up the abstract
            abstract = re.sub(r'\s+', ' ', abstract)
            return abstract
        
        # Try alternative pattern
        match = re.search(self.secondary_patterns['abstract_alt'], text, re.DOTALL)
        if match:
            abstract = match.group(1).strip()
            # Clean up the abstract
            abstract = re.sub(r'\s+', ' ', abstract)
            return abstract
        
        return None
    
    def _extract_claims(self, text):
        """Extract the patent claims from text."""
        claims = []
        
        # Find the start of claims section
        match = re.search(self.patterns['claims_start'], text)
        if not match:
            return claims
        
        start_pos = match.end()
        claims_text = text[start_pos:]
        
        # Try to find numbered claims
        claim_matches = re.finditer(r'(?:^|\n)([0-9]+)\.\s+(.+?)(?=\n[0-9]+\.\s+|\n\s*$|$)', claims_text, re.DOTALL)
        
        for match in claim_matches:
            claim_num = match.group(1)
            claim_text = match.group(2).strip()
            # Clean up the claim text
            claim_text = re.sub(r'\s+', ' ', claim_text)
            claims.append({
                'number': int(claim_num),
                'text': claim_text
            })
        
        # If no numbered claims found, try to split by paragraphs
        if not claims:
            paragraphs = re.split(r'\n\s*\n', claims_text)
            for i, para in enumerate(paragraphs[:20]):  # Limit to first 20 paragraphs
                if len(para.strip()) > 50:  # Minimum length for a claim
                    claims.append({
                        'number': i + 1,
                        'text': re.sub(r'\s+', ' ', para.strip())
                    })
        
        return claims
    
    def _extract_figures(self, text):
        """Extract figure descriptions from text."""
        figures = []
        
        # Find all figure references
        fig_matches = re.finditer(self.patterns['figures'], text)
        
        for match in fig_matches:
            fig_text = match.group(0).strip()
            # Clean up the figure text
            fig_text = re.sub(r'\s+', ' ', fig_text)
            
            # Try to extract figure number
            fig_num_match = re.search(r'FIG\.?\s*([0-9]+[A-Za-z]*)', fig_text)
            if fig_num_match:
                fig_num = fig_num_match.group(1)
            else:
                fig_num = f"FIGURE_{len(figures)+1}"
            
            figures.append({
                'number': fig_num,
                'text': fig_text
            })
        
        return figures
    
    def _extract_body(self, text):
        """Extract the main body text from the patent."""
        # Try to find the start of the detailed description
        match = re.search(self.secondary_patterns['body_start'], text)
        if match:
            body_start = match.start()
            
            # Try to find the end (usually the claims section)
            claims_start = re.search(self.patterns['claims_start'], text[body_start:])
            if claims_start:
                body_end = body_start + claims_start.start()
                body_text = text[body_start:body_end].strip()
            else:
                body_text = text[body_start:].strip()
                
            return body_text
        
        # Fallback: just take the middle part of the document
        # (after abstract, before claims)
        abstract_match = re.search(self.patterns['abstract'], text, re.DOTALL)
        claims_match = re.search(self.patterns['claims_start'], text)
        
        if abstract_match and claims_match:
            body_text = text[abstract_match.end():claims_match.start()].strip()
            return body_text
        
        # Last resort: just return everything that's not the abstract or claims
        if abstract_match:
            return text[abstract_match.end():].strip()
        if claims_match:
            return text[:claims_match.start()].strip()
        
        # Couldn't identify sections, return None
        return None
    
    def save_components(self, components, output_dir, patent_id):
        """
        Save extracted components to separate files
        
        Args:
            components (dict): Extracted components
            output_dir (str): Directory to save files
            patent_id (str): Patent ID for filenames
        """
        # Create component directory
        component_dir = os.path.join(output_dir, 'Components', patent_id)
        os.makedirs(component_dir, exist_ok=True)
        
        # Save info file (patent number, title, abstract)
        info_path = os.path.join(component_dir, f"{patent_id}_info.txt")
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(f"Patent Number: {components['patent_number'] or 'Unknown'}\n\n")
            f.write(f"Title: {components['title'] or 'Unknown'}\n\n")
            f.write(f"Abstract:\n{components['abstract'] or 'Not available'}\n")
        
        # Save claims
        if components['claims']:
            claims_path = os.path.join(component_dir, f"{patent_id}_claims.txt")
            with open(claims_path, 'w', encoding='utf-8') as f:
                for claim in components['claims']:
                    f.write(f"Claim {claim['number']}:\n")
                    f.write(f"{claim['text']}\n\n")
        
        # Save figures
        if components['figures']:
            figures_path = os.path.join(component_dir, f"{patent_id}_figures.txt")
            with open(figures_path, 'w', encoding='utf-8') as f:
                for figure in components['figures']:
                    f.write(f"Figure {figure['number']}:\n")
                    f.write(f"{figure['text']}\n\n")
        
        # Save body text
        if components['body']:
            body_path = os.path.join(component_dir, f"{patent_id}_body.txt")
            with open(body_path, 'w', encoding='utf-8') as f:
                f.write(components['body'])
        
        # Save summary file
        summary_path = os.path.join(component_dir, f"{patent_id}_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Patent: {components['patent_number'] or patent_id}\n")
            f.write(f"Title: {components['title'] or 'Unknown'}\n\n")
            
            f.write(f"Abstract: {len(components['abstract'] or '') > 0}\n")
            f.write(f"Claims: {len(components['claims'])}\n")
            f.write(f"Figures: {len(components['figures'])}\n")
            f.write(f"Body Text: {len(components['body'] or '') > 0}\n")
        
        logger.info(f"Saved components for {patent_id} to {component_dir}")

def process_patent_text(text_file, output_dir):
    """
    Process a patent text file and extract components
    
    Args:
        text_file (str): Path to the text file
        output_dir (str): Directory to save extracted components
    
    Returns:
        dict: Extracted components
    """
    # Get patent ID from filename
    patent_id = os.path.basename(text_file).split('.')[0]
    
    # Read the text file
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Extract components
    extractor = PatentComponentExtractor()
    components = extractor.extract_components(text, patent_id)
    
    # Save components
    extractor.save_components(components, output_dir, patent_id)
    
    return components

def main():
    """Process patent text files in the Results directory."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract components from patent text files')
    parser.add_argument('--input-dir', '-i', default='Results', help='Directory containing patent text files')
    parser.add_argument('--output-dir', '-o', default='Results', help='Directory to save extracted components')
    parser.add_argument('--patent', '-p', help='Process only the specified patent (filename without extension)')
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.join(args.output_dir, 'Components'), exist_ok=True)
    
    # Get list of text files to process
    if args.patent:
        text_files = [os.path.join(args.input_dir, f"{args.patent}.txt")]
    else:
        text_files = [f for f in os.listdir(args.input_dir) 
                    if f.endswith('.txt') and not f.endswith('_chunks.txt')]
        text_files = [os.path.join(args.input_dir, f) for f in text_files]
    
    logger.info(f"Found {len(text_files)} patent text files to process")
    
    # Process each text file
    for text_file in text_files:
        if os.path.exists(text_file):
            try:
                logger.info(f"Processing {os.path.basename(text_file)}")
                process_patent_text(text_file, args.output_dir)
            except Exception as e:
                logger.error(f"Error processing {text_file}: {str(e)}")
        else:
            logger.warning(f"File not found: {text_file}")
    
    logger.info("Component extraction complete")

if __name__ == "__main__":
    main() 