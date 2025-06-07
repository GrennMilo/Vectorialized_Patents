import re
import os
import logging
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PatentComponentExtractor:
    """
    Extract key components from patent OCR text files.
    
    This class parses OCR-extracted text from patent documents to identify and
    extract key components such as:
    - Patent number
    - Title
    - Abstract
    - Claims
    - Figures descriptions
    - Main body text
    """
    
    def __init__(self):
        # Patterns for identifying patent sections
        self.patterns = {
            'patent_number': r'US\s*([0-9,]{6,})\s*[A-Z][0-9]',
            'title': r'(?:\(54\)|^54\))\s+(.*?)(?=\(|$)',
            'abstract': r'(?:ABSTRACT|^\(57\)\s+ABSTRACT)[\s\n]+(.+?)(?=\n\s*(?:[0-9]+\s+Claims|BRIEF DESCRIPTION|DETAILED DESCRIPTION|BACKGROUND|\[FIGURE\]|FIG\.)|\n\s*$)',
            'claims_start': r'(?:^|\n)(?:What is claimed is:|I claim:|We claim:|Claims?:)(?:\s*\n|\s+)',
            'claims_end': r'(?:\n\s*(?:DETAILED DESCRIPTION|DESCRIPTION OF|BACKGROUND|\[FIGURE\]|FIG\.)|\Z)',
            'figures': r'(?:FIG\.?\s*[0-9]+[A-Za-z]*|^\[FIGURE\]).*?(?=FIG\.?\s*[0-9]+[A-Za-z]*|\[FIGURE\]|\n\n|$)',
        }
        
    def extract_components(self, text: str) -> Dict[str, Any]:
        """
        Extract all components from patent text.
        
        Args:
            text: OCR-extracted text from a patent document
            
        Returns:
            Dictionary containing extracted components
        """
        # Clean the text first
        text = self._clean_text(text)
        
        # Extract components
        components = {
            'patent_number': self._extract_patent_number(text),
            'title': self._extract_title(text),
            'abstract': self._extract_abstract(text),
            'claims': self._extract_claims(text),
            'figures': self._extract_figures(text),
            'body_text': self._extract_body_text(text)
        }
        
        return components
    
    def _clean_text(self, text: str) -> str:
        """Clean the OCR text for better extraction."""
        # Remove multiple blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove line numbers at start of lines
        text = re.sub(r'^\s*\d+\s*', '', text, flags=re.MULTILINE)
        
        # Remove page numbering artifacts
        text = re.sub(r'\b\d+\s*\|\s*\w+', '', text)
        
        # Remove headers/footers that typically appear on patent pages
        text = re.sub(r'^US\s+\d+\s+[A-Z]\d+\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'U\.S\. Patent\s+[A-Za-z0-9\.]+\s+Sheet \d+ of \d+\s+US \d+,\d+ [A-Z]\d+', '', text)
        
        # Handle page markers
        text = re.sub(r'\[Page \d+\]', '', text)
        text = re.sub(r'Page \d+\s*\n', '', text)
        
        # Remove common OCR artifacts
        text = re.sub(r'[|}{~`]', '', text)
        text = re.sub(r'\s+([.,:;!?])', r'\1', text)
        
        return text.strip()
    
    def _extract_patent_number(self, text: str) -> str:
        """Extract the patent number."""
        match = re.search(self.patterns['patent_number'], text)
        if match:
            # Clean and format the patent number
            number = match.group(1).replace(',', '')
            return f"US{number}"
        return ""
    
    def _extract_title(self, text: str) -> str:
        """Extract the patent title."""
        # Try to find title in multiple ways
        
        # Method 1: Look for title after (54) marker
        match = re.search(self.patterns['title'], text, re.IGNORECASE | re.DOTALL)
        if match:
            title = match.group(1).strip()
            # Clean up title (often split across multiple lines)
            title = re.sub(r'\s+', ' ', title)
            return title
            
        # Method 2: Look for title in first few lines (often appears near the top)
        lines = text.split('\n')[:10]  # Check first 10 lines
        for i, line in enumerate(lines):
            if '(54)' in line or line.strip().startswith('54)'):
                # Title likely in this line or next
                title_text = line.replace('(54)', '').replace('54)', '').strip()
                
                # Title might continue to next line
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith('('):
                    title_text += ' ' + lines[j].strip()
                    j += 1
                
                return title_text.strip()
        
        return ""
    
    def _extract_abstract(self, text: str) -> str:
        """Extract the abstract."""
        # Try different abstract patterns
        match = re.search(self.patterns['abstract'], text, re.IGNORECASE | re.DOTALL)
        if match:
            abstract = match.group(1).strip()
            # Clean up abstract
            abstract = re.sub(r'\s+', ' ', abstract)
            return abstract
            
        # If not found with regex, try locating by section markers
        if '(57)' in text or 'ABSTRACT' in text.upper():
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if '(57)' in line or 'ABSTRACT' in line.upper():
                    # Abstract starts here, collect until next section
                    abstract_lines = []
                    j = i + 1
                    while j < len(lines) and not any(marker in lines[j].upper() for marker in 
                                                   ['BACKGROUND', 'FIELD OF', 'DESCRIPTION', 'BRIEF', 'CLAIMS']):
                        if lines[j].strip():  # Skip empty lines
                            abstract_lines.append(lines[j].strip())
                        j += 1
                    
                    if abstract_lines:
                        return ' '.join(abstract_lines)
        
        return ""
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract the claims section as a list of individual claims."""
        # Find the claims section
        claims_section = ""
        match_start = re.search(self.patterns['claims_start'], text, re.IGNORECASE)
        
        if match_start:
            start_pos = match_start.end()
            match_end = re.search(self.patterns['claims_end'], text[start_pos:], re.IGNORECASE)
            
            if match_end:
                end_pos = start_pos + match_end.start()
                claims_section = text[start_pos:end_pos].strip()
            else:
                claims_section = text[start_pos:].strip()
        
        # If claims section not found, try another approach
        if not claims_section:
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if any(claim_marker in line for claim_marker in ['What is claimed is:', 'I claim:', 'We claim:', 'Claims:']):
                    # Claims start here, collect until next section
                    claims_lines = []
                    j = i + 1
                    while j < len(lines) and not any(marker in lines[j].upper() for marker in 
                                                   ['DETAILED DESCRIPTION', 'BACKGROUND', 'DESCRIPTION OF']):
                        if lines[j].strip():  # Skip empty lines
                            claims_lines.append(lines[j].strip())
                        j += 1
                    
                    if claims_lines:
                        claims_section = ' '.join(claims_lines)
                        break
        
        # Parse individual claims
        claims = []
        if claims_section:
            # Split by claim numbers
            claim_pattern = r'(?:^|\n|\s)(\d+\.\s+.+?)(?=\n\s*\d+\.\s+|\Z)'
            found_claims = re.findall(claim_pattern, claims_section, re.DOTALL)
            
            if found_claims:
                claims = [claim.strip() for claim in found_claims]
            else:
                # If no numbered claims found, return the whole section
                claims = [claims_section]
        
        return claims
    
    def _extract_figures(self, text: str) -> List[str]:
        """Extract figure descriptions and captions."""
        figures = []
        
        # Find all figure references
        fig_matches = re.findall(self.patterns['figures'], text, re.MULTILINE | re.IGNORECASE)
        
        # Clean up the matches
        for match in fig_matches:
            # Remove excessive whitespace
            cleaned_match = re.sub(r'\s+', ' ', match).strip()
            if cleaned_match:
                figures.append(cleaned_match)
        
        return figures
    
    def _extract_body_text(self, text: str) -> str:
        """
        Extract the main body text of the patent.
        This typically includes background, detailed description, etc.
        """
        # Find start of body text (usually after abstract and before claims)
        body_start = None
        body_end = None
        
        # Try to find starting point (after abstract)
        abstract_match = re.search(r'(?:ABSTRACT|^\(57\)\s+ABSTRACT)', text, re.IGNORECASE)
        if abstract_match:
            # Look for the next section heading after abstract
            section_match = re.search(r'\n\s*(?:BACKGROUND|FIELD OF|SUMMARY|DETAILED DESCRIPTION)', text[abstract_match.end():], re.IGNORECASE)
            if section_match:
                body_start = abstract_match.end() + section_match.start()
        
        # Try to find ending point (before claims)
        claims_match = re.search(self.patterns['claims_start'], text, re.IGNORECASE)
        if claims_match:
            body_end = claims_match.start()
        
        # Extract body text if start and end found
        if body_start is not None and body_end is not None and body_start < body_end:
            body_text = text[body_start:body_end].strip()
        else:
            # Fallback: try to find the body between known section headings
            lines = text.split('\n')
            body_lines = []
            in_body = False
            
            for line in lines:
                if not in_body and re.search(r'^(?:BACKGROUND|FIELD OF|SUMMARY|DETAILED DESCRIPTION)', line, re.IGNORECASE):
                    in_body = True
                    body_lines.append(line)
                elif in_body and re.search(r'^(?:CLAIMS|What is claimed)', line, re.IGNORECASE):
                    in_body = False
                elif in_body:
                    body_lines.append(line)
            
            body_text = '\n'.join(body_lines).strip()
        
        return body_text
    
    def process_patent_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a patent OCR text file and extract all components.
        
        Args:
            file_path: Path to the patent OCR text file
            
        Returns:
            Dictionary containing extracted components
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            return self.extract_components(text)
            
        except Exception as e:
            logger.error(f"Error processing patent file {file_path}: {str(e)}")
            return {
                'patent_number': '',
                'title': '',
                'abstract': '',
                'claims': [],
                'figures': [],
                'body_text': ''
            }
    
    def save_components_to_files(self, components: Dict[str, Any], output_dir: str, base_filename: str) -> None:
        """
        Save extracted components to separate text files.
        
        Args:
            components: Dictionary of extracted components
            output_dir: Directory to save the files
            base_filename: Base name for the output files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save patent info
        with open(os.path.join(output_dir, f"{base_filename}_info.txt"), 'w', encoding='utf-8') as f:
            f.write(f"Patent Number: {components['patent_number']}\n")
            f.write(f"Title: {components['title']}\n\n")
            f.write(f"Abstract:\n{components['abstract']}\n")
        
        # Save claims
        with open(os.path.join(output_dir, f"{base_filename}_claims.txt"), 'w', encoding='utf-8') as f:
            for i, claim in enumerate(components['claims'], 1):
                f.write(f"Claim {i}:\n{claim}\n\n")
        
        # Save figures
        with open(os.path.join(output_dir, f"{base_filename}_figures.txt"), 'w', encoding='utf-8') as f:
            for i, figure in enumerate(components['figures'], 1):
                f.write(f"Figure {i}:\n{figure}\n\n")
        
        # Save body text
        with open(os.path.join(output_dir, f"{base_filename}_body.txt"), 'w', encoding='utf-8') as f:
            f.write(components['body_text'])
        
        logger.info(f"Saved patent components to {output_dir}/{base_filename}_*.txt")


def main():
    """Example usage of the PatentComponentExtractor."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python patent_component_extractor.py <path_to_patent_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    output_dir = os.path.dirname(file_path)
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    
    extractor = PatentComponentExtractor()
    components = extractor.process_patent_file(file_path)
    
    # Print summary
    print(f"Patent Number: {components['patent_number']}")
    print(f"Title: {components['title']}")
    print(f"Abstract: {components['abstract'][:100]}...")
    print(f"Claims: {len(components['claims'])}")
    print(f"Figures: {len(components['figures'])}")
    print(f"Body Text: {len(components['body_text'])} characters")
    
    # Save to files
    extractor.save_components_to_files(components, output_dir, base_filename)

if __name__ == "__main__":
    main() 