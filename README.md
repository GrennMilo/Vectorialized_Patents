# Vactorialized Patents

A comprehensive solution for processing, analyzing, and searching patent documents using OCR, vector embeddings, and semantic search.

## 📋 Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Patent Processing Pipeline](#patent-processing-pipeline)
- [Component Extraction Details](#component-extraction-details)
- [Web Application](#web-application)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Patent OCR** | Convert patent PDFs to text using advanced OCR techniques |
| **Component Extraction** | Extract title, abstract, claims, figures, and body text from patents |
| **Semantic Search** | Find relevant patents based on meaning, not just keywords |
| **Vector Database** | Store and search patents using FAISS vector indexing |
| **Web Interface** | User-friendly web application for searching and viewing patents |
| **Parallel Processing** | Process multiple patents concurrently for improved performance |
| **Figure Display** | View patent figures and diagrams in an image gallery |
| **Dark Theme UI** | Minimalist, modern interface designed for readability |
| **Lightbox Integration** | Zoom and browse patent images with lightbox viewer |

## 🏗️ System Architecture

The system follows a modular architecture with three main phases:

1. **Processing Phase**
   - PDF conversion → OCR → Text extraction
   - Component extraction → Structured data

2. **Indexing Phase**
   - Text chunking → Vector embeddings → FAISS indexing

3. **Retrieval Phase**
   - Web UI → Query vectorization → Semantic search → Results display

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  PDF Files  │ → │  OCR & Text  │ → │  Component  │ → │  Structured  │
│             │    │  Extraction  │    │  Extraction │    │     Data    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                ↓
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Search    │ ← │ FAISS Index  │ ← │   Vector    │ ← │    Text     │
│     UI      │    │             │    │  Embeddings │    │   Chunking  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## 📦 Installation

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.8+ | Core programming language |
| Poppler | Latest | PDF to image conversion |
| Tesseract OCR | 4.0+ | Text extraction from images |
| CUDA (optional) | 11.0+ | GPU acceleration for vector operations |

### Setup Steps

1. Clone this repository
   ```bash
   git clone https://github.com/GrennMilo/Vactorialized_Patents.git
   cd Vactorialized_Patents
   ```

2. Install Python dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Install system dependencies:
   - **Poppler** (for PDF conversion): 
     - Windows: [Download from poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases/)
     - Linux: `sudo apt install poppler-utils`
     - macOS: `brew install poppler`
   
   - **Tesseract OCR**:
     - Windows: [Download from UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
     - Linux: `sudo apt install tesseract-ocr`
     - macOS: `brew install tesseract`

4. Create necessary directories
   ```bash
   mkdir -p Patents Results/Components Results/Patent_Images
   ```

## 🚀 Usage

### Patent Processing

#### Process a Single Patent

Process a single patent PDF file and extract its components:

```bash
# Linux/macOS
python process_patent.py Patents/US1234567.pdf

# Windows
process_patent.bat Patents\US1234567.pdf
```

Available options:
```
--output, -o DIR    Directory to save results (default: Results)
--skip-ocr          Skip OCR if text file already exists
```

#### Process Multiple Patents

Process all patents in the Patents directory:

```bash
# Linux/macOS
python process_patents.py

# Windows
process_all_patents.bat
```

Available options:
```
--input, -i DIR     Directory containing patent PDFs (default: Patents)
--output, -o DIR    Directory to save results (default: Results)
--workers, -w NUM   Maximum number of worker processes
--ocr-only          Only perform OCR without component extraction
--limit, -l NUM     Maximum number of patents to process
```

### Web Application

Start the web application:

```bash
python app.py
```

Then open your browser and navigate to `http://localhost:5000`

## 📁 Project Structure

```
vactorialized_patents/
│
├── app.py                           # Flask web application
├── Patents_Vectorizer.py            # Patent OCR and vectorization tool
├── patent_component_extractor.py    # Extract patent components from text
├── process_patent.py                # Process a single patent PDF
├── process_patents.py               # Process multiple patent PDFs
├── process_patent.bat               # Windows batch file for processing a patent
├── process_all_patents.bat          # Windows batch file for processing all patents
├── requirements.txt                 # Python dependencies
│
├── Patents/                         # Directory containing patent PDF files
│
├── Results/                         # Processed data
│   ├── embeddings.npy               # Patent chunk embeddings
│   ├── chunk_mapping.csv            # Mapping between embeddings and patents
│   ├── patent_index.faiss           # FAISS index for similarity search
│   ├── *.txt                        # OCR-extracted text files
│   ├── *_chunks.txt                 # Chunked text files for vectorization
│   │
│   ├── Patent_Images/               # Directory containing extracted patent images
│   │   └── [PatentID]/              # Images for each patent
│   │
│   └── Components/                  # Extracted patent components
│       └── [PatentID]/              # Components for each patent
│           ├── [PatentID]_info.txt  # Patent number, title, abstract
│           ├── [PatentID]_claims.txt# Patent claims
│           ├── [PatentID]_figures.txt# Figure descriptions
│           ├── [PatentID]_body.txt  # Main body text
│           └── [PatentID]_summary.txt# Summary of extracted components
│
├── static/                          # Static assets
│   ├── css/                         # CSS stylesheets
│   │   └── style.css                # Main stylesheet
│   └── js/                          # JavaScript files
│       └── main.js                  # Main JavaScript functionality
│
└── templates/                       # HTML templates
    ├── base.html                    # Base template with common layout
    ├── index.html                   # Search page template
    └── patent.html                  # Patent details page template
```

## 🧩 Core Components

### Patents_Vectorizer.py

The main OCR and vectorization engine with the following capabilities:

| Module | Function | Description |
|--------|----------|-------------|
| `PatentOCRVectorizer` | OCR Processing | Converts PDFs to images and extracts text |
| | Text Chunking | Splits text into semantic chunks for vectorization |
| | Vector Embedding | Converts text to embeddings with sentence-transformers |
| | FAISS Indexing | Creates efficient similarity search index |
| | Patent Search | Retrieves semantically similar patents |

### patent_component_extractor.py

Extracts structured components from patent OCR text:

| Component | Description | Extraction Method |
|-----------|-------------|------------------|
| Patent Number | The official USPTO patent number | Regex pattern matching |
| Title | Official patent title | Multi-method approach with marker detection |
| Abstract | Brief summary of the invention | Section boundary detection |
| Claims | Legal claims defining the invention | Numbered claim detection |
| Figures | Descriptions and captions of diagrams | Figure reference detection |
| Body Text | Main technical description | Section boundary detection |

### process_patent.py / process_patents.py

Patent processing scripts for single or batch processing:

| Script | Function | Parallelization |
|--------|----------|----------------|
| `process_patent.py` | Single patent processing | N/A |
| `process_patents.py` | Batch processing | Process pool executor |

## 📊 Patent Processing Pipeline

The patent processing pipeline consists of the following steps:

1. **PDF Conversion**
   - Convert PDF pages to images using `pdf2image`
   - DPI configurable (default: 300)

2. **OCR Processing**
   - Apply Tesseract OCR to each image
   - Merge text from all pages
   - Apply text cleaning to improve quality

3. **Component Extraction**
   - Extract patent number, title, abstract
   - Extract claims as individual items
   - Extract figure descriptions
   - Extract main body text

4. **Vector Embedding**
   - Chunk text into semantic segments
   - Generate embeddings using sentence-transformers
   - Create FAISS index for similarity search

## 📑 Component Extraction Details

The component extraction process uses a combination of techniques:

| Component | Extraction Technique | Challenges | Solutions |
|-----------|----------------------|------------|-----------|
| Patent Number | Regex pattern | OCR errors, formatting variations | Multiple regex patterns |
| Title | Marker detection | Multi-line titles, formatting | Combined strategies |
| Abstract | Section boundaries | Varying positions, formatting | Flexible boundary detection |
| Claims | Numbered claim detection | Claim formatting variations | Multi-pass extraction |
| Figures | Figure reference detection | Inconsistent references | Pattern matching with cleanup |
| Body Text | Section boundaries | Complex document structure | Fallback strategies |

### Regex Patterns Used

```python
'patent_number': r'US\s*([0-9,]{6,})\s*[A-Z][0-9]',
'title': r'(?:\(54\)|^54\))\s+(.*?)(?=\(|$)',
'abstract': r'(?:ABSTRACT|^\(57\)\s+ABSTRACT)[\s\n]+(.+?)(?=\n\s*(?:[0-9]+\s+Claims|BRIEF DESCRIPTION|DETAILED DESCRIPTION|BACKGROUND|\[FIGURE\]|FIG\.)|\n\s*$)',
'claims_start': r'(?:^|\n)(?:What is claimed is:|I claim:|We claim:|Claims?:)(?:\s*\n|\s+)',
'figures': r'(?:FIG\.?\s*[0-9]+[A-Za-z]*|^\[FIGURE\]).*?(?=FIG\.?\s*[0-9]+[A-Za-z]*|\[FIGURE\]|\n\n|$)',
```

## 🌐 Web Application

The web application provides a user-friendly interface for searching and viewing patents:

| Feature | Description |
|---------|-------------|
| Semantic Search | Find patents based on meaning using vector similarity |
| Results Ranking | Display patents ranked by semantic relevance |
| Patent Viewer | View full patent text and components |
| Figure Gallery | Browse patent figures and diagrams |
| Dark Theme | Modern UI with eye-friendly dark theme |

### Search Algorithm

The search process follows these steps:

1. Convert search query to vector embedding
2. Calculate similarity against all patent chunk embeddings
3. Retrieve top k most similar chunks
4. Group results by patent and display in ranked order

## 🔧 Customization

### Search Parameters

Adjust the number of search results by changing the `k` parameter in the `search_patents` function:

```python
# In app.py
results = vectorizer.search_patents(query, k=10)  # Change 10 to desired number
```

### UI Theme

Modify the CSS variables in `static/css/style.css` to customize the appearance:

```css
:root {
  --bg-color: #121212;
  --card-bg: #1e1e1e;
  --text-color: #e0e0e0;
  --accent-color: #007bff;
  /* Add or modify as needed */
}
```

### Component Extraction

Improve extraction accuracy by modifying regex patterns in `patent_component_extractor.py`:

```python
self.patterns = {
    'patent_number': r'US\s*([0-9,]{6,})\s*[A-Z][0-9]',
    # Modify patterns as needed
}
```

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [Sentence-Transformers](https://www.sbert.net/) for text embeddings
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for optical character recognition
- [pdf2image](https://github.com/Belval/pdf2image) for PDF to image conversion 