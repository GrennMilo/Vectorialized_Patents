# Sample Data

This directory contains empty folders that maintain the expected directory structure for the project. 

In a real deployment, you would:

1. Place patent PDF files in the `Patents` directory
2. Process them using the provided scripts
3. The results would be stored in the `Results` directory

## Directory Structure

- `Patents/` - Place your patent PDF files here
- `Results/` - Contains processed output
  - `Components/` - Extracted patent components (title, abstract, claims, etc.)
  - `Patent_Images/` - Extracted images from patents

## Processing Patents

To process patents in this directory, use:

```bash
python process_patents.py --input Sample_Data/Patents --output Sample_Data/Results
``` 