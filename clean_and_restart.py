import os
import shutil
import sys

def clean_results_directory(results_dir='Results'):
    """Clean up previous OCR results."""
    print(f"Cleaning up directory: {results_dir}")
    
    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} does not exist. Creating it.")
        os.makedirs(results_dir)
        return
    
    # Delete all text and chunk files
    for filename in os.listdir(results_dir):
        filepath = os.path.join(results_dir, filename)
        
        # Skip directories other than Patent_Images
        if os.path.isdir(filepath) and filename != 'Patent_Images':
            continue
            
        # Delete text files and chunk files
        if filename.endswith('.txt') or filename.endswith('_chunks.txt') or filename in ['embeddings.npy', 'chunk_mapping.csv', 'patent_index.faiss']:
            try:
                os.remove(filepath)
                print(f"Deleted: {filename}")
            except Exception as e:
                print(f"Error deleting {filename}: {str(e)}")
    
    # Clean up Patent_Images directory
    images_dir = os.path.join(results_dir, 'Patent_Images')
    if os.path.exists(images_dir):
        choice = input(f"Do you want to delete all images in {images_dir}? (y/n): ")
        if choice.lower() == 'y':
            try:
                shutil.rmtree(images_dir)
                os.makedirs(images_dir)
                print(f"Cleaned and recreated: {images_dir}")
            except Exception as e:
                print(f"Error cleaning images directory: {str(e)}")
        else:
            print("Keeping image files.")
    
    print("Cleanup complete.")

def run_extraction():
    """Run the patent extraction script."""
    print("\nStarting patent extraction process...")
    try:
        import Extract_Patent_Text
        Extract_Patent_Text.main()
    except Exception as e:
        print(f"Error running extraction script: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("="*60)
    print("🧹 Patent OCR Cleanup and Restart Tool 🧹")
    print("="*60)
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Clean up previous OCR results")
        print("2. Run patent extraction")
        print("3. Both (clean up AND run extraction)")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == "1":
            clean_results_directory()
        elif choice == "2":
            run_extraction()
        elif choice == "3":
            clean_results_directory()
            run_extraction()
        elif choice == "4":
            print("Exiting...")
            sys.exit(0)
        else:
            print("Invalid choice. Please enter a number between 1 and 4.") 