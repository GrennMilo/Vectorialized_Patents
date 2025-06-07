#!/usr/bin/env python
import os
import sys
import subprocess
import webbrowser
import time
import socket
from threading import Thread

def check_port_in_use(port):
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_available_port(start_port=5000, max_attempts=10):
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if not check_port_in_use(port):
            return port
    return None

def print_header():
    """Print a nice header in the terminal."""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("\033[1;35m")  # Purple
    print("=" * 60)
    print("🔍 \033[1;37mPATENT DATABASE SEARCH\033[1;35m 🔍")
    print("=" * 60)
    print("\033[0m")  # Reset color

def print_section(title):
    """Print a section header."""
    print(f"\n\033[1;36m{title}\033[0m")
    print("\033[1;36m" + "-" * len(title) + "\033[0m")

def check_requirements():
    """Check if required packages are installed."""
    print_section("📋 Checking Requirements")
    
    required_packages = [
        "flask", 
        "sentence-transformers", 
        "faiss-cpu", 
        "pandas", 
        "numpy",
        "tqdm"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print("\n\033[1;33m[!] Some required packages are missing.\033[0m")
        install = input("    Would you like to install them now? (y/n): ")
        
        if install.lower() == 'y':
            cmd = [sys.executable, "-m", "pip", "install"] + missing_packages
            print(f"\n\033[1;34m[*] Running: {' '.join(cmd)}\033[0m")
            subprocess.call(cmd)
            print("\n\033[1;32m[+] Installation completed!\033[0m")
        else:
            print("\n\033[1;31m[!] Please install the missing packages manually:\033[0m")
            print(f"    pip install {' '.join(missing_packages)}")
            return False
    
    return True

def check_database():
    """Check if the vector database files exist."""
    print_section("🗄️ Checking Vector Database")
    
    required_files = [
        os.path.join("Results", "embeddings.npy"),
        os.path.join("Results", "chunk_mapping.csv"),
        os.path.join("Results", "patent_index.faiss")
    ]
    
    all_exist = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            all_exist = False
            print(f"❌ {file_path} is missing")
    
    if not all_exist:
        print("\n\033[1;33m[!] Vector database is incomplete or missing.\033[0m")
        print("    Please run Patents_Vectorizer.py first to create the database:")
        print("    python Patents_Vectorizer.py")
        
        create_db = input("\n    Would you like to run the vectorizer now? (y/n): ")
        
        if create_db.lower() == 'y':
            print(f"\n\033[1;34m[*] Running Patents_Vectorizer.py...\033[0m")
            try:
                subprocess.call([sys.executable, "Patents_Vectorizer.py"])
                print("\n\033[1;32m[+] Database creation completed!\033[0m")
                
                # Recheck after creation
                all_exist = all(os.path.exists(f) for f in required_files)
            except Exception as e:
                print(f"\n\033[1;31m[!] Error creating database: {str(e)}\033[0m")
                return False
        else:
            return False
    
    return all_exist

def open_browser(port):
    """Open the browser after a short delay."""
    time.sleep(2)
    webbrowser.open(f"http://localhost:{port}")
    print(f"\n\033[1;32m[+] Browser opened at http://localhost:{port}\033[0m")

def run_app(port=5000):
    """Run the Flask application."""
    print_section("🚀 Starting Web Application")
    
    # Find an available port if the default is in use
    if check_port_in_use(port):
        print(f"\033[1;33m[!] Port {port} is already in use.\033[0m")
        port = find_available_port(port)
        
        if port is None:
            print("\033[1;31m[!] Could not find an available port.\033[0m")
            return False
        
        print(f"\033[1;32m[+] Using port {port} instead.\033[0m")
    
    # Set environment variables for Flask
    os.environ["FLASK_APP"] = "app.py"
    if os.name == 'nt':  # Windows
        os.environ["FLASK_RUN_HOST"] = "localhost"
    else:
        os.environ["FLASK_RUN_HOST"] = "0.0.0.0"
    os.environ["FLASK_RUN_PORT"] = str(port)
    
    # Open browser in a separate thread
    browser_thread = Thread(target=open_browser, args=(port,))
    browser_thread.daemon = True
    browser_thread.start()
    
    print(f"\n\033[1;32m[+] Starting server at http://localhost:{port}\033[0m")
    print("\033[1;33m[!] Press Ctrl+C to stop the server\033[0m\n")
    
    # Run Flask app
    try:
        subprocess.call([
            sys.executable, "app.py"
        ])
    except KeyboardInterrupt:
        print("\n\033[1;33m[!] Server stopped by user.\033[0m")
    except Exception as e:
        print(f"\n\033[1;31m[!] Error running server: {str(e)}\033[0m")
        return False
    
    return True

if __name__ == "__main__":
    print_header()
    
    if check_requirements() and check_database():
        run_app()
    else:
        print("\n\033[1;31m[!] Setup incomplete. Please fix the issues above and try again.\033[0m")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    print("\n\033[1;32m[+] Patent Database Search app has been shut down.\033[0m")
    print("\033[1;35m" + "=" * 60 + "\033[0m") 