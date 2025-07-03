import os
import requests
import zipfile
from tqdm import tqdm

# Configuration
DOWNLOAD_URLS = [
    "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip",
    "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships.json.zip",
    "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip"
]

RAW_DIR = "raw"
os.makedirs(RAW_DIR, exist_ok=True)

def download_file(url, dest_folder):
    """Download a file with progress bar"""
    local_filename = os.path.join(dest_folder, url.split('/')[-1])
    
    # Streaming download with progress
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        
        with open(local_filename, 'wb') as f:
            with tqdm.wrapattr(r.raw, "read", total=total_size, desc=f"Downloading {os.path.basename(local_filename)}") as raw:
                f.write(raw.read())
    
    return local_filename

def extract_zip(zip_path, dest_folder):
    """Extract zip file with progress"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get total files to extract for progress bar
        file_list = zip_ref.infolist()
        with tqdm(total=len(file_list), desc=f"Extracting {os.path.basename(zip_path)}") as pbar:
            for file in file_list:
                zip_ref.extract(file, dest_folder)
                pbar.update(1)

def cleanup_zip(zip_path):
    """Remove downloaded zip file"""
    os.remove(zip_path)
    print(f"Cleaned up: {os.path.basename(zip_path)}")

def main():
    print(f"Downloading Visual Genome dataset to {RAW_DIR}")
    
    # Download and process each file
    for url in DOWNLOAD_URLS:
        try:
            # Download
            zip_path = download_file(url, RAW_DIR)
            
            # Extract
            extract_zip(zip_path, RAW_DIR)
            
            # Cleanup
            cleanup_zip(zip_path)
            
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            continue
    
    print("Download and extraction complete!")
    print(f"Files available in: {os.path.abspath(RAW_DIR)}")

if __name__ == "__main__":
    main()