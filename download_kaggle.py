import os
import subprocess
import sys

def download_arxiv_dataset():
    dataset_dir = "raw_data"
    file_path = os.path.join(dataset_dir, "arxiv-metadata-oai-snapshot.json")
    
    # 1. Check if it already exists
    if os.path.exists(file_path):
        print(f"✅ Dataset already exists at {file_path}. No need to download.")
        return

    print("Dataset not found. Preparing to download from Kaggle...")
    os.makedirs(dataset_dir, exist_ok=True)

    # 2. Check for Kaggle library
    try:
        import kaggle
    except ImportError:
        print("Installing kaggle library...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
    except OSError:
        # Kaggle throws OSError on import if credentials are missing
        pass

    # 3. Check for credentials
    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
    has_env_vars = "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ

    if not os.path.exists(kaggle_json_path) and not has_env_vars:
        print("\n❌ KAGGLE CREDENTIALS MISSING ❌")
        print("Kaggle requires an account to download this 3.3GB dataset.")
        print("\nHow to fix:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Click 'Create New Token' to download a kaggle.json file.")
        print("3. Set them as environment variables in your terminal before running this script:")
        print("   export KAGGLE_USERNAME='your_username'")
        print("   export KAGGLE_KEY='your_key'")
        print("   python download_kaggle.py")
        sys.exit(1)

    # 4. Download and unzip
    print("\n⏳ Downloading 3.3GB arXiv dataset from Kaggle... (This may take a while)")
    try:
        subprocess.check_call([
            "kaggle", "datasets", "download", 
            "-d", "Cornell-University/arxiv", 
            "-p", dataset_dir, 
            "--unzip"
        ])
        print(f"\n✅ Download complete! File stored at {file_path}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Download failed: {e}")

if __name__ == "__main__":
    download_arxiv_dataset()
