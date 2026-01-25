import os
import pandas as pd
import yt_dlp
from tqdm import tqdm
from typing import Dict, Any

CSV_FILE = "Final_dataset_pairs.csv"
OUTPUT_DIR = "data/raw_smp"

COL_ORIGINAL = "ori_link"
COL_SUSPICIOUS = "comp_link"

def download_audio(url, output_path):
    """Downloads audio from YouTube and saves it as .wav"""
    
    ydl_opts: Dict[str, Any] = {
        'format': 'bestaudio/best',
        'outtmpl': output_path.replace('.wav', ''), 
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: # type: ignore
            ydl.download([url])
        return True
    except Exception as e:
        print(f"\nError downloading {url}: {e}")
        return False

def main():
    if not os.path.exists(CSV_FILE):
        print(f"Couldn't find the file {CSV_FILE}.")
        return

    # Read CSV
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"Found {len(df)} pairs inside the CSV.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    success_count = 0

    # Iteration over rows
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading Dataset"):
        # Create pair folder
        pair_folder = os.path.join(OUTPUT_DIR, f"pair_{idx}")
        os.makedirs(pair_folder, exist_ok=True)

        # Download Original
        orig_url = row[COL_ORIGINAL]
        orig_path = os.path.join(pair_folder, "original.wav")
        
        if not os.path.exists(orig_path):
            ok1 = download_audio(orig_url, orig_path)
        else:
            ok1 = True

        # Download Suspicious (Plagiarized)
        susp_url = row[COL_SUSPICIOUS]
        susp_path = os.path.join(pair_folder, "suspicious.wav")
        
        if not os.path.exists(susp_path):
            ok2 = download_audio(susp_url, susp_path)
        else:
            ok2 = True

        if ok1 and ok2:
            success_count += 1
    
    print(f"\nFinished! Downloaded {success_count}/{len(df)} pairs.")
    print(f"Files can be found at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
