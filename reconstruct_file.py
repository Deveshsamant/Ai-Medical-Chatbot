import os
import glob

def reconstruct_file(base_filename="parquet cromadb/chroma.sqlite3"):
    """Reconstruct a file from its chunks"""
    
    # Find all chunk files
    chunk_pattern = f"{base_filename}.part*"
    chunks = sorted(glob.glob(chunk_pattern), key=lambda x: int(x.split('.part')[-1]))
    
    if not chunks:
        print(f"No chunks found matching pattern: {chunk_pattern}")
        return
    
    print(f"Found {len(chunks)} chunks")
    
    # Reconstruct the file
    with open(base_filename, 'wb') as outfile:
        for chunk_file in chunks:
            print(f"Merging: {chunk_file}")
            with open(chunk_file, 'rb') as infile:
                outfile.write(infile.read())
    
    file_size = os.path.getsize(base_filename)
    print(f"\nReconstruction complete!")
    print(f"Output file: {base_filename}")
    print(f"Size: {file_size / (1024**3):.2f} GB")

if __name__ == "__main__":
    reconstruct_file()
