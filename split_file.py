import os

def split_file(filename, chunk_size_mb=1024):
    """Split a large file into chunks"""
    chunk_size = chunk_size_mb * 1024 * 1024  # Convert MB to bytes
    
    if not os.path.exists(filename):
        print(f"Error: {filename} not found")
        return
    
    file_size = os.path.getsize(filename)
    print(f"File size: {file_size / (1024**3):.2f} GB")
    
    chunk_num = 0
    with open(filename, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            chunk_filename = f"{filename}.part{chunk_num}"
            with open(chunk_filename, 'wb') as chunk_file:
                chunk_file.write(chunk)
            
            chunk_size_mb_actual = len(chunk) / (1024**2)
            print(f"Created: {chunk_filename} ({chunk_size_mb_actual:.2f} MB)")
            chunk_num += 1
    
    print(f"\nTotal chunks created: {chunk_num}")
    print(f"\nTo reconstruct the file, run:")
    print(f"python reconstruct_file.py")

if __name__ == "__main__":
    split_file("parquet cromadb/chroma.sqlite3", chunk_size_mb=1024)
