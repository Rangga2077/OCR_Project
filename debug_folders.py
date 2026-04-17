import os

base_path = r'Z:\HKLight'
folders = os.listdir(base_path)[:20]

for folder_id in folders:
    path = os.path.join(base_path, folder_id)
    if not os.path.isdir(path):
        continue
    
    files = os.listdir(path)
    has_flag = [f for f in files if "施工後" in f]
    
    print(f"\n {folder_id} ({len(files)} files)")
    print(f"   Has '施工後': {len(has_flag)} → {has_flag[:2]}")
    print(f"   All files: {files[:5]}")  # Show first 5 filenames
