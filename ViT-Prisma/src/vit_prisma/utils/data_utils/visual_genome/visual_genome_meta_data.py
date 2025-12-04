import json
import os
from dotenv import load_dotenv

load_dotenv()
base_dir = os.getenv('visual_genome_directory')
metadata_dir = os.path.join(base_dir,'image_data.json/image_data.json')

try:
    with open(metadata_dir, 'r') as json_file:
        content = json_file.read() # Read first
        metadata_list = json.loads(content)
        print(f"Type of data: {type(metadata_list)}")
        print(f"First image metadata: {metadata_list[0]}")
except FileNotFoundError as e:
    print(f"Error {e}")