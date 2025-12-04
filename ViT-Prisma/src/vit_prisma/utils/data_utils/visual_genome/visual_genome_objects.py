import json
import os
from dotenv import load_dotenv


load_dotenv()
base_dir = os.getenv("visual_genome_directory")

def create_dict(base_dir)-> dict:
    obj_dir = os.path.join(base_dir,"objects.json/objects.json")
    with open(obj_dir) as json_file:
        objects = json.load(json_file)
        print(f"Type of data: {type(objects)}")
        print (f"First image: {objects[0]}")
        print(f"First image index: {objects[0]['image_id']}")
        print(f"Number of first image objects: {(len(objects[0]['objects']))}")
        print(f"Number of images: {len(objects)}")
        
        object_names = [name for image in objects for obj in image['objects'] for name in obj['names']] # Obtain all the labels in all images
        # Create a label dictionary
        objects_dict = {}
        for image in objects:
            image_id = image['image_id']
            names = [name for objs in image['objects'] for name in objs['names']]
            objects_dict[image_id] = names
        
        first_key, first_values = next(iter(objects_dict.items()))
        print(f"First key: {first_key}, First values: {first_values}")
        return objects, objects_dict




