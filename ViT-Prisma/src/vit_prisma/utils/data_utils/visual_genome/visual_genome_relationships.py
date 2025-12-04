
import json
import os
from dotenv import load_dotenv

rel_base_dir = os.getenv('visual_genome_directory')

def create_rel_data(rel_base_dir):
    relationships_dir = os.path.join(rel_base_dir,"relationships.json/relationships.json")
    with open(relationships_dir,'r') as json_file:
        content = json_file.read() # Read the JSON file first
        relationships_list = json.loads(content)
        print(f"Type of data: {type(relationships_list)}")
        print(f"First image's relationships:{relationships_list[0]}")
    return relationships_list


def get_relationships_by_predicate(image_id, relationships_data,predicate=None):
    cases = []
    for item in relationships_data:
        if item['image_id'] == image_id:
            for rel in item['relationships']:
                if predicate is None or rel['predicate'] == predicate:
                        subject_name = rel['subject'].get('name', rel['subject'].get('names',[None])[0])
                        object_name = rel['object'].get('name',rel['object'].get('names',[None])[0])
                        cases.append({
                            'image_id': image_id,
                            'subject': subject_name,
                            'object': object_name,
                            'predicate': rel['predicate']
                        })
            break # Exit after having found the matching image_id
    return cases
