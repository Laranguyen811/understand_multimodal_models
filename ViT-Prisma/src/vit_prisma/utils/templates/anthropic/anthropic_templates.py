#TODO: Build out templates for Anthropic's Claude models

from anthropic import Anthropic
ANTHROPIC_CLAUDE_IMAGENET_TEMPLATES = (
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a photo of many {c}.',
    lambda c: f'a sculpture of a {c}.',
    lambda c: f'a photo of the hard to see {c}.',
    lambda c: f'a low resolution photo of the {c}.',
    lambda c: f'a rendering of a {c}.',
    lambda c: f'graffiti of a {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a cropped photo of the {c}.',
    lambda c: f'a tattoo of a {c}.',
    lambda c: f'the embroidered {c}.',
    lambda c: f'a photo of a hard to see {c}.',
    lambda c: f'a bright photo of a {c}.',
    lambda c: f'a photo of a clean {c}.',
    lambda c: f'a photo of a dirty {c}.',
    lambda c: f'a dark photo of the {c}.',
    lambda c: f'a drawing of a {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'the plastic {c}.',
    lambda c: f'a photo of the cool {c}.',
    lambda c: f'a close-up photo of a {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a painting of the {c}.',
    lambda c: f'a painting of a {c}.',
    lambda c: f'a pixelated photo of the {c}.',
    lambda c: f'a sculpture of the {c}.',
    lambda c: f'a bright photo of the {c}.',
    lambda c: f'a cropped photo of a {c}.',
    lambda c: f'a plastic {c}.',
    lambda c: f'a photo of the dirty {c}.',
    lambda c: f'a jpeg corrupted photo of a {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a rendering of the {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'a photo of one {c}.',
    lambda c: f'a doodle of a {c}.',
    lambda c: f'a close-up photo of the {c}.',
    lambda c: f'a photo of a {c}.',
    lambda c: f'the origami {c}.',
    lambda c: f'the {c} in a video game.',
    lambda c: f'a sketch of a {c}.',
    lambda c: f'a doodle of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a low resolution photo of a {c}.',
    lambda c: f'the toy {c}.',
    lambda c: f'a rendition of the {c}.',
    lambda c: f'a photo of the clean {c}.',
    lambda c: f'a photo of a large {c}.',
    lambda c: f'a rendition of a {c}.',
    lambda c: f'a photo of a nice {c}.',
    lambda c: f'a photo of a weird {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a cartoon {c}.',
    lambda c: f'art of a {c}.',
    lambda c: f'a sketch of the {c}.',
    lambda c: f'a embroidered {c}.',
    lambda c: f'a pixelated photo of a {c}.',
    lambda c: f'itap of the {c}.',
    lambda c: f'a jpeg corrupted photo of the {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a plushie {c}.',
    lambda c: f'a photo of the nice {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the weird {c}.',
    lambda c: f'the cartoon {c}.',
    lambda c: f'art of the {c}.',
    lambda c: f'a drawing of the {c}.',
    lambda c: f'a photo of the large {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'the plushie {c}.',
    lambda c: f'a dark photo of a {c}.',
    lambda c: f'itap of a {c}.',
    lambda c: f'graffiti of the {c}.',
    lambda c: f'a toy {c}.',
    lambda c: f'itap of my {c}.',
    lambda c: f'a photo of a cool {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a tattoo of the {c}.',
)

# Initialise client
client = Anthropic()

# Template for a sematic description of the scene
CLAUDE_COCO_SEMANTIC_DESCRIPTION_TEMPLATE = """You are analyzing an image from the COCO dataset.
Image description: {image_description}
Based on the image description, provide a detailed semantic description of the scene, including:
- The main objects present and their attributes (e.g., color, size, texture)
- The spatial relationships between the objects
- The overall scene context (e.g., indoor/outdoor, lighting conditions)
Format your response as a JSON object with the structure:
{
  "objects": [
    {
      "category": "object_category",
      "attributes": {
        "color": "color_value",
        "size": "size_value",
        "texture": "texture_value"
      }
    },
    ...
  ],
  "relationships": [
    {
      "subject": "object_category_1",
      "predicate": "relationship",
      "object": "object_category_2"
    },
    ...
  ],
  "scene_context": {
    "setting": "indoor/outdoor",
    "lighting": "lighting_conditions"
  }
}
"""

# Template for object detection questions
CLAUDE_COCO_DETECTION_TEMPLATE = """
Given this COCO image data:
- Image ID: {image_id}
- Main objects: {objects}
- Scene context: {scene}

Question: {question}

Provide a detailed answer based on the image information provided.
"""

# Template for scene understanding
CLAUDE_COCO_SCENE_TEMPLATE = """
Analyze this COCO image:

Objects present: {object_list}
Spatial relationships: {relationships}
Activity/Action: {activity}

Describe the scene's context, what's happening, and why these objects are arranged this way.
"""

# Template for localising 2D and 3D objects in the image
CLAUDE_COCO_LOCALISATION_TEMPLATE = """
You are analyzing an image from the COCO dataset.
Image description: {image_description}
Object categories detected: {categories}
For each object category, provide the following:
- 2D bounding box coordinates (x_min, y_min, x_max, y_max)
- 3D bounding box parameters (if available): (x_center, y_center, z
_center, width, height, depth)
- Confidence score for the detection
Format your response as a JSON object with the structure:
{
  "objects": [
    {
      "category": "object_category",
      "bbox_2d": [x_min, y_min, x_max, y_max],
      "bbox_3d": [x_center, y_center, z_center, width, height, depth],
      "confidence": confidence_score
    },
    ...
  ]
}
"""

# Template for recognising what objects are present
CLAUDE_COCO_RECOGNITION_TEMPLATE = """You are analyzing an image from the COCO dataset.
Image description: {image_description}
Based on the image description, identify the main objects present in the image.
List the objects in order of prominence, starting with the most prominent one.
Format your response as a JSON array of object categories, like this:
["object_category_1", "object_category_2", ...]
"""
# Template for characterising objects' and scene's attributes
CLAUDE_COCO_CHARACTERISATION_TEMPLATE = """You are analyzing an image from the COCO dataset.
Image description: {image_description}
Identify the main objects present in the image and describe their attributes (e.g., color, size, texture).
Also, describe the overall scene attributes (e.g., indoor/outdoor, lighting conditions).
Format your response as a JSON object with the structure:
{
  "objects": [
    {
      "category": "object_category",
      "attributes": {
        "color": "color_value",
        "size": "size_value",
        "texture": "texture_value"
      }
    },
    ...
  ],
  "scene_attributes": {
    "setting": "indoor/outdoor",
    "lighting": "lighting_conditions"
  }
}
"""
