
from vit_prisma.models.model_config_registry import MODEL_CATEGORIES
import pytest
from vit_prisma.models.build_zero_shot_classifier import build_zero_shot_classifier
from transformers import AutoTokenizer
# Test for building zero-shot classifiers
def test_zero_shot_classifier():
    classifier = build_zero_shot_classifier("openai/clip-vit-large-patch14", tokenizer=AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14"), classnames=["cat", "dog"], templates=OPENAI_IMAGENET_TEMPLATES, device="cpu", amp=True, use_format=False)
    print(classifier)
    assert classifier is not None

def test_zero_shot_classifier_all_models():
    for model_name in MODEL_CATEGORIES:
        print(f"Testing zero-shot classifier for model: {model_name}")
        classifier = build_zero_shot_classifier(model_name, tokenizer=AutoTokenizer.from_pretrained(model_name), classnames=["cat", "dog"], templates=OPENAI_IMAGENET_TEMPLATES, device="cpu", amp=True, use_format=False)
        print(classifier)
        assert classifier is not None
#TODO: test building zero-shot classifiers for all models, and also test that the classifiers have the expected shape and values
def test_build_zero_shot_classifier_all_models():
    for model_name in MODEL_CATEGORIES:
        print(f"Testing zero-shot classifier for model: {model_name}")
        classifier = build_zero_shot_classifier(model_name, tokenizer=AutoTokenizer.from_pretrained(model_name), classnames=["cat", "dog"], templates=OPENAI_IMAGENET_TEMPLATES, device="cpu", amp=True, use_format=False)
        print(classifier)
        assert classifier is not None
