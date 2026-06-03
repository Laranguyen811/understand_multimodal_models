
from vit_prisma.models.model_config_registry import MODEL_CATEGORIES
import pytest
from vit_prisma.models.build_zero_shot_classifier import zero_shot_classifier, build_zero_shot_classifier
from transformers import AutoTokenizer
from vit_prisma.utils.templates.open_ai.openai_templates import OPENAI_IMAGENET_TEMPLATES
from vit_prisma.utils.templates.anthropic.anthropic_templates import ANTHROPIC_CLAUDE_IMAGENET_TEMPLATES
templates_list = [OPENAI_IMAGENET_TEMPLATES, ANTHROPIC_CLAUDE_IMAGENET_TEMPLATES]

def test_zero_shot_classifier():
    classifier = zero_shot_classifier("openai/clip-vit-large-patch14", tokenizer=AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14"), classnames=["cat", "dog"], templates=OPENAI_IMAGENET_TEMPLATES, device="cpu", amp=True, use_format=False)
    print(classifier)
    assert classifier is not None

def test_zero_shot_classifier_invalid_model():
    invalid_model_name = "invalid/model-name"
    with pytest.raises(ValueError):
        zero_shot_classifier(invalid_model_name, tokenizer=AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14"), classnames=["cat", "dog"], templates=OPENAI_IMAGENET_TEMPLATES, device="cpu", amp=True, use_format=False)
    print(ValueError, f"Expected ValueError for invalid model name: {invalid_model_name}")

def test_zero_shot_classifier_no_model_specified():
    with pytest.raises(ValueError):
        zero_shot_classifier("", tokenizer=AutoTokenizer.from_pretrained("claude/claude-2"), classnames=["cat", "dog"], templates=ANTHROPIC_CLAUDE_IMAGENET_TEMPLATES, device="cpu", amp=True, use_format=False)
    print(ValueError, "Expected ValueError for no model specified")

def test_zero_shot_classifier_invalid_tokenizer():
    invalid_tokenizer_name = "invalid/tokenizer-name"
    with pytest.raises(ValueError):
        zero_shot_classifier("vit_base_patch16_224", tokenizer=AutoTokenizer.from_pretrained(invalid_tokenizer_name), classnames=["cat", "dog"], templates=OPENAI_IMAGENET_TEMPLATES, device="cpu", amp=True, use_format=False)
    print(ValueError, f"Expected ValueError for invalid tokenizer name: {invalid_tokenizer_name}")
    
def test_zero_shot_classifier_invalid_classnames():
    invalid_classnames = ["", ""]  # Empty class names
    with pytest.raises(ValueError):
        zero_shot_classifier("vit_base_patch16_224", tokenizer=AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14"), classnames=invalid_classnames, templates=OPENAI_IMAGENET_TEMPLATES, device="cpu", amp=True, use_format=False)
    print(ValueError, f"Expected ValueError for invalid class names: {invalid_classnames}")

def test_zero_shot_classifier_all_models():
    for model_name in MODEL_CATEGORIES:
        print(f"Testing zero-shot classifier for model: {model_name}")
        classifier = zero_shot_classifier(model_name, tokenizer=AutoTokenizer.from_pretrained(model_name), classnames=["cat", "dog"], templates=OPENAI_IMAGENET_TEMPLATES, device="cpu", amp=True, use_format=False)
        print(classifier)
        assert classifier is not None

def test_zero_shot_classifier_invalid_device():
    invalid_device = "invalid_device"
    with pytest.raises(ValueError):
        zero_shot_classifier("vit_base_patch16_224", tokenizer=AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14"), classnames=["cat", "dog"], templates=OPENAI_IMAGENET_TEMPLATES, device=invalid_device, amp=True, use_format=False)
    print(ValueError, f"Expected ValueError for invalid device: {invalid_device}")

def test_zero_shot_classifier_invalid_amp():
    invalid_amp = "invalid_amp"
    with pytest.raises(ValueError):
        zero_shot_classifier("claude/claude-2", tokenizer=AutoTokenizer.from_pretrained("claude/claude-2"), classnames=["horse","person"], templates=ANTHROPIC_CLAUDE_IMAGENET_TEMPLATES, device="cpu", amp=invalid_amp, use_format=False)
    print(ValueError, f"Expected ValueError for invalid amp setting: {invalid_amp}")
# TODO: Add more tests with different datasets, models and prompt templates to ensure robustness and versatility of the zero-shot classifier across various scenarios.
def test_build_zero_shot_classifier_invalid_model():
    invalid_model_name = "invalid/model-name"
    with pytest.raises(ValueError):
        build_zero_shot_classifier(invalid_model_name, tokenizer=AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14"), classnames=["cat", "dog"], templates=OPENAI_IMAGENET_TEMPLATES, device="cpu", amp=True, use_format=False)
    print(ValueError, f"Expected ValueError for invalid model name: {invalid_model_name}")

# TODO: test for different models with corresponding templates
def test_build_zero_shot_classifier_all_models():
    for model_name in MODEL_CATEGORIES:
         for template in templates_list:
            print(f"Testing zero-shot classifier for model: {model_name}")
            classifier = build_zero_shot_classifier(model_name, tokenizer=AutoTokenizer.from_pretrained(model_name), classnames=["cat", "dog"], templates=template, device="cpu", amp=True, use_format=False)
            print(classifier)
            assert classifier is not None



