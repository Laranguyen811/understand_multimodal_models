
import os

from vit_prisma.models.model_config_registry import MODEL_CATEGORIES
import pytest
from vit_prisma.models.build_zero_shot_classifier import zero_shot_classifier, build_zero_shot_classifier, clean_model_name
from transformers import AutoTokenizer
from vit_prisma.utils.templates.open_ai.openai_templates import OPENAI_IMAGENET_TEMPLATES
from vit_prisma.utils.templates.anthropic.anthropic_templates import ANTHROPIC_CLAUDE_IMAGENET_TEMPLATES
from vit_prisma.utils.get_templates import get_all_templates

from dotenv import load_dotenv
load_dotenv()
templates_dir = os.getenv("templates_directory")
templates_list = get_all_templates(templates_dir)

class TestZeroShotClassifier:
    """
    Test suite for zero_shot_classifier function.
    """
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
            assert classifier is not None, "Classifier is None. Please sure the correct models."

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
class TestBuildZeroShotClassifier:

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


    def test_build_zero_shot_classifier_invalid_device():
        invalid_device = "invalid_device"
        with pytest.raises(ValueError):
            build_zero_shot_classifier("blip", tokenizer=AutoTokenizer.from_pretrained("blip"), classnames=["cat", "dog"], templates=OPENAI_IMAGENET_TEMPLATES, device=invalid_device, amp=True, use_format=False)
        print(ValueError, f"Expected ValueError for invalid device: {invalid_device}")

    def test_build_zero_shot_classifier_invalid_device_all_models_all_templates():
        invalid_device = "invalid_device"
        for model_name in MODEL_CATEGORIES:
            for template in templates_list:
                with pytest.raises(ValueError):
                    build_zero_shot_classifier(model_name, tokenizer=AutoTokenizer.from_pretrained(model_name), classnames=["cat", "dog"], templates=template, device=invalid_device, amp=True, use_format=False)
                print(ValueError, f"Expected ValueError for invalid device: {invalid_device} with model: {model_name} and template: {template}")


    def test_build_zero_shot_classifier_invalid_amp():
        invalid_amp = "invalid_amp"
        with pytest.raises(ValueError):
            build_zero_shot_classifier("blip", tokenizer=AutoTokenizer.from_pretrained("blip"), classnames=["cat", "dog"], templates=OPENAI_IMAGENET_TEMPLATES, device="cpu", amp=invalid_amp, use_format=False)
        print(ValueError, f"Expected ValueError for invalid amp setting: {invalid_amp}")

    #TODO: test for all models and templates with invalid amp settings to ensure consistent error handling across different configurations
    def test_build_zero_shot_classifier_invalid_amp_all_models_all_templates():
        invalid_amp = "invalid_amp"
        for model_name in MODEL_CATEGORIES:
            for template in templates_list:
                with pytest.raises(ValueError):
                    build_zero_shot_classifier(model_name, tokenizer=AutoTokenizer.from_pretrained(model_name), classnames=["person","toothbrsh"], templates=template, device="cpu", amp=invalid_amp, use_format=False)
                print(ValueError, f"Expected ValueError for invalid amp setting: {invalid_amp} with model: {model_name} and template: {template}")


    def test_build_zero_shot_classifier_invalid_device_invalid_classnames():
        invalid_device = "invalid_device"
        with pytest.raises(ValueError):
            build_zero_shot_classifier("vit_base_patch16_224", tokenizer=AutoTokenizer.from_pretrained("vit_base_patch16_224"), classnames=["picture","book"], templates=OPENAI_IMAGENET_TEMPLATES, device=invalid_device, amp=True, use_format=False)
        print(ValueError, f"Expected ValueError for invalid device: {invalid_device} and invalid class names: ['picture', 'book']")

    #TODO: test for all models and templates with invalid device and invalid class names to ensure consistent error handling across different configurations
    def test_build_zero_shot_classifier_invalid_device_invalid_classnames_all_models_all_templates():
        invalid_device = "invalid_device"
        for model_name in MODEL_CATEGORIES:
            for template in templates_list:
                with pytest.raises(ValueError):
                    build_zero_shot_classifier(model_name, tokenizer=AutoTokenizer.from_pretrained(model_name), classnames=["snowboard","srfboard"], templates=template, device=invalid_device, amp=True, use_format=False)
                print(ValueError, f"Expected ValueError for invalid device: {invalid_device} and invalid class names: ['snowboard', 'srfboard'] with model: {model_name} and template: {template}")

    def test_build_zero_shot_classifier_invalid_use_format():
        invalid_use_format = "invalid_use_format"
        with pytest.raises(ValueError):
            build_zero_shot_classifier("google/vivit-b-16x2-kinetics400", tokenizer=AutoTokenizer.from_pretrained("vit_base_patch16_224"), classnames=["motorcycle","traffic light"], templates=OPENAI_IMAGENET_TEMPLATES, device="cpu", amp=True, use_format=invalid_use_format)
        print(ValueError, f"Expected ValueError for invalid use_format setting: {invalid_use_format}")

    #TODO: test for all models and templates with invalid use_format settings to ensure consistent error handling across different configurations
    def test_build_zero_shot_classifier_invalid_use_format_all_models_all_templates():
        invalid_use_format = "invalid_use_format"
        for model_name in MODEL_CATEGORIES:
            for template in templates_list:
                with pytest.raises(ValueError):
                    build_zero_shot_classifier(model_name, tokenizer=AutoTokenizer.from_pretrained(model_name), classnames=["tennis racket","bottle"], templates=template, device="cpu", amp=True, use_format=invalid_use_format)
                print(ValueError, f"Expected ValueError for invalid use_format setting: {invalid_use_format} with model: {model_name} and template: {template}")


# This test suite below is AI-generated
class TestCleanModelName:
    """Test suite for clean_model_name function."""

    # ============ Success Cases: Single Character Replacements ============

    def test_single_hyphen(self):
        """Test replacement of single hyphen."""
        assert clean_model_name("model-name") == "model_name"

    def test_single_em_dash(self):
        """Test replacement of single em dash."""
        assert clean_model_name("model—name") == "model_name"

    def test_single_en_dash(self):
        """Test replacement of single en dash."""
        assert clean_model_name("model–name") == "model_name"

    def test_single_forward_slash(self):
        """Test replacement of single forward slash."""
        assert clean_model_name("model/name") == "model_name"

    def test_single_colon(self):
        """Test replacement of single colon."""
        assert clean_model_name("model:name") == "model_name"

    # ============ Success Cases: Multiple Different Characters ============

    def test_multiple_different_special_characters(self):
        """Test replacement of multiple different special characters."""
        result = clean_model_name("model-name—test–path/to:file")
        assert result == "model_name_test_path_to_file"

    def test_real_world_model_name(self):
        """Test with a real-world CLIP model name."""
        result = clean_model_name("hf-hub:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K")
        assert result == "hf_hub_laion_CLIP_ViT_B_32_DataComp.XL_s13B_b90K"

    def test_mixed_dash_types(self):
        """Test with all three types of dashes."""
        result = clean_model_name("model-test—another–value")
        assert result == "model_test_another_value"

    # ============ Success Cases: No Replacements Needed ============

    def test_no_special_characters(self):
        """Test string with no special characters that need replacement."""
        assert clean_model_name("modelname") == "modelname"

    def test_already_clean_name(self):
        """Test string that is already clean with underscores."""
        assert clean_model_name("model_name") == "model_name"

    def test_with_alphanumeric_only(self):
        """Test alphanumeric string without special characters."""
        assert clean_model_name("ModelName123") == "ModelName123"

    def test_with_dots_preserved(self):
        """Test that dots are preserved (not in replacement list)."""
        assert clean_model_name("model.name.v1") == "model.name.v1"

    def test_with_underscores_preserved(self):
        """Test that existing underscores are preserved."""
        assert clean_model_name("model_name_test") == "model_name_test"

    # ============ Edge Cases: Empty and Extreme Cases ============

    def test_empty_string(self):
        """Test with empty string."""
        assert clean_model_name("") == ""

    def test_only_special_characters_hyphens(self):
        """Test string with only hyphens."""
        assert clean_model_name("---") == "___"

    def test_only_special_characters_mixed(self):
        """Test string with only special characters."""
        assert clean_model_name("---///:::") == "_________"

    def test_only_em_dashes(self):
        """Test string with only em dashes."""
        assert clean_model_name("————") == "____"

    def test_only_en_dashes(self):
        """Test string with only en dashes."""
        assert clean_model_name("––––") == "____"

    def test_only_colons(self):
        """Test string with only colons."""
        assert clean_model_name("::::") == "____"

    def test_only_slashes(self):
        """Test string with only slashes."""
        assert clean_model_name("////") == "____"

    # ============ Edge Cases: Leading and Trailing Characters ============

    def test_leading_special_character(self):
        """Test with leading special character."""
        assert clean_model_name("-modelname") == "_modelname"

    def test_trailing_special_character(self):
        """Test with trailing special character."""
        assert clean_model_name("modelname-") == "modelname_"

    def test_leading_and_trailing_special_characters(self):
        """Test with leading and trailing special characters."""
        assert clean_model_name("-modelname-") == "_modelname_"

    def test_leading_and_trailing_mixed_special(self):
        """Test with different special chars at start and end."""
        assert clean_model_name(":model-name/") == "_model_name_"

    # ============ Edge Cases: Consecutive Special Characters ============

    def test_consecutive_hyphens(self):
        """Test with consecutive hyphens."""
        assert clean_model_name("model---name") == "model___name"

    def test_consecutive_slashes(self):
        """Test with consecutive slashes."""
        assert clean_model_name("model///name") == "model___name"

    def test_consecutive_colons(self):
        """Test with consecutive colons."""
        assert clean_model_name("model:::name") == "model___name"

    def test_consecutive_mixed_special_chars(self):
        """Test with consecutive mixed special characters."""
        assert clean_model_name("model-/:name") == "model___name"

    # ============ Edge Cases: Case Sensitivity ============

    def test_uppercase_preserved(self):
        """Test that uppercase is preserved."""
        assert clean_model_name("MODEL-NAME") == "MODEL_NAME"

    def test_mixed_case_preserved(self):
        """Test that mixed case is preserved."""
        assert clean_model_name("Model-Name") == "Model_Name"

    def test_lowercase_preserved(self):
        """Test that lowercase is preserved."""
        assert clean_model_name("model-name") == "model_name"

    def test_camelcase_preserved(self):
        """Test that camelCase is preserved."""
        assert clean_model_name("modelName-test") == "modelName_test"

    # ============ Edge Cases: Numbers and Special Cases ============

    def test_with_numbers(self):
        """Test with numbers in the name."""
        assert clean_model_name("model-32-v2:test") == "model_32_v2_test"

    def test_version_number_with_colons(self):
        """Test version number format with colons."""
        assert clean_model_name("model:v1:0:0") == "model_v1_0_0"

    def test_path_like_structure(self):
        """Test path-like structure with slashes."""
        assert clean_model_name("org/model-name:v1") == "org_model_name_v1"

    def test_url_like_structure(self):
        """Test URL-like structure (without http://)."""
        assert clean_model_name("huggingface/CLIP-ViT-B-32") == "huggingface_CLIP_ViT_B_32"

    # ============ Edge Cases: Unicode and Special Dashes ============

    def test_em_dash_multiple(self):
        """Test with multiple em dashes."""
        assert clean_model_name("model—test—name") == "model_test_name"

    def test_en_dash_multiple(self):
        """Test with multiple en dashes."""
        assert clean_model_name("model–test–name") == "model_test_name"

    def test_all_dash_types_mixed(self):
        """Test with all dash types mixed together."""
        result = clean_model_name("model-test–name—value")
        assert result == "model_test_name_value"

    # ============ Type and Return Value Validation ============

    def test_return_type_is_string(self):
        """Test that return type is always string."""
        result = clean_model_name("test-model")
        assert isinstance(result, str)

    def test_return_type_empty_string(self):
        """Test that return type is string even for empty input."""
        result = clean_model_name("")
        assert isinstance(result, str)

    def test_return_type_only_special_chars(self):
        """Test that return type is string for special-char-only input."""
        result = clean_model_name("---///")
        assert isinstance(result, str)

    # ============ Integration and Real-World Cases ============

    def test_real_clip_model_from_open_clip(self):
        """Test with real CLIP model name from open_clip."""
        result = clean_model_name("hf-hub:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K")
        # Should replace hyphens, colons, and slashes
        expected = "hf_hub_laion_CLIP_ViT_B_32_DataComp.XL_s13B_b90K"
        assert result == expected

    def test_long_string_with_many_replacements(self):
        """Test with long string containing many special characters."""
        long_name = "model-v1:dataset/type-variant–test—name"
        result = clean_model_name(long_name)
        expected = "model_v1_dataset_type_variant_test_name"
        assert result == expected

    def test_special_characters_at_every_position(self):
        """Test with special characters distributed throughout."""
        result = clean_model_name("-model:name/test—another–char:")
        assert result == "_model_name_test_another_char_"

    # ============ Boundary Conditions ============

    def test_single_character_hyphen(self):
        """Test with single hyphen as input."""
        assert clean_model_name("-") == "_"

    def test_single_character_colon(self):
        """Test with single colon as input."""
        assert clean_model_name(":") == "_"

    def test_single_character_slash(self):
        """Test with single slash as input."""
        assert clean_model_name("/") == "_"

    def test_single_character_normal(self):
        """Test with single normal character."""
        assert clean_model_name("a") == "a"

    def test_two_character_with_separator(self):
        """Test with two characters separated by special char."""
        assert clean_model_name("a-b") == "a_b"

    def test_whitespace_not_replaced(self):
        """Test that whitespace is not replaced (not in replacement list)."""
        assert clean_model_name("model name") == "model name"

    def test_special_symbols_not_replaced(self):
        """Test that symbols not in replacement list are preserved."""
        assert clean_model_name("model@name#test$value") == "model@name#test$value"
