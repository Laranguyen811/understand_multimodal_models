"""
Unit tests for build_zero_shot_classifier module.
Tests the clean_model_name function with success cases, failure modes, and edge cases.
"""

import pytest
import sys
from pathlib import Path

# Add src to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vit_prisma.models.build_zero_shot_classifier import clean_model_name


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
