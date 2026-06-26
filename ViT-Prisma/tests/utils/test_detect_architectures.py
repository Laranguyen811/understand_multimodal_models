from vit_prisma.utils.detect_architectures import get_architecture_category, get_architecture_info
import pytest

class TestGetArchitectureCategory:
    """
    Test suite for get_architecture_category function.
    """

    # Test for success cases

    def test_get_architecture_category_valid_model():
        """
        Test for a valid category with a valid model.
        """
        model = "LLaVa"
        with pytest.raises(ValueError):
            category = get_architecture_category(model)
        assert category == "llava", "Invalid Category. Please check the category."

    #def test_get_architecture_category_valid_models():
    #    """
    #    Test for valid categories with multiple valid models.
    #    """
    #    #TODO: continue the test for multiple valid models
    #    model_1 ="BLIP"
    #    model_2 = "CLIP"
    #    category_1 = get_architecture_category(model_1)
    #    category_2 = get_architecture_category(model_2)
    #
    #    assert category_1 == "blip", "Invalid Category. Please check the category."
    #    assert category_2 == "clip", "Invalid Category. Please check the category."

