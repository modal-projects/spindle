import modal
import pytest

from spindle.providers.modal.miles_image import image, mount_local_miles_tree


def test_local_miles_tree_must_be_a_miles_checkout(tmp_path) -> None:
    with pytest.raises(ValueError, match="is not a Miles checkout"):
        mount_local_miles_tree(image, str(tmp_path))


def test_local_miles_checkout_is_mounted_into_the_image(tmp_path) -> None:
    (tmp_path / "miles").mkdir()
    (tmp_path / "miles" / "__init__.py").write_text("")
    assert isinstance(mount_local_miles_tree(image, str(tmp_path)), modal.Image)
