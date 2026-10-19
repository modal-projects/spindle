from spindle.providers.modal.app_tags import tag_value


def test_tag_value_keeps_model_basename_in_modal_tag_charset():
    assert tag_value("Qwen/Qwen3.6-35B-A3B") == "qwen3.6-35b-a3b"
    assert tag_value("nvidia/Nemotron 3.5 Super (VL)") == "nemotron-3.5-super-vl"
