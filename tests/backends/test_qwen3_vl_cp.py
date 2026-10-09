from types import SimpleNamespace

from spindle.backends.miles_runtime.qwen3_vl_cp import install_qwen3_vl_cp_position_ids


def _install(pre_sharded: bool):
    calls = []

    class Qwen3VLModel:
        def forward(self, *args, **kwargs):
            calls.append(kwargs)
            return "out"

    def get_rope_index(*args, **kwargs):
        raise AssertionError("positions come from Miles")

    def build_positions(model, parsed, kwargs, rope_index):
        assert rope_index is get_rope_index
        return ("positions", parsed)

    bridge_model = SimpleNamespace(
        Qwen3VLModel=Qwen3VLModel, get_rope_index=get_rope_index
    )
    miles_qwen3_vl = SimpleNamespace(
        _parse_packed_thd=lambda args, kwargs: kwargs["input_ids"],
        _prepare_cp_local_context=lambda parsed: (
            {"psp": parsed} if pre_sharded else None
        ),
        _build_packed_positions=build_positions,
    )
    install_qwen3_vl_cp_position_ids(
        bridge_model=bridge_model, miles_qwen3_vl=miles_qwen3_vl
    )
    install_qwen3_vl_cp_position_ids(
        bridge_model=bridge_model, miles_qwen3_vl=miles_qwen3_vl
    )
    return Qwen3VLModel(), calls


def test_pre_sharded_cp_input_gets_explicit_position_ids():
    model, calls = _install(pre_sharded=True)
    assert model.forward(input_ids="row", position_ids=None) == "out"
    assert calls == [{"input_ids": "row", "position_ids": ("positions", "row")}]


def test_other_inputs_pass_through():
    model, calls = _install(pre_sharded=False)
    model.forward(input_ids="row", position_ids=None)
    model.forward(input_ids="row", position_ids="explicit")
    assert calls == [
        {"input_ids": "row", "position_ids": None},
        {"input_ids": "row", "position_ids": "explicit"},
    ]
