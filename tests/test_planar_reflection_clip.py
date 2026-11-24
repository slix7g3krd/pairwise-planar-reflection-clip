import pytest
import graphics.planar_reflection_clip as mod

CASES = [
    ("oblique_clip", (1,)),
    ("recursion_cap", (1,)),
    ("visibility_match", (1,)),
    ("transparent_order", (1,)),
    ("resolution_link", (1,)),
    ("intensity", (1,)),
    ("reflection_cam", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("depth_precision", (1,)),
    ("metrics", (1,)),
    ("cull_plane", (1,)),
    ("blur", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
