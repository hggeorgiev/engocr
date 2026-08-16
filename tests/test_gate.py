from engocr.gate import math_density, page_needs_vision


def test_pure_prose_page_skips_vision():
    assert page_needs_vision(
        text_chars=3000, image_count=0, drawing_count=0,
        math_sym_density=0.0, math_sym_count=0,
    ) is False


def test_scanned_page_triggers_vision():
    assert page_needs_vision(
        text_chars=40, image_count=0, drawing_count=0,
        math_sym_density=0.0, math_sym_count=0,
    ) is True


def test_embedded_figure_triggers_vision():
    assert page_needs_vision(
        text_chars=3000, image_count=1, drawing_count=0,
        math_sym_density=0.0, math_sym_count=0,
    ) is True


def test_vector_drawings_trigger_vision():
    assert page_needs_vision(
        text_chars=3000, image_count=0, drawing_count=12,
        math_sym_density=0.0, math_sym_count=0,
    ) is True


def test_math_density_triggers_vision():
    assert page_needs_vision(
        text_chars=3000, image_count=0, drawing_count=0,
        math_sym_density=0.05, math_sym_count=150,
    ) is True


def test_sparse_math_still_triggers_vision():
    # quality-first: even a handful of math symbols routes to vision
    assert page_needs_vision(
        text_chars=3000, image_count=0, drawing_count=0,
        math_sym_density=0.003, math_sym_count=9,
    ) is True


def test_force_all_overrides_everything():
    assert page_needs_vision(
        text_chars=5000, image_count=0, drawing_count=0,
        math_sym_density=0.0, math_sym_count=0, force_all=True,
    ) is True


def test_math_density_helper():
    density, count = math_density("Let α ∈ ℝ and ∫ f dx = ∑ aₙ over n samples.")
    assert count >= 4
    assert 0.0 < density < 1.0
    assert math_density("") == (0.0, 0)
