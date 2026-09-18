from otc_research.signals.confidence import ALTA, BAJA, MEDIA, classify_confidence


def test_alta_requires_big_margin_and_ci_low_clearing_break_even():
    assert classify_confidence(0.06, 0.60, 0.54) == ALTA


def test_alta_not_given_when_ci_low_does_not_clear_break_even_despite_big_margin():
    # point estimate implies a big margin, but the CI lower bound doesn't
    # clear break-even -- must not be ALTA (or the CI would be pointless)
    assert classify_confidence(0.06, 0.50, 0.54) == MEDIA


def test_media_for_margin_at_or_above_the_candidacy_floor():
    assert classify_confidence(0.03, 0.50, 0.54) == MEDIA
    assert classify_confidence(0.045, 0.20, 0.54) == MEDIA


def test_baja_below_the_media_floor():
    assert classify_confidence(0.01, 0.60, 0.54) == BAJA
    assert classify_confidence(-0.10, 0.60, 0.54) == BAJA


def test_baja_when_any_grounding_value_is_missing():
    assert classify_confidence(None, 0.60, 0.54) == BAJA
    assert classify_confidence(0.06, None, 0.54) == BAJA
    assert classify_confidence(0.06, 0.60, None) == BAJA
    assert classify_confidence(None, None, None) == BAJA
