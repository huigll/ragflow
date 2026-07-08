from types import SimpleNamespace
from unittest.mock import MagicMock

from PIL import Image

from deepdoc.parser.pdf_parser import RAGFlowPdfParser


def _ocr_returning(conf):
    # One OCR region shaped like the real OCR output: (bbox, (text, confidence)).
    bbox = [(0, 0), (10, 0), (10, 10), (0, 10)]
    return [(bbox, ("cell", conf))]


def _fake_self(ocr_conf):
    s = SimpleNamespace()
    s.ocr = MagicMock(return_value=_ocr_returning(ocr_conf))
    return s


def _table_img():
    return Image.new("RGB", (40, 20), "white")


def test_upright_table_skips_extra_rotations():
    # A confident upright table (0deg score >= 0.8) must short-circuit: the
    # threshold rule always keeps 0deg in that case, so the 90/180/270 OCR
    # passes are pure waste and must be skipped.
    s = _fake_self(0.95)
    best_angle, best_img, results = RAGFlowPdfParser._evaluate_table_orientation(s, _table_img())
    assert best_angle == 0
    assert s.ocr.call_count == 1
    assert best_img is not None


def test_low_confidence_table_evaluates_all_rotations():
    # When 0deg is not confident (< 0.8) a rotation could still win, so all four
    # angles must be evaluated (behavior unchanged from before the optimization).
    s = _fake_self(0.5)
    best_angle, best_img, results = RAGFlowPdfParser._evaluate_table_orientation(s, _table_img())
    assert s.ocr.call_count == 4
