from __future__ import annotations

import unittest

from clean_extracted_text import clean_hybrid_pdf_pages
from ocr_support import detected_language, select_languages


class OcrLanguageSelectionTests(unittest.TestCase):
    def test_zotero_language_selects_primary_language_plus_english(self):
        selection = select_languages({"eng", "pol", "osd"}, "pl-PL", "", "auto")
        self.assertEqual(selection.languages, "pol+eng")
        self.assertEqual(selection.detected, "pol")
        self.assertIsNone(selection.missing)

    def test_title_characters_detect_polish_when_metadata_is_empty(self):
        self.assertEqual(detected_language("", "Wstępna propozycja projektu badawczego"), "pol")

    def test_words_can_detect_polish_without_diacritics(self):
        self.assertEqual(detected_language("", "Raport dla praktykow prawa oraz podmiotow prawnych"), "pol")

    def test_missing_detected_language_falls_back_and_is_reported(self):
        selection = select_languages({"eng", "osd"}, "pl", "", "auto")
        self.assertEqual(selection.languages, "eng")
        self.assertEqual(selection.missing, "pol")

    def test_manual_language_override_remains_available(self):
        selection = select_languages({"eng", "deu"}, "pl", "", "deu+eng")
        self.assertEqual(selection.languages, "deu+eng")
        self.assertEqual(selection.mode, "manual")


class PageLevelOcrMergeTests(unittest.TestCase):
    @staticmethod
    def _layout_block(text: str, y: float = 100.0):
        return {
            "text": text, "lines": [text], "bbox": (50.0, y, 500.0, y + 30.0),
            "size": 11.0, "bold_ratio": 0.0, "italic_ratio": 0.0,
        }

    def test_only_the_ocr_page_replaces_native_layout(self):
        native = "This native paragraph remains available and keeps its PDF layout."
        recovered = "Recovered OCR paragraph from a previously image-only page."
        pages = [
            {"number": 1, "width": 600.0, "height": 800.0, "native_text": native,
             "blocks": [self._layout_block(native)]},
            {"number": 2, "width": 600.0, "height": 800.0, "native_text": "2",
             "blocks": [self._layout_block("2")]},
        ]
        blocks, metrics, method = clean_hybrid_pdf_pages(pages, [(1, native), (2, recovered)])
        combined = "\n".join(block["text"] for block in blocks)
        self.assertEqual(method, "pdf-layout+page-ocr")
        self.assertEqual(metrics["ocr_pages"], 1)
        self.assertIn(native, combined)
        self.assertIn(recovered, combined)
        self.assertNotIn("\n2\n", f"\n{combined}\n")
        self.assertTrue(any("ocr-page" in block["quality_flags"] for block in blocks))

    def test_native_only_document_keeps_layout_pipeline(self):
        native = "A normal text PDF remains in the layout-aware pipeline."
        pages = [{"number": 1, "width": 600.0, "height": 800.0, "native_text": native,
                  "blocks": [self._layout_block(native)]}]
        blocks, metrics, method = clean_hybrid_pdf_pages(pages, [(1, native)])
        self.assertEqual(method, "pdf-layout")
        self.assertEqual(metrics["ocr_pages"], 0)
        self.assertIn(native, "\n".join(block["text"] for block in blocks))


if __name__ == "__main__":
    unittest.main()
