from pathlib import Path, PurePosixPath
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
LICENSE_DIR = ROOT / "THIRD-PARTY-LICENSES"


def _runtime_files() -> list[str]:
    return [
        line.strip()
        for line in (ROOT / "OCR-RUNTIME-FILES.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


class ThirdPartyLicenseTests(unittest.TestCase):
    def test_ocr_runtime_allowlist_contains_only_safe_unique_relative_paths(self):
        paths = _runtime_files()
        self.assertTrue(paths)
        self.assertEqual(len(paths), len(set(paths)))
        for value in paths:
            path = PurePosixPath(value)
            self.assertFalse(path.is_absolute())
            self.assertNotIn("..", path.parts)
            self.assertNotIn("\\", value)
            self.assertIsNone(re.match(r"^[A-Za-z]:", value))

    def test_every_bundled_ocr_file_is_mapped_to_a_component_and_license(self):
        index = (LICENSE_DIR / "README.md").read_text(encoding="utf-8")
        for runtime_file in _runtime_files():
            self.assertIn(f"`{runtime_file}`", index)

    def test_every_license_notice_named_by_the_index_is_present_and_nonempty(self):
        index = (LICENSE_DIR / "README.md").read_text(encoding="utf-8")
        notice_names = {
            name
            for name in re.findall(r"`([^`]+\.(?:txt|md))`", index, flags=re.IGNORECASE)
            if "/" not in name and "\\" not in name
        }
        self.assertTrue(notice_names)
        for name in notice_names:
            notice = LICENSE_DIR / name
            self.assertTrue(notice.is_file(), name)
            self.assertGreater(notice.stat().st_size, 100, name)

    def test_copyleft_runtime_libraries_have_corresponding_source_links(self):
        source_offer = (LICENSE_DIR / "SOURCE-OFFER.md").read_text(encoding="utf-8")
        expected = {
            "libjbig-0.dll": "jbigkit-2.1.tar.gz",
            "libiconv-2.dll": "libiconv-1.17.tar.gz",
            "libgcc_s_seh-1.dll": "gcc-14.1.0.tar.xz",
            "libstdc++-6.dll": "gcc-14.1.0.tar.xz",
        }
        for library, archive in expected.items():
            self.assertIn(f"`{library}`", source_offer)
            self.assertIn(archive, source_offer)

    def test_release_workflow_enforces_and_packages_license_inventory(self):
        workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("OCR-RUNTIME-FILES.txt", workflow)
        self.assertIn("THIRD-PARTY-LICENSES", workflow)
        self.assertIn("ocr-expected-sorted.txt", workflow)
        self.assertIn("ocr-actual.txt", workflow)
        self.assertIn("diff -u", workflow)
        self.assertNotIn('cp -R "$runtime_root', workflow)
