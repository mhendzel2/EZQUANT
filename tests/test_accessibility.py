import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QPushButton, QLineEdit, QWidget, QVBoxLayout

from gui.accessibility import AccessibilityManager, CVDPalette, apply_accessible_names


class TestAccessibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_apply_accessible_names(self):
        root = QWidget()
        layout = QVBoxLayout(root)

        button = QPushButton("Run Segmentation")
        line_edit = QLineEdit()
        line_edit.setObjectName("project_name_input")

        layout.addWidget(button)
        layout.addWidget(line_edit)

        apply_accessible_names(root)

        self.assertEqual(button.accessibleName(), "Run Segmentation")
        self.assertEqual(line_edit.accessibleName(), "project name input")

    def test_colorblind_palette_channels(self):
        AccessibilityManager.update_state(colorblind_safe=True, palette=CVDPalette.OKABE_ITO.value)
        colors = AccessibilityManager.get_channel_colors(9)
        self.assertEqual(len(colors), 9)
        self.assertTrue(all(isinstance(c, str) and c.startswith("#") for c in colors))

    def test_segmentation_description_format(self):
        text = AccessibilityManager.build_segmentation_description(
            nucleus_count=246,
            channel_name="DAPI",
            zoom_percent=100.0,
            slice_index=0,
        )
        self.assertIn("246 nuclei", text)
        self.assertIn("DAPI", text)
        self.assertIn("zoom 100%", text)


if __name__ == "__main__":
    unittest.main()
