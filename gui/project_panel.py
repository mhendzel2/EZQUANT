"""
Project panel for managing images in a project
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QListWidget, QListWidgetItem, QLabel, QMenu,
                               QMessageBox, QToolButton)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon
from pathlib import Path
from typing import Optional, List


class ProjectPanel(QWidget):
    """
    Panel for managing project images
    """
    
    # Signals
    add_images_requested = Signal()  # User wants to add images
    image_selected = Signal(int)  # Image index selected
    remove_image_requested = Signal(int)  # Remove image at index
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Header
        header_layout = QHBoxLayout()
        header_label = QLabel("<b>Project Images</b>")
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # Image list
        self.image_list = QListWidget()
        self.image_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.image_list.customContextMenuRequested.connect(self._show_context_menu)
        self.image_list.itemSelectionChanged.connect(self._on_selection_changed)
        self.image_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.image_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.add_btn = QPushButton("Add Images...")
        self.add_btn.setToolTip("Add one or more images to the project")
        self.add_btn.clicked.connect(self.add_images_requested.emit)
        button_layout.addWidget(self.add_btn)
        
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.setToolTip("Remove selected image from project")
        self.remove_btn.setEnabled(False)
        self.remove_btn.clicked.connect(self._on_remove_clicked)
        button_layout.addWidget(self.remove_btn)
        
        layout.addLayout(button_layout)
        
        # Status
        self.status_label = QLabel("0 images")
        self.status_label.setStyleSheet("QLabel { color: gray; font-size: 10pt; }")
        layout.addWidget(self.status_label)
    
    def add_image(self, filename: str, has_segmentation: bool = False):
        """Add an image to the list"""
        item = QListWidgetItem(filename)
        
        # Set icon based on segmentation status
        if has_segmentation:
            item.setIcon(self._get_icon("✓"))  # Checkmark for segmented
            item.setToolTip(f"{filename}\n[Segmented]")
        else:
            item.setIcon(self._get_icon("○"))  # Circle for not segmented
            item.setToolTip(f"{filename}\n[Not segmented]")
        
        self.image_list.addItem(item)
        self._update_status()
    
    def remove_image(self, index: int):
        """Remove an image from the list"""
        if 0 <= index < self.image_list.count():
            self.image_list.takeItem(index)
            self._update_status()
    
    def clear_images(self):
        """Clear all images from the list"""
        self.image_list.clear()
        self._update_status()
    
    def set_image_segmented(self, index: int, segmented: bool = True):
        """Mark an image as segmented or not"""
        if 0 <= index < self.image_list.count():
            item = self.image_list.item(index)
            filename = item.text()
            
            if segmented:
                item.setIcon(self._get_icon("✓"))
                item.setToolTip(f"{filename}\n[Segmented]")
            else:
                item.setIcon(self._get_icon("○"))
                item.setToolTip(f"{filename}\n[Not segmented]")
    
    def get_selected_index(self) -> Optional[int]:
        """Get the currently selected image index"""
        current_row = self.image_list.currentRow()
        return current_row if current_row >= 0 else None
    
    def set_selected_index(self, index: int):
        """Set the selected image"""
        if 0 <= index < self.image_list.count():
            self.image_list.setCurrentRow(index)
    
    def get_image_count(self) -> int:
        """Get the number of images in the project"""
        return self.image_list.count()
    
    def _update_status(self):
        """Update the status label"""
        count = self.image_list.count()
        
        # Count segmented images
        segmented_count = 0
        for i in range(count):
            item = self.image_list.item(i)
            if "✓" in item.icon().name() or "[Segmented]" in item.toolTip():
                segmented_count += 1
        
        if count == 0:
            self.status_label.setText("0 images")
        elif segmented_count == 0:
            self.status_label.setText(f"{count} image(s), none segmented")
        else:
            self.status_label.setText(f"{count} image(s), {segmented_count} segmented")
        
        # Enable/disable remove button
        self.remove_btn.setEnabled(self.image_list.currentRow() >= 0)
    
    def _on_selection_changed(self):
        """Handle selection change"""
        index = self.get_selected_index()
        if index is not None:
            self.image_selected.emit(index)
        self._update_status()
    
    def _on_item_double_clicked(self, item: QListWidgetItem):
        """Handle double-click to select and load image"""
        index = self.image_list.row(item)
        self.image_selected.emit(index)
    
    def _on_remove_clicked(self):
        """Handle remove button click"""
        index = self.get_selected_index()
        if index is not None:
            # Confirm removal
            item = self.image_list.item(index)
            reply = QMessageBox.question(
                self,
                "Remove Image",
                f"Remove '{item.text()}' from project?\n\n"
                "This will not delete the file from disk.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.remove_image_requested.emit(index)
    
    def _show_context_menu(self, pos):
        """Show context menu for image list"""
        item = self.image_list.itemAt(pos)
        if item is None:
            return
        
        menu = QMenu(self)
        
        load_action = menu.addAction("Load Image")
        load_action.triggered.connect(lambda: self._on_item_double_clicked(item))
        
        menu.addSeparator()
        
        remove_action = menu.addAction("Remove from Project")
        remove_action.triggered.connect(self._on_remove_clicked)
        
        menu.exec(self.image_list.mapToGlobal(pos))
    
    def _get_icon(self, text: str):
        """Create a simple text-based icon"""
        # For simplicity, we'll use the text as icon name
        # In a real app, you'd use proper icons
        icon = QIcon()
        icon.addFile(text)
        return icon
