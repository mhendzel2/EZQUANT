"""
Manual correction tools for editing segmentation masks
"""

from PySide6.QtWidgets import (QWidget, QToolBar, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QMessageBox, QInputDialog)
from PySide6.QtCore import Signal, Qt, QPoint
from PySide6.QtGui import QAction, QIcon, QCursor
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.measure import label
from typing import List, Tuple, Optional
from collections import deque


class EditCommand:
    """Base class for undoable edit commands"""
    
    def __init__(self, mask: np.ndarray):
        self.old_mask = mask.copy()
        self.new_mask = None
    
    def execute(self, mask: np.ndarray) -> np.ndarray:
        """Execute the command and return new mask"""
        raise NotImplementedError
    
    def get_description(self) -> str:
        """Get description of this edit"""
        raise NotImplementedError


class SplitCommand(EditCommand):
    """Split a nucleus along a drawn line"""
    
    def __init__(self, mask: np.ndarray, nucleus_id: int, line_points: List[Tuple[int, int]]):
        super().__init__(mask)
        self.nucleus_id = nucleus_id
        self.line_points = line_points
    
    def execute(self, mask: np.ndarray) -> np.ndarray:
        """Split nucleus using watershed"""
        new_mask = mask.copy()
        
        # Create binary mask of target nucleus
        nucleus_mask = (mask == self.nucleus_id).astype(np.uint8)
        
        # Create marker mask with line as separation
        markers = np.zeros_like(nucleus_mask)
        
        # Draw line on markers (set to background)
        for y, x in self.line_points:
            if 0 <= y < markers.shape[0] and 0 <= x < markers.shape[1]:
                markers[y, x] = -1  # Barrier
        
        # Erode nucleus slightly to create markers
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(nucleus_mask, iterations=2)
        
        # Label eroded regions - these become seed markers
        seed_labels = label(eroded)
        
        # Apply line barrier to seed labels
        for y, x in self.line_points:
            if 0 <= y < seed_labels.shape[0] and 0 <= x < seed_labels.shape[1]:
                seed_labels[y, x] = 0
        
        # If we have 2+ separate regions, perform watershed
        n_seeds = seed_labels.max()
        if n_seeds >= 2:
            # Compute distance transform for watershed
            distance = ndimage.distance_transform_edt(nucleus_mask)
            
            # Watershed to split
            split_result = watershed(-distance, seed_labels, mask=nucleus_mask)
            
            # Replace original nucleus with split results
            new_mask[mask == self.nucleus_id] = 0
            
            # Find next available ID
            max_id = new_mask.max()
            
            for i in range(1, n_seeds + 1):
                region = (split_result == i)
                if region.sum() > 0:
                    new_mask[region] = max_id + i
        
        self.new_mask = new_mask
        return new_mask
    
    def get_description(self) -> str:
        return f"Split nucleus {self.nucleus_id}"


class MergeCommand(EditCommand):
    """Merge multiple nuclei into one"""
    
    def __init__(self, mask: np.ndarray, nucleus_ids: List[int]):
        super().__init__(mask)
        self.nucleus_ids = nucleus_ids
    
    def execute(self, mask: np.ndarray) -> np.ndarray:
        """Merge nuclei into first ID"""
        new_mask = mask.copy()
        
        if len(self.nucleus_ids) < 2:
            return new_mask
        
        # Merge all into the first ID
        target_id = self.nucleus_ids[0]
        
        for nuc_id in self.nucleus_ids[1:]:
            new_mask[mask == nuc_id] = target_id
        
        self.new_mask = new_mask
        return new_mask
    
    def get_description(self) -> str:
        return f"Merge nuclei {self.nucleus_ids}"


class DeleteCommand(EditCommand):
    """Delete a nucleus"""
    
    def __init__(self, mask: np.ndarray, nucleus_id: int):
        super().__init__(mask)
        self.nucleus_id = nucleus_id
    
    def execute(self, mask: np.ndarray) -> np.ndarray:
        """Remove nucleus from mask"""
        new_mask = mask.copy()
        new_mask[mask == self.nucleus_id] = 0
        
        self.new_mask = new_mask
        return new_mask
    
    def get_description(self) -> str:
        return f"Delete nucleus {self.nucleus_id}"


class AddCommand(EditCommand):
    """Add a new nucleus from drawn ROI"""
    
    def __init__(self, mask: np.ndarray, roi_points: List[Tuple[int, int]]):
        super().__init__(mask)
        self.roi_points = roi_points
    
    def execute(self, mask: np.ndarray) -> np.ndarray:
        """Add new nucleus from polygon ROI"""
        new_mask = mask.copy()
        
        if len(self.roi_points) < 3:
            return new_mask
        
        # Create polygon mask
        from skimage.draw import polygon
        
        # Convert points to arrays
        rr = [p[0] for p in self.roi_points]
        cc = [p[1] for p in self.roi_points]
        
        # Create filled polygon
        poly_mask = np.zeros(mask.shape, dtype=bool)
        
        try:
            poly_r, poly_c = polygon(rr, cc, shape=mask.shape)
            poly_mask[poly_r, poly_c] = True
        except Exception as e:
            print(f"Error creating polygon: {e}")
            return new_mask
        
        # Find next available ID
        new_id = new_mask.max() + 1
        
        # Add new nucleus
        new_mask[poly_mask] = new_id
        
        self.new_mask = new_mask
        return new_mask
    
    def get_description(self) -> str:
        return f"Add new nucleus"


class CorrectionToolbar(QWidget):
    """Toolbar for manual correction tools"""
    
    tool_changed = Signal(str)  # Emit tool name: "split", "merge", "delete", "add", "select"
    undo_requested = Signal()
    redo_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_tool = "select"
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup toolbar UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Tool selection buttons
        self.select_btn = QPushButton("Select")
        self.select_btn.setCheckable(True)
        self.select_btn.setChecked(True)
        self.select_btn.clicked.connect(lambda: self._set_tool("select"))
        layout.addWidget(self.select_btn)
        
        self.split_btn = QPushButton("Split")
        self.split_btn.setCheckable(True)
        self.split_btn.clicked.connect(lambda: self._set_tool("split"))
        self.split_btn.setToolTip("Draw line to split nucleus")
        layout.addWidget(self.split_btn)
        
        self.merge_btn = QPushButton("Merge")
        self.merge_btn.setCheckable(True)
        self.merge_btn.clicked.connect(lambda: self._set_tool("merge"))
        self.merge_btn.setToolTip("Click nuclei to merge (Ctrl+click for multiple)")
        layout.addWidget(self.merge_btn)
        
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setCheckable(True)
        self.delete_btn.clicked.connect(lambda: self._set_tool("delete"))
        self.delete_btn.setToolTip("Click nucleus to delete")
        layout.addWidget(self.delete_btn)
        
        self.add_btn = QPushButton("Add")
        self.add_btn.setCheckable(True)
        self.add_btn.clicked.connect(lambda: self._set_tool("add"))
        self.add_btn.setToolTip("Draw polygon to add nucleus")
        layout.addWidget(self.add_btn)
        
        layout.addSpacing(20)
        
        # Undo/Redo buttons
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.undo_requested.emit)
        layout.addWidget(self.undo_btn)
        
        self.redo_btn = QPushButton("Redo")
        self.redo_btn.clicked.connect(self.redo_requested.emit)
        layout.addWidget(self.redo_btn)
        
        layout.addStretch()
        
        # Edit counter
        self.edit_count_label = QLabel("Edits: 0")
        layout.addWidget(self.edit_count_label)
    
    def _set_tool(self, tool: str):
        """Set active tool"""
        self.current_tool = tool
        
        # Update button states
        self.select_btn.setChecked(tool == "select")
        self.split_btn.setChecked(tool == "split")
        self.merge_btn.setChecked(tool == "merge")
        self.delete_btn.setChecked(tool == "delete")
        self.add_btn.setChecked(tool == "add")
        
        self.tool_changed.emit(tool)
    
    def update_edit_count(self, count: int):
        """Update edit counter"""
        self.edit_count_label.setText(f"Edits: {count}")
    
    def set_undo_enabled(self, enabled: bool):
        """Enable/disable undo button"""
        self.undo_btn.setEnabled(enabled)
    
    def set_redo_enabled(self, enabled: bool):
        """Enable/disable redo button"""
        self.redo_btn.setEnabled(enabled)


class ManualCorrectionManager:
    """Manager for manual correction operations with undo/redo"""
    
    def __init__(self, initial_mask: np.ndarray):
        self.initial_mask = initial_mask.copy()
        self.current_mask = initial_mask.copy()
        
        self.undo_stack: deque = deque(maxlen=50)  # Max 50 undo operations
        self.redo_stack: deque = deque(maxlen=50)
        
        self.edit_log: List[str] = []
        self.edited_nuclei: set = set()
    
    def execute_command(self, command: EditCommand) -> np.ndarray:
        """Execute a command and add to undo stack"""
        # Execute command
        new_mask = command.execute(self.current_mask)
        
        # Add to undo stack
        self.undo_stack.append(command)
        
        # Clear redo stack
        self.redo_stack.clear()
        
        # Update current mask
        self.current_mask = new_mask
        
        # Log the edit
        self.edit_log.append(command.get_description())
        
        # Track edited nuclei
        if isinstance(command, (SplitCommand, MergeCommand, DeleteCommand)):
            if isinstance(command, SplitCommand):
                self.edited_nuclei.add(command.nucleus_id)
            elif isinstance(command, MergeCommand):
                self.edited_nuclei.update(command.nucleus_ids)
            elif isinstance(command, DeleteCommand):
                self.edited_nuclei.add(command.nucleus_id)
        
        return new_mask
    
    def undo(self) -> Optional[np.ndarray]:
        """Undo last operation"""
        if not self.undo_stack:
            return None
        
        # Pop last command
        command = self.undo_stack.pop()
        
        # Add to redo stack
        self.redo_stack.append(command)
        
        # Restore previous mask
        if self.undo_stack:
            # Re-execute all commands up to this point
            self.current_mask = self.initial_mask.copy()
            for cmd in self.undo_stack:
                self.current_mask = cmd.execute(self.current_mask)
        else:
            # No more commands, restore initial
            self.current_mask = self.initial_mask.copy()
        
        return self.current_mask
    
    def redo(self) -> Optional[np.ndarray]:
        """Redo last undone operation"""
        if not self.redo_stack:
            return None
        
        # Pop from redo stack
        command = self.redo_stack.pop()
        
        # Re-execute
        self.current_mask = command.execute(self.current_mask)
        
        # Add back to undo stack
        self.undo_stack.append(command)
        
        return self.current_mask
    
    def get_current_mask(self) -> np.ndarray:
        """Get current mask"""
        return self.current_mask
    
    def can_undo(self) -> bool:
        """Check if undo is available"""
        return len(self.undo_stack) > 0
    
    def can_redo(self) -> bool:
        """Check if redo is available"""
        return len(self.redo_stack) > 0
    
    def get_edit_count(self) -> int:
        """Get number of edits"""
        return len(self.undo_stack)
    
    def get_edit_log(self) -> List[str]:
        """Get list of edit descriptions"""
        return self.edit_log.copy()
    
    def get_edited_nuclei(self) -> set:
        """Get set of nucleus IDs that were edited"""
        return self.edited_nuclei.copy()
