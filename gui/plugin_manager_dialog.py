"""
Plugin Manager Dialog for managing measurement plugins
"""

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QListWidget, QListWidgetItem, QGroupBox,
                               QTextEdit, QFileDialog, QMessageBox, QCheckBox,
                               QSplitter, QWidget)
from PySide6.QtCore import Qt, Signal
from pathlib import Path
from typing import Optional


class PluginManagerDialog(QDialog):
    """Dialog for managing measurement plugins"""
    
    plugins_changed = Signal()
    
    def __init__(self, plugin_loader, parent=None):
        super().__init__(parent)
        self.plugin_loader = plugin_loader
        self.changes_made = False
        
        self.setWindowTitle("Plugin Manager")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        
        self._setup_ui()
        self._populate_plugins()
    
    def _setup_ui(self):
        """Setup dialog UI"""
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel(
            "<h3>Measurement Plugins</h3>"
            "<p>Manage custom measurement plugins for extended analysis capabilities.</p>"
        )
        layout.addWidget(header_label)
        
        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Plugin list
        list_widget = QWidget()
        list_layout = QVBoxLayout(list_widget)
        list_layout.setContentsMargins(0, 0, 0, 0)
        
        list_layout.addWidget(QLabel("<b>Installed Plugins</b>"))
        
        self.plugin_list = QListWidget()
        self.plugin_list.currentItemChanged.connect(self._on_plugin_selected)
        list_layout.addWidget(self.plugin_list)
        
        # Plugin list buttons
        list_btn_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_plugins)
        list_btn_layout.addWidget(self.refresh_btn)
        
        self.add_btn = QPushButton("Add Plugin...")
        self.add_btn.clicked.connect(self._add_plugin)
        list_btn_layout.addWidget(self.add_btn)
        
        self.open_folder_btn = QPushButton("Open Folder")
        self.open_folder_btn.clicked.connect(self._open_plugin_folder)
        list_btn_layout.addWidget(self.open_folder_btn)
        
        list_layout.addLayout(list_btn_layout)
        
        splitter.addWidget(list_widget)
        
        # Right side: Plugin details
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        details_layout.setContentsMargins(0, 0, 0, 0)
        
        details_layout.addWidget(QLabel("<b>Plugin Details</b>"))
        
        # Details group
        details_group = QGroupBox()
        details_group_layout = QVBoxLayout()
        
        self.name_label = QLabel("<b>Name:</b> -")
        details_group_layout.addWidget(self.name_label)
        
        self.version_label = QLabel("<b>Version:</b> -")
        details_group_layout.addWidget(self.version_label)
        
        self.file_label = QLabel("<b>File:</b> -")
        details_group_layout.addWidget(self.file_label)
        
        self.enabled_check = QCheckBox("Enabled")
        self.enabled_check.setChecked(True)
        self.enabled_check.stateChanged.connect(self._on_enabled_changed)
        details_group_layout.addWidget(self.enabled_check)
        
        details_group.setLayout(details_group_layout)
        details_layout.addWidget(details_group)
        
        # Description
        details_layout.addWidget(QLabel("<b>Description:</b>"))
        self.description_text = QTextEdit()
        self.description_text.setReadOnly(True)
        self.description_text.setMaximumHeight(100)
        details_layout.addWidget(self.description_text)
        
        # Measurements provided
        details_layout.addWidget(QLabel("<b>Measurements Provided:</b>"))
        self.measurements_text = QTextEdit()
        self.measurements_text.setReadOnly(True)
        details_layout.addWidget(self.measurements_text)
        
        # Errors
        self.errors_group = QGroupBox("Load Errors")
        errors_layout = QVBoxLayout()
        self.errors_text = QTextEdit()
        self.errors_text.setReadOnly(True)
        self.errors_text.setMaximumHeight(80)
        self.errors_text.setStyleSheet("color: red;")
        errors_layout.addWidget(self.errors_text)
        self.errors_group.setLayout(errors_layout)
        self.errors_group.setVisible(False)
        details_layout.addWidget(self.errors_group)
        
        splitter.addWidget(details_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        layout.addWidget(splitter)
        
        # Dialog buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.reload_btn = QPushButton("Reload All")
        self.reload_btn.clicked.connect(self._reload_all)
        button_layout.addWidget(self.reload_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
    
    def _populate_plugins(self):
        """Populate plugin list"""
        self.plugin_list.clear()
        
        # Get all plugin info
        plugins = self.plugin_loader.get_all_plugin_info()
        
        for plugin in plugins:
            item = QListWidgetItem(plugin['name'])
            item.setData(Qt.UserRole, plugin)
            
            # Mark enabled/disabled
            if plugin.get('enabled', True):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            
            self.plugin_list.addItem(item)
        
        # Show errors if any
        errors = self.plugin_loader.get_errors()
        if errors:
            self.errors_group.setVisible(True)
            error_text = "\n".join([f"• {k}: {v}" for k, v in errors.items()])
            self.errors_text.setText(error_text)
        else:
            self.errors_group.setVisible(False)
    
    def _on_plugin_selected(self, current: Optional[QListWidgetItem], previous: Optional[QListWidgetItem]):
        """Handle plugin selection"""
        if current is None:
            self._clear_details()
            return
        
        plugin_info = current.data(Qt.UserRole)
        if plugin_info is None:
            self._clear_details()
            return
        
        # Update details
        self.name_label.setText(f"<b>Name:</b> {plugin_info.get('name', '-')}")
        self.version_label.setText(f"<b>Version:</b> {plugin_info.get('version', '1.0.0')}")
        self.file_label.setText(f"<b>File:</b> {plugin_info.get('file', '-')}")
        self.enabled_check.setChecked(plugin_info.get('enabled', True))
        self.description_text.setText(plugin_info.get('description', 'No description available.'))
        
        # Get measurements from plugin instance
        plugin_name = plugin_info.get('file')
        instance = self.plugin_loader.get_plugin_instance(plugin_name)
        
        if instance:
            try:
                measurements = instance.get_required_measurements()
                if measurements:
                    self.measurements_text.setText("\n".join([f"• {m}" for m in measurements]))
                else:
                    self.measurements_text.setText("(Dynamic measurements)")
            except Exception:
                self.measurements_text.setText("Unable to retrieve measurements list.")
        else:
            self.measurements_text.setText("-")
    
    def _clear_details(self):
        """Clear plugin details panel"""
        self.name_label.setText("<b>Name:</b> -")
        self.version_label.setText("<b>Version:</b> -")
        self.file_label.setText("<b>File:</b> -")
        self.description_text.clear()
        self.measurements_text.clear()
    
    def _on_enabled_changed(self, state: int):
        """Handle enabled checkbox change"""
        current = self.plugin_list.currentItem()
        if current:
            plugin_info = current.data(Qt.UserRole)
            if plugin_info:
                plugin_info['enabled'] = (state == Qt.Checked)
                current.setData(Qt.UserRole, plugin_info)
                current.setCheckState(Qt.Checked if state == Qt.Checked else Qt.Unchecked)
                self.changes_made = True
    
    def _refresh_plugins(self):
        """Refresh plugin list"""
        # Discover new plugins
        self.plugin_loader.discover_plugins()
        self._populate_plugins()
        self.changes_made = True
        QMessageBox.information(self, "Refreshed", "Plugin list has been refreshed.")
    
    def _reload_all(self):
        """Reload all plugins"""
        count = self.plugin_loader.reload_plugins()
        self._populate_plugins()
        self.changes_made = True
        QMessageBox.information(
            self,
            "Plugins Reloaded",
            f"Reloaded {count} plugins successfully."
        )
    
    def _add_plugin(self):
        """Add a plugin file"""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Plugin File",
            str(self.plugin_loader.plugin_directory),
            "Python Files (*.py);;All Files (*)"
        )
        
        if filepath:
            # Copy to plugins directory if not already there
            src_path = Path(filepath)
            dest_dir = self.plugin_loader.plugin_directory / "examples"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / src_path.name
            
            if src_path.parent != dest_dir:
                try:
                    import shutil
                    shutil.copy2(src_path, dest_path)
                    QMessageBox.information(
                        self,
                        "Plugin Added",
                        f"Plugin copied to:\n{dest_path}\n\n"
                        "Click 'Reload All' to load the new plugin."
                    )
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "Error",
                        f"Could not copy plugin:\n{e}"
                    )
            else:
                QMessageBox.information(
                    self,
                    "Plugin Location",
                    "Plugin is already in the plugins directory.\n"
                    "Click 'Reload All' to reload plugins."
                )
            
            self.changes_made = True
    
    def _open_plugin_folder(self):
        """Open plugins folder in file explorer"""
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        
        folder_path = self.plugin_loader.plugin_directory
        if folder_path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder_path)))
        else:
            QMessageBox.warning(
                self,
                "Folder Not Found",
                f"Plugins folder not found:\n{folder_path}"
            )
    
    def accept(self):
        """Handle dialog acceptance"""
        if self.changes_made:
            self.plugins_changed.emit()
        super().accept()
