"""
Worker thread for running measurements in the background
"""

from PySide6.QtCore import QThread, Signal
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import time

from core.measurements import MeasurementEngine
from core.plugin_loader import PluginLoader


class MeasurementWorker(QThread):
    """Worker thread for executing measurements"""
    
    # Signals
    finished = Signal(pd.DataFrame)  # measurements DataFrame
    error = Signal(str)  # error message
    progress = Signal(int)  # progress percentage
    status = Signal(str)  # status message
    
    def __init__(self,
                 masks: np.ndarray,
                 intensity_images: Dict[str, np.ndarray],
                 config: Dict,
                 plugin_loader: Optional[PluginLoader] = None):
        """
        Initialize measurement worker
        
        Args:
            masks: Labeled segmentation masks
            intensity_images: Dict of channel_name -> intensity image
            config: Configuration dict with:
                - is_3d: bool
                - enabled_categories: List[str]
                - enabled_plugins: List[str]
                - assign_phases: bool
                - dna_channel: str (optional)
            plugin_loader: PluginLoader instance for custom plugins
        """
        super().__init__()
        
        self.masks = masks
        self.intensity_images = intensity_images
        self.config = config
        self.plugin_loader = plugin_loader
    
    def run(self):
        """Execute measurements"""
        try:
            self.status.emit("Initializing measurement engine...")
            self.progress.emit(5)
            
            # Create measurement engine
            engine = MeasurementEngine()
            
            # Set enabled categories
            enabled_categories = self.config.get('enabled_categories', [])
            if enabled_categories:
                engine.set_enabled_categories(enabled_categories)
            
            # Determine DNA channel
            dna_channel = self.config.get('dna_channel')
            if dna_channel is None:
                # Try to find DNA channel
                for ch_name in self.intensity_images.keys():
                    if 'dna' in ch_name.lower():
                        dna_channel = ch_name
                        break
            
            # Extract core measurements
            self.status.emit("Extracting measurements...")
            self.progress.emit(20)
            
            df = engine.extract_measurements(
                masks=self.masks,
                intensity_images=self.intensity_images,
                is_3d=self.config.get('is_3d', False),
                dna_channel=dna_channel,
                assign_phases=self.config.get('assign_phases', False)
            )
            
            self.progress.emit(60)
            
            # Execute plugins if any
            enabled_plugins = self.config.get('enabled_plugins', [])
            if enabled_plugins and self.plugin_loader:
                self.status.emit("Running custom plugins...")
                self.progress.emit(70)
                
                # Get plugin instances
                plugin_instances = []
                for plugin_file in enabled_plugins:
                    instance = self.plugin_loader.get_plugin_instance(plugin_file)
                    if instance:
                        plugin_instances.append(instance)
                
                if plugin_instances:
                    plugin_df = engine.execute_plugins(
                        masks=self.masks,
                        intensity_images=self.intensity_images,
                        plugins=plugin_instances
                    )
                    
                    # Merge with main measurements
                    if not plugin_df.empty:
                        df = df.merge(plugin_df, on='nucleus_id', how='left')
                
                self.progress.emit(90)
            
            self.status.emit("Measurements complete!")
            self.progress.emit(100)
            
            time.sleep(0.2)  # Brief pause so user sees 100%
            
            # Emit results
            self.finished.emit(df)
            
        except Exception as e:
            self.error.emit(str(e))
