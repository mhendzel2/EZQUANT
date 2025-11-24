"""
Plugin loader for discovering and managing measurement plugins
"""

import os
import sys
import importlib.util
from typing import List, Dict, Optional, Type
from pathlib import Path
import warnings


class PluginLoader:
    """
    Dynamically discover and load measurement plugins
    """
    
    def __init__(self, plugin_directory: Optional[str] = None):
        """
        Initialize plugin loader
        
        Args:
            plugin_directory: Path to directory containing plugins.
                            Defaults to 'plugins' in project root.
        """
        if plugin_directory is None:
            # Use plugins directory relative to this file
            project_root = Path(__file__).parent.parent
            self.plugin_directory = project_root / 'plugins'
        else:
            self.plugin_directory = Path(plugin_directory)
        
        self.loaded_plugins: Dict[str, Type] = {}
        self.plugin_instances: Dict[str, object] = {}
        self.plugin_errors: Dict[str, str] = {}
    
    def discover_plugins(self) -> List[str]:
        """
        Scan plugin directory for Python files
        
        Returns:
            List of discovered plugin file paths
        """
        plugin_files = []
        
        if not self.plugin_directory.exists():
            warnings.warn(f"Plugin directory not found: {self.plugin_directory}")
            return plugin_files
        
        # Search for .py files in plugins directory and subdirectories
        for path in self.plugin_directory.rglob('*.py'):
            # Skip __init__.py, plugin_template.py, and private files
            if path.name.startswith('_') or path.name == 'plugin_template.py':
                continue
            
            plugin_files.append(str(path))
        
        return plugin_files
    
    def load_plugin(self, plugin_path: str) -> Optional[Type]:
        """
        Load a plugin from file path
        
        Args:
            plugin_path: Path to plugin .py file
        
        Returns:
            Plugin class if successful, None otherwise
        """
        try:
            # Get module name from file path
            path = Path(plugin_path)
            module_name = f"plugin_{path.stem}"
            
            # Load module from file
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load spec for {plugin_path}")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Find MeasurementPlugin subclass
            from plugins.plugin_template import MeasurementPlugin
            
            plugin_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                
                # Check if it's a class and subclass of MeasurementPlugin
                if (isinstance(attr, type) and 
                    issubclass(attr, MeasurementPlugin) and 
                    attr is not MeasurementPlugin):
                    plugin_class = attr
                    break
            
            if plugin_class is None:
                self.plugin_errors[path.name] = "No MeasurementPlugin subclass found"
                return None
            
            # Store loaded plugin
            self.loaded_plugins[path.name] = plugin_class
            
            # Create instance
            self.plugin_instances[path.name] = plugin_class()
            
            return plugin_class
        
        except Exception as e:
            self.plugin_errors[plugin_path] = str(e)
            warnings.warn(f"Failed to load plugin {plugin_path}: {e}")
            return None
    
    def load_all_plugins(self) -> int:
        """
        Discover and load all plugins
        
        Returns:
            Number of successfully loaded plugins
        """
        plugin_files = self.discover_plugins()
        
        loaded_count = 0
        for plugin_file in plugin_files:
            if self.load_plugin(plugin_file) is not None:
                loaded_count += 1
        
        return loaded_count
    
    def get_plugin_instance(self, plugin_name: str) -> Optional[object]:
        """
        Get instance of loaded plugin by name
        
        Args:
            plugin_name: Name of plugin file (e.g., 'texture_analysis.py')
        
        Returns:
            Plugin instance or None if not found
        """
        return self.plugin_instances.get(plugin_name)
    
    def get_all_plugin_instances(self) -> List[object]:
        """
        Get list of all loaded plugin instances
        
        Returns:
            List of plugin instances
        """
        return list(self.plugin_instances.values())
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, str]]:
        """
        Get information about a plugin
        
        Args:
            plugin_name: Name of plugin file
        
        Returns:
            Dict with plugin info (name, description, version) or None
        """
        instance = self.plugin_instances.get(plugin_name)
        
        if instance is None:
            return None
        
        return {
            'name': instance.get_name(),
            'description': instance.get_description(),
            'version': getattr(instance, 'version', '1.0.0'),
            'file': plugin_name,
            'enabled': True
        }
    
    def get_all_plugin_info(self) -> List[Dict[str, str]]:
        """
        Get information about all loaded plugins
        
        Returns:
            List of plugin info dicts
        """
        info_list = []
        
        for plugin_name in self.plugin_instances.keys():
            info = self.get_plugin_info(plugin_name)
            if info:
                info_list.append(info)
        
        return info_list
    
    def get_errors(self) -> Dict[str, str]:
        """
        Get dictionary of plugin loading errors
        
        Returns:
            Dict mapping plugin name to error message
        """
        return self.plugin_errors.copy()
    
    def reload_plugins(self) -> int:
        """
        Reload all plugins (useful for development)
        
        Returns:
            Number of successfully reloaded plugins
        """
        # Clear existing plugins
        self.loaded_plugins.clear()
        self.plugin_instances.clear()
        self.plugin_errors.clear()
        
        # Remove from sys.modules
        to_remove = [key for key in sys.modules.keys() if key.startswith('plugin_')]
        for key in to_remove:
            del sys.modules[key]
        
        # Reload all
        return self.load_all_plugins()
    
    def execute_plugins(self,
                       region,
                       intensity_images: Optional[Dict],
                       metadata: Optional[Dict] = None,
                       enabled_plugins: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Execute specified plugins on a region
        
        Args:
            region: skimage regionprops region
            intensity_images: Dict of channel_name -> intensity image
            metadata: Optional metadata dict
            enabled_plugins: List of plugin names to execute (None = all)
        
        Returns:
            Dict mapping plugin name to measurement results
        """
        results = {}
        
        # Determine which plugins to execute
        if enabled_plugins is None:
            plugins_to_run = self.plugin_instances.items()
        else:
            plugins_to_run = [(name, inst) for name, inst in self.plugin_instances.items()
                            if name in enabled_plugins]
        
        # Execute each plugin
        for plugin_name, plugin_instance in plugins_to_run:
            try:
                measurements = plugin_instance.measure(region, intensity_images, metadata)
                results[plugin_name] = measurements
            except Exception as e:
                warnings.warn(f"Plugin {plugin_name} execution failed: {e}")
                results[plugin_name] = {'error': str(e)}
        
        return results


class PluginManagerDialog:
    """
    Dialog for managing plugins (placeholder - would be QDialog in full implementation)
    """
    
    def __init__(self, plugin_loader: PluginLoader):
        self.plugin_loader = plugin_loader
    
    def show(self):
        """Show plugin manager dialog"""
        # This would create and show a QDialog with:
        # - Table of plugins (name, description, version, enabled checkbox)
        # - Reload button
        # - Open plugin folder button
        # - Error display for failed plugins
        pass
