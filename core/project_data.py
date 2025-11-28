"""
Project data management with adaptive JSON/SQLite storage
Automatically migrates to SQLite when project size exceeds 500MB
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict, field


@dataclass
class ImageData:
    """Data class for storing image information"""
    path: str
    filename: str
    added_date: str
    channels: List[str]
    dna_channel: Optional[int] = None
    group: str = "Default"  # Group/Treatment name for pooled analysis
    shape: tuple = None
    dtype: str = None
    pixel_size: Optional[float] = None
    bit_depth: int = 8
    
    # Segmentation data
    current_segmentation_id: Optional[int] = None
    segmentation_history: List[Dict] = field(default_factory=list)
    
    # QC data
    qc_flags: List[int] = field(default_factory=list)  # nucleus IDs flagged
    qc_confirmations: Dict[int, bool] = field(default_factory=dict)  # nucleus_id: is_error
    
    # Edit history
    edit_log: List[Dict] = field(default_factory=list)
    
    # Measurements
    measurements_df: Optional[pd.DataFrame] = None


@dataclass
class ProjectSettings:
    """Global project settings"""
    analysis_mode: str = "2D"  # "2D" or "3D"
    enabled_measurements: Dict[str, bool] = field(default_factory=dict)
    cell_cycle_enabled: bool = False
    cell_cycle_clusters: int = 3
    phase_boundaries: List[float] = field(default_factory=list)
    phase_labels: List[str] = field(default_factory=list)
    enabled_plugins: List[str] = field(default_factory=list)


class Project:
    """
    Main project class managing images, segmentations, and measurements
    Supports both JSON (lightweight) and SQLite (large projects) backends
    """
    
    def __init__(self, project_path: Optional[str] = None):
        self.project_path = project_path
        self.project_name = "Untitled Project"
        self.created_date = datetime.now().isoformat()
        self.modified_date = datetime.now().isoformat()
        
        self.images: List[ImageData] = []
        self.settings = ProjectSettings()
        self.export_templates: Dict[str, Dict] = {}
        
        # Storage backend
        self.storage_backend = "json"  # "json" or "sqlite"
        self.db_connection: Optional[sqlite3.Connection] = None
        
        # Auto-save settings
        self.auto_save_enabled = True
        self.auto_save_interval = 300  # seconds
        
        if project_path and Path(project_path).exists():
            self.load()
    
    def add_image(self, image_data: ImageData) -> int:
        """Add a new image to the project and return its index"""
        self.images.append(image_data)
        self.modified_date = datetime.now().isoformat()
        return len(self.images) - 1
    
    def get_image(self, index: int) -> Optional[ImageData]:
        """Get image data by index"""
        if 0 <= index < len(self.images):
            return self.images[index]
        return None
    
    def remove_image(self, index: int):
        """Remove image from project"""
        if 0 <= index < len(self.images):
            del self.images[index]
            self.modified_date = datetime.now().isoformat()
    
    def get_aggregated_measurements(self) -> pd.DataFrame:
        """Get measurements from all images combined into single DataFrame"""
        all_measurements = []
        
        for img_data in self.images:
            if img_data.measurements_df is not None:
                df = img_data.measurements_df.copy()
                df.insert(0, 'source_image', img_data.filename)
                all_measurements.append(df)
        
        if all_measurements:
            return pd.concat(all_measurements, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def estimate_size(self) -> int:
        """Estimate current project size in bytes"""
        size = 0
        
        # Estimate measurements size
        for img_data in self.images:
            if img_data.measurements_df is not None:
                # Rough estimate: DataFrame memory usage
                size += img_data.measurements_df.memory_usage(deep=True).sum()
        
        # Add metadata size (rough estimate)
        size += len(json.dumps(self._get_metadata_dict()).encode('utf-8'))
        
        return size
    
    def check_and_migrate_storage(self) -> bool:
        """
        Check project size and migrate from JSON to SQLite if > 500MB
        Returns True if migration occurred
        """
        current_size = self.estimate_size()
        size_mb = current_size / (1024 * 1024)
        
        if size_mb > 500 and self.storage_backend == "json":
            print(f"Project size ({size_mb:.1f} MB) exceeds 500 MB threshold.")
            print("Migrating to SQLite database for better performance...")
            self._migrate_to_sqlite()
            return True
        
        return False
    
    def _get_metadata_dict(self) -> Dict:
        """Get project metadata as dictionary (without large data)"""
        return {
            "project_name": self.project_name,
            "created_date": self.created_date,
            "modified_date": self.modified_date,
            "storage_backend": self.storage_backend,
            "settings": asdict(self.settings),
            "export_templates": self.export_templates,
            "images": [
                {
                    "path": img.path,
                    "filename": img.filename,
                    "added_date": img.added_date,
                    "channels": img.channels,
                    "dna_channel": img.dna_channel,
                    "shape": img.shape,
                    "dtype": img.dtype,
                    "pixel_size": img.pixel_size,
                    "bit_depth": img.bit_depth,
                    "current_segmentation_id": img.current_segmentation_id,
                    "segmentation_history": img.segmentation_history,
                    "qc_flags": img.qc_flags,
                    "qc_confirmations": img.qc_confirmations,
                    "edit_log": img.edit_log,
                }
                for img in self.images
            ]
        }
    
    def save(self, path: Optional[str] = None):
        """Save project to file"""
        if path:
            self.project_path = path
        
        if not self.project_path:
            raise ValueError("No project path specified")
        
        self.modified_date = datetime.now().isoformat()
        
        # Check if migration needed
        self.check_and_migrate_storage()
        
        if self.storage_backend == "json":
            self._save_json()
        else:
            self._save_sqlite()
    
    def _save_json(self):
        """Save project as JSON file"""
        project_dir = Path(self.project_path).parent
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = self._get_metadata_dict()
        
        with open(self.project_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save measurements as separate CSV files
        measurements_dir = project_dir / "measurements"
        measurements_dir.mkdir(exist_ok=True)
        
        for i, img in enumerate(self.images):
            if img.measurements_df is not None:
                csv_path = measurements_dir / f"image_{i}_{Path(img.filename).stem}.csv"
                img.measurements_df.to_csv(csv_path, index=False)
    
    def load(self, path: Optional[str] = None):
        """Load project from file"""
        if path:
            self.project_path = path
        
        if not self.project_path or not Path(self.project_path).exists():
            raise FileNotFoundError(f"Project file not found: {self.project_path}")
        
        # Detect storage backend
        if self.project_path.endswith('.db'):
            self.storage_backend = "sqlite"
            self._load_sqlite()
        else:
            self.storage_backend = "json"
            self._load_json()
    
    def _load_json(self):
        """Load project from JSON file"""
        with open(self.project_path, 'r') as f:
            data = json.load(f)
        
        self.project_name = data.get("project_name", "Untitled Project")
        self.created_date = data.get("created_date", datetime.now().isoformat())
        self.modified_date = data.get("modified_date", datetime.now().isoformat())
        self.storage_backend = data.get("storage_backend", "json")
        
        # Load settings
        settings_dict = data.get("settings", {})
        self.settings = ProjectSettings(**settings_dict)
        
        self.export_templates = data.get("export_templates", {})
        
        # Load images
        project_dir = Path(self.project_path).parent
        measurements_dir = project_dir / "measurements"
        
        self.images = []
        for i, img_dict in enumerate(data.get("images", [])):
            img_data = ImageData(**img_dict)
            
            # Load measurements if available
            csv_path = measurements_dir / f"image_{i}_{Path(img_dict['filename']).stem}.csv"
            if csv_path.exists():
                img_data.measurements_df = pd.read_csv(csv_path)
            
            self.images.append(img_data)
    
    def _migrate_to_sqlite(self):
        """Migrate project from JSON to SQLite"""
        if not self.project_path:
            raise ValueError("No project path set for migration")
        
        # Create SQLite database path
        db_path = Path(self.project_path).with_suffix('.db')
        
        # Initialize SQLite database
        self._init_sqlite_database(str(db_path))
        
        # Save current data to SQLite
        self.storage_backend = "sqlite"
        self.project_path = str(db_path)
        self._save_sqlite()
        
        print(f"Migration complete. Database saved to: {db_path}")
    
    def _init_sqlite_database(self, db_path: str):
        """Initialize SQLite database schema"""
        self.db_connection = sqlite3.connect(db_path)
        cursor = self.db_connection.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS project_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT,
                filename TEXT,
                added_date TEXT,
                channels TEXT,
                dna_channel INTEGER,
                shape TEXT,
                dtype TEXT,
                pixel_size REAL,
                bit_depth INTEGER,
                current_segmentation_id INTEGER
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS segmentations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER,
                timestamp TEXT,
                model_name TEXT,
                parameters TEXT,
                nucleus_count INTEGER,
                median_area REAL,
                cv_area REAL,
                processing_time REAL,
                FOREIGN KEY (image_id) REFERENCES images(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER,
                nucleus_id INTEGER,
                measurements TEXT,
                FOREIGN KEY (image_id) REFERENCES images(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS qc_data (
                image_id INTEGER,
                nucleus_id INTEGER,
                is_flagged BOOLEAN,
                is_confirmed_error BOOLEAN,
                PRIMARY KEY (image_id, nucleus_id),
                FOREIGN KEY (image_id) REFERENCES images(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER,
                timestamp TEXT,
                operation TEXT,
                affected_nuclei TEXT,
                parameters TEXT,
                FOREIGN KEY (image_id) REFERENCES images(id)
            )
        """)
        
        self.db_connection.commit()
    
    def _save_sqlite(self):
        """Save project to SQLite database"""
        if not self.db_connection:
            self._init_sqlite_database(self.project_path)
        
        cursor = self.db_connection.cursor()
        
        # Save metadata
        metadata = self._get_metadata_dict()
        for key in ["project_name", "created_date", "modified_date", "storage_backend"]:
            cursor.execute(
                "INSERT OR REPLACE INTO project_metadata (key, value) VALUES (?, ?)",
                (key, str(metadata[key]))
            )
        
        cursor.execute(
            "INSERT OR REPLACE INTO project_metadata (key, value) VALUES (?, ?)",
            ("settings", json.dumps(metadata["settings"]))
        )
        
        cursor.execute(
            "INSERT OR REPLACE INTO project_metadata (key, value) VALUES (?, ?)",
            ("export_templates", json.dumps(metadata["export_templates"]))
        )
        
        # Clear and save images
        cursor.execute("DELETE FROM images")
        cursor.execute("DELETE FROM measurements")
        cursor.execute("DELETE FROM qc_data")
        
        for img in self.images:
            cursor.execute("""
                INSERT INTO images (path, filename, added_date, channels, dna_channel,
                                  shape, dtype, pixel_size, bit_depth, current_segmentation_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                img.path, img.filename, img.added_date, json.dumps(img.channels),
                img.dna_channel, json.dumps(img.shape), img.dtype, img.pixel_size,
                img.bit_depth, img.current_segmentation_id
            ))
            
            image_id = cursor.lastrowid
            
            # Save measurements
            if img.measurements_df is not None:
                for _, row in img.measurements_df.iterrows():
                    cursor.execute("""
                        INSERT INTO measurements (image_id, nucleus_id, measurements)
                        VALUES (?, ?, ?)
                    """, (image_id, row.get('nucleus_id', 0), json.dumps(row.to_dict())))
            
            # Save QC data
            for nucleus_id in img.qc_flags:
                is_confirmed = img.qc_confirmations.get(nucleus_id, False)
                cursor.execute("""
                    INSERT INTO qc_data (image_id, nucleus_id, is_flagged, is_confirmed_error)
                    VALUES (?, ?, ?, ?)
                """, (image_id, nucleus_id, True, is_confirmed))
        
        self.db_connection.commit()
    
    def _load_sqlite(self):
        """Load project from SQLite database"""
        self.db_connection = sqlite3.connect(self.project_path)
        cursor = self.db_connection.cursor()
        
        # Load metadata
        cursor.execute("SELECT key, value FROM project_metadata")
        metadata = dict(cursor.fetchall())
        
        self.project_name = metadata.get("project_name", "Untitled Project")
        self.created_date = metadata.get("created_date", datetime.now().isoformat())
        self.modified_date = metadata.get("modified_date", datetime.now().isoformat())
        
        if "settings" in metadata:
            self.settings = ProjectSettings(**json.loads(metadata["settings"]))
        
        if "export_templates" in metadata:
            self.export_templates = json.loads(metadata["export_templates"])
        
        # Load images
        cursor.execute("SELECT * FROM images")
        self.images = []
        
        for row in cursor.fetchall():
            img_data = ImageData(
                path=row[1],
                filename=row[2],
                added_date=row[3],
                channels=json.loads(row[4]),
                dna_channel=row[5],
                shape=json.loads(row[6]) if row[6] else None,
                dtype=row[7],
                pixel_size=row[8],
                bit_depth=row[9],
                current_segmentation_id=row[10]
            )
            
            image_id = row[0]
            
            # Load measurements
            cursor.execute("SELECT measurements FROM measurements WHERE image_id = ?", (image_id,))
            measurements = cursor.fetchall()
            if measurements:
                rows_data = [json.loads(m[0]) for m in measurements]
                img_data.measurements_df = pd.DataFrame(rows_data)
            
            # Load QC data
            cursor.execute("SELECT nucleus_id, is_confirmed_error FROM qc_data WHERE image_id = ?", (image_id,))
            qc_data = cursor.fetchall()
            for nucleus_id, is_confirmed in qc_data:
                img_data.qc_flags.append(nucleus_id)
                img_data.qc_confirmations[nucleus_id] = bool(is_confirmed)
            
            self.images.append(img_data)
    
    def close(self):
        """Close database connection if using SQLite"""
        if self.db_connection:
            self.db_connection.close()
            self.db_connection = None
