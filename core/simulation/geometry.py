"""
Compartment geometry definitions for 3D simulations.

Supports:
- Nuclear envelope with typical geometry
- Nucleolus and other sub-compartments
- Boundary detection and reflection
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class NuclearEnvelope:
    """
    Nuclear envelope geometry (simplified as ellipsoid).
    
    Attributes:
        center: Center position (x, y, z) in µm
        radii: Semi-axes (a, b, c) in µm
        thickness: Membrane thickness in µm (typically 0.04 µm)
    """
    center: np.ndarray
    radii: np.ndarray  # (a, b, c)
    thickness: float = 0.04  # µm
    
    def is_inside(self, position: np.ndarray) -> bool:
        """Check if position is inside the envelope."""
        rel_pos = position - self.center
        normalized = np.sum((rel_pos / self.radii)**2)
        return normalized <= 1.0
    
    def distance_to_surface(self, position: np.ndarray) -> float:
        """Compute signed distance to nuclear envelope surface."""
        rel_pos = position - self.center
        normalized = np.sqrt(np.sum((rel_pos / self.radii)**2))
        
        # Approximate distance
        distance = (normalized - 1.0) * np.mean(self.radii)
        return distance
    
    def surface_normal(self, position: np.ndarray) -> np.ndarray:
        """Compute outward normal at closest surface point."""
        rel_pos = position - self.center
        # Gradient of ellipsoid equation
        gradient = 2 * rel_pos / (self.radii**2)
        normal = gradient / np.linalg.norm(gradient)
        return normal


class CompartmentGeometry:
    """
    Multi-compartment 3D geometry for nuclear simulations.
    
    Typical nucleus structure:
    - Outer boundary: nuclear envelope (~5-10 µm radius)
    - Inner compartments: nucleolus, speckles, etc.
    """
    
    def __init__(
        self,
        nuclear_envelope: Optional[NuclearEnvelope] = None,
        nucleolus_centers: Optional[list] = None,
        nucleolus_radii: Optional[list] = None
    ):
        """
        Initialize compartment geometry.
        
        Args:
            nuclear_envelope: Nuclear envelope geometry
            nucleolus_centers: List of nucleolus center positions
            nucleolus_radii: List of nucleolus radii
        """
        # Default typical mammalian nucleus if not specified
        if nuclear_envelope is None:
            nuclear_envelope = NuclearEnvelope(
                center=np.array([0.0, 0.0, 0.0]),
                radii=np.array([5.0, 5.0, 5.0])  # ~5 µm radius
            )
        
        self.nuclear_envelope = nuclear_envelope
        
        # Default nucleolus if not specified
        if nucleolus_centers is None:
            nucleolus_centers = [np.array([0.0, 0.0, 0.0])]
            nucleolus_radii = [1.5]  # ~1.5 µm radius
        
        self.nucleoli = []
        for center, radius in zip(nucleolus_centers, nucleolus_radii):
            self.nucleoli.append({
                'center': center,
                'radius': radius
            })
    
    def get_compartment(self, position: np.ndarray) -> str:
        """
        Determine which compartment a position is in.
        
        Returns:
            'outside', 'cytoplasm', 'nucleoplasm', 'nucleolus'
        """
        # Check if inside nucleus
        if not self.nuclear_envelope.is_inside(position):
            return 'cytoplasm'
        
        # Check if inside any nucleolus
        for nucleolus in self.nucleoli:
            dist = np.linalg.norm(position - nucleolus['center'])
            if dist <= nucleolus['radius']:
                return 'nucleolus'
        
        return 'nucleoplasm'
    
    def check_boundary_crossing(
        self,
        pos_old: np.ndarray,
        pos_new: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Check if trajectory crosses a compartment boundary.
        
        Args:
            pos_old: Previous position
            pos_new: New position
            
        Returns:
            crossed: True if boundary was crossed
            normal: Surface normal at crossing point (if crossed)
        """
        comp_old = self.get_compartment(pos_old)
        comp_new = self.get_compartment(pos_new)
        
        if comp_old == comp_new:
            return False, None
        
        # Determine which boundary was crossed
        # For simplicity, use nuclear envelope normal
        # More sophisticated version would determine exact crossing point
        normal = self.nuclear_envelope.surface_normal(pos_new)
        
        return True, normal
    
    def reflect_position(
        self,
        pos_old: np.ndarray,
        pos_new: np.ndarray,
        normal: np.ndarray
    ) -> np.ndarray:
        """
        Reflect position across boundary.
        
        Args:
            pos_old: Previous position
            pos_new: Proposed new position
            normal: Surface normal
            
        Returns:
            Reflected position
        """
        # Compute reflection using vector reflection formula
        displacement = pos_new - pos_old
        
        # Project displacement onto normal
        proj = np.dot(displacement, normal) * normal
        
        # Reflect the component perpendicular to boundary
        reflected_displacement = displacement - 2 * proj
        
        return pos_old + reflected_displacement
    
    def create_from_mask(self, mask: np.ndarray, voxel_size: Tuple[float, float, float]):
        """
        Create geometry from 3D segmentation mask.
        
        Args:
            mask: 3D boolean array (segmented nucleus)
            voxel_size: Physical voxel dimensions (dx, dy, dz) in µm
        """
        # Find center of mass
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return
        
        center_voxels = np.mean(coords, axis=0)
        center_physical = center_voxels * np.array(voxel_size)
        
        # Estimate radii by fitting ellipsoid
        centered_coords = coords - center_voxels
        
        # Compute principal components
        cov = np.cov(centered_coords.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Radii are proportional to sqrt of eigenvalues
        # Scale to approximate 95% of mass
        radii_voxels = 2 * np.sqrt(eigenvalues)
        radii_physical = radii_voxels * np.array(voxel_size)
        
        self.nuclear_envelope = NuclearEnvelope(
            center=center_physical,
            radii=radii_physical
        )
    
    def to_dict(self) -> dict:
        """Export geometry to dictionary."""
        return {
            'nuclear_envelope': {
                'center': self.nuclear_envelope.center.tolist(),
                'radii': self.nuclear_envelope.radii.tolist(),
                'thickness': self.nuclear_envelope.thickness
            },
            'nucleoli': [
                {
                    'center': nuc['center'].tolist(),
                    'radius': float(nuc['radius'])
                }
                for nuc in self.nucleoli
            ]
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Load geometry from dictionary."""
        envelope = NuclearEnvelope(
            center=np.array(data['nuclear_envelope']['center']),
            radii=np.array(data['nuclear_envelope']['radii']),
            thickness=data['nuclear_envelope']['thickness']
        )
        
        nucleolus_centers = [np.array(n['center']) for n in data['nucleoli']]
        nucleolus_radii = [n['radius'] for n in data['nucleoli']]
        
        return cls(envelope, nucleolus_centers, nucleolus_radii)
