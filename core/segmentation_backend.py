from typing import Protocol, Optional, Dict, Literal, Union, Any
import numpy as np
import importlib
import logging
import sys

# Configure logging
logger = logging.getLogger(__name__)

class SegmentationBackend(Protocol):
    """
    Abstract base class / Protocol for segmentation backends.
    """
    def segment(self, 
                volume: np.ndarray, 
                *, 
                structure_id: Optional[str] = None, 
                workflow_id: Optional[str] = None, 
                config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Segment the input volume.

        Args:
            volume: Input image volume. Expected to be 3D (Z, Y, X) or 4D (C, Z, Y, X) / (Z, C, Y, X).
                    If 4D, the specific channel to segment should be handled before or specified in config.
                    However, standard aicssegmentation workflows operate on single channel 3D images.
                    We will assume the input volume is already the correct single channel 3D array (Z, Y, X)
                    for the structure of interest, OR that the backend handles channel selection if config provided.
            structure_id: Identifier for the structure (e.g., "LAMP1", "DNA").
            workflow_id: Identifier for the specific workflow/model.
            config: Additional configuration parameters.

        Returns:
            Binary 3D mask with the same spatial dimensions as the input volume.
        """
        ...

def supports_segmenter_ml() -> bool:
    """
    Check if the environment supports ML segmentation (Torch + GPU).
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        # Optional: Check CUDA capability if needed
        return True
    except ImportError:
        return False

class ClassicSegmenterBackend:
    """
    Backend using aicssegmentation classic workflows.
    """
    def __init__(self):
        try:
            import aicssegmentation
            from aicssegmentation.structure_wrapper import seg_lamp1, seg_dna, seg_tomm20
            # We will discover more dynamically
        except ImportError:
            logger.error("aicssegmentation not installed.")
            raise ImportError("aicssegmentation is required for ClassicSegmenterBackend")

        self._registry = self._build_registry()

    def _build_registry(self) -> Dict[str, Any]:
        """
        Build a registry of available structure wrappers.
        """
        registry = {}
        # Dynamic discovery could be implemented here by inspecting aicssegmentation.structure_wrapper
        # For now, we map some common ones and allow dynamic import
        
        # Example mapping based on common structures
        # In a real implementation, we might iterate over pkgutil.iter_modules
        structure_map = {
            "LAMP1": "aicssegmentation.structure_wrapper.seg_lamp1",
            "DNA": "aicssegmentation.structure_wrapper.seg_dna",
            "TOMM20": "aicssegmentation.structure_wrapper.seg_tomm20",
            "LMNB1": "aicssegmentation.structure_wrapper.seg_lmnb1",
            # Add others as needed
        }
        return structure_map

    def _get_wrapper_function(self, structure_id: str):
        if structure_id not in self._registry:
             # Try to find it dynamically if not explicitly mapped
             module_name = f"aicssegmentation.structure_wrapper.seg_{structure_id.lower()}"
             try:
                 module = importlib.import_module(module_name)
                 return module.Workflow_Wrapper() # Usually the class is Workflow_Wrapper
             except ImportError:
                 raise ValueError(f"Structure ID '{structure_id}' not found in registry or dynamic lookup.")
        
        module_path = self._registry[structure_id]
        module = importlib.import_module(module_path)
        # The wrappers usually expose a class named Workflow_Wrapper
        if hasattr(module, "Workflow_Wrapper"):
            return module.Workflow_Wrapper()
        else:
            raise ValueError(f"Module {module_path} does not have Workflow_Wrapper")

    def _to_segmenter_array(self, volume: np.ndarray) -> np.ndarray:
        """
        Convert input array to (Z, Y, X) format expected by aicssegmentation.
        """
        # Handle dimensions. 
        # If 2D (Y, X), promote to (1, Y, X)
        if volume.ndim == 2:
            return volume[np.newaxis, ...]
        
        # If 3D (Z, Y, X), return as is
        if volume.ndim == 3:
            return volume
            
        # If 4D, we assume the caller has already selected the channel. 
        # If not, we raise error as we don't know which channel to pick without config.
        if volume.ndim == 4:
             raise ValueError("ClassicSegmenterBackend expects 2D or 3D input. Please select a channel before calling segment().")
             
        return volume

    def segment(self, 
                volume: np.ndarray, 
                *, 
                structure_id: Optional[str] = None, 
                workflow_id: Optional[str] = None, 
                config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        
        if not structure_id:
            raise ValueError("structure_id is required for ClassicSegmenterBackend")

        # Convert image
        img_3d = self._to_segmenter_array(volume)
        
        # Get wrapper
        wrapper = self._get_wrapper_function(structure_id)
        
        # Prepare parameters
        # wrapper.segment_image usually takes (image, config) or similar
        # We need to check the specific signature of the wrapper or use the generic workflow engine
        
        # The Workflow_Wrapper class in aicssegmentation usually has:
        # print_config(), get_default_config(), segment_image(image, config_dict)
        
        params = config.get("parameters", {}) if config else {}
        
        # If config is not provided, use defaults
        if not params:
            params = wrapper.get_default_config()
            
        # Run segmentation
        # The output of segment_image is usually the result (mask)
        result = wrapper.segment_image(img_3d, params)
        
        # Ensure binary mask
        mask = result > 0
        return mask.astype(np.uint8)


class MlSegmenterBackend:
    """
    Backend using segmenter_model_zoo / aicsmlsegment.
    """
    def __init__(self):
        if not supports_segmenter_ml():
            logger.warning("ML Segmentation requirements (Torch+CUDA) not met.")
            # We don't raise here to allow instantiation, but segment() will fail or warn
            
        try:
            import segmenter_model_zoo
            from segmenter_model_zoo.zoo import ModelZoo
            self.ModelZoo = ModelZoo
        except ImportError:
            raise ImportError("segmenter_model_zoo is required for MlSegmenterBackend")

    def segment(self, 
                volume: np.ndarray, 
                *, 
                structure_id: Optional[str] = None, 
                workflow_id: Optional[str] = None, 
                config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        
        if not supports_segmenter_ml():
            raise RuntimeError("ML segmentation requires CUDA-capable GPU and PyTorch.")

        if not workflow_id:
             # Try to infer from structure_id if possible, or raise
             raise ValueError("workflow_id (model name) is required for MlSegmenterBackend")

        # Load model
        # segmenter_model_zoo API usage:
        # model = ModelZoo.load_model(workflow_id)
        # result = model.predict(volume)
        
        try:
            model = self.ModelZoo.load_model(workflow_id)
        except Exception as e:
            raise ValueError(f"Failed to load model '{workflow_id}': {e}")

        # Preprocessing might be needed. The model zoo usually handles normalization if using the high level API
        # But we should check input shape.
        
        # Run prediction
        # The predict method usually handles tiling and 3D
        prediction = model.predict(volume)
        
        # Post-processing to binary mask if needed
        # Some models return probability maps
        if prediction.dtype == float or np.max(prediction) <= 1.0:
             threshold = config.get("threshold", 0.5) if config else 0.5
             mask = prediction > threshold
        else:
             mask = prediction > 0
             
        return mask.astype(np.uint8)

class Hybrid2D3DSegmenterBackend:
    """
    Backend using hybrid 2D segmentation + 3D linking.
    
    This backend combines robust 2D segmentation (e.g., Cellpose) with
    3D instance linking via overlap-based graph matching.
    """
    def __init__(self, base_2d_backend=None):
        self.base_2d_backend = base_2d_backend or ClassicSegmenterBackend()
    
    def segment(self,
               volume: np.ndarray,
               *,
               structure_id: Optional[str] = None,
               workflow_id: Optional[str] = None,
               config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Segment using hybrid 2D+3D approach.
        
        Args:
            volume: 3D volume (Z, Y, X)
            structure_id: Structure identifier
            workflow_id: Workflow identifier
            config: Configuration parameters
            
        Returns:
            3D labeled mask with instance consistency
        """
        from core.segmentation_3d import Hybrid2D3DBackend
        
        # Create 2D segmentation function that wraps the base backend
        def segment_2d_fn(slice_2d):
            return self.base_2d_backend.segment(
                slice_2d,
                structure_id=structure_id,
                workflow_id=workflow_id,
                config=config
            )
        
        # Get parameters from config
        min_overlap = config.get('min_overlap_ratio', 0.3) if config else 0.3
        max_distance_z = config.get('max_distance_z', 2) if config else 2
        
        # Create and run hybrid backend
        hybrid = Hybrid2D3DBackend(
            segmentation_2d_fn=segment_2d_fn,
            min_overlap_ratio=min_overlap,
            max_distance_z=max_distance_z
        )
        
        masks = hybrid.segment(volume)
        return masks.astype(np.uint8)


def get_segmenter_backend(mode: Literal["classic", "ml", "auto", "hybrid2d3d"], **kwargs) -> SegmentationBackend:
    if mode == "ml":
        return MlSegmenterBackend()
    elif mode == "classic":
        return ClassicSegmenterBackend()
    elif mode == "hybrid2d3d":
        return Hybrid2D3DSegmenterBackend()
    elif mode == "auto":
        if supports_segmenter_ml():
            try:
                return MlSegmenterBackend()
            except Exception:
                return ClassicSegmenterBackend()
        else:
            return ClassicSegmenterBackend()
    else:
        raise ValueError(f"Unknown mode: {mode}")
