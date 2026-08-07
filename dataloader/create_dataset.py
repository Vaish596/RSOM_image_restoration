"""
Photoacoustic Image Restoration Dataset Creation Pipeline

This pipeline creates paired HQ/LQ datasets for deep learning restoration models.
- HQ: Reconstruction from FULL raw acquisition data
- LQ: Reconstruction from UNDERSAMPLED raw acquisition data

The undersampling is applied to the RAW data BEFORE reconstruction, which is
critical for realistic artifact modeling in photoacoustic imaging.

Key domain-specific considerations:
- Photoacoustic signals are strongest near the center of the scan
- Side regions have weak/empty signals and should be cropped
- Multiple undersampling patterns and ratios are generated per volume
- Train/val/test split is done at VOLUME level to prevent leakage
"""

import hashlib
import os
import json
import numpy as np
import hdf5storage as h5
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from time import time
import shutil

# Import your reconstruction functions
from rsom_reconstruction import (
    SensitivityField, 
    saft_matfile_adapter, 
    recon2rgb, 
    write_to_matfile
)


# =============================================================================
# MASK GENERATION FUNCTIONS
# =============================================================================

def generate_random_mask(n_positions: int, sampling_ratio: float, seed: int) -> np.ndarray:
    """
    Generate random undersampling mask.
    
    Args:
        n_positions: Total number of scan positions in the raw data
        sampling_ratio: Fraction of positions to keep (e.g., 0.3 = 30%)
        seed: Random seed for reproducibility
    
    Returns:
        Boolean array of shape (n_positions,) indicating which positions to keep
    
    Note: This is the most common undersampling pattern in photoacoustic imaging
    where acquisition time is limited.
    """
    np.random.seed(seed)
    mask = np.random.binomial(1, sampling_ratio, size=n_positions) == 1
    
    # Ensure at least some positions are selected
    if mask.sum() < 5:
        # If too few selected, randomly select at least 5
        indices = np.random.choice(n_positions, size=max(5, int(n_positions * sampling_ratio)), replace=False)
        mask = np.zeros(n_positions, dtype=bool)
        mask[indices] = True
    
    return mask


def generate_semistructured_mask(n_positions: int, sampling_ratio: float, seed: int) -> np.ndarray:
    """
    Generate semi-structured undersampling mask.
    
    This creates a more regular pattern by selecting every Nth position,
    then adding some random variation. This mimics systematic undersampling
    patterns that might occur in real acquisition protocols.
    
    Args:
        n_positions: Total number of scan positions
        sampling_ratio: Target fraction of positions to keep
        seed: Random seed for reproducibility
    
    Returns:
        Boolean array indicating which positions to keep
    """
    np.random.seed(seed)
    
    # Calculate stride for regular sampling
    stride = max(1, int(1.0 / sampling_ratio))
    
    # Start with regular sampling
    mask = np.zeros(n_positions, dtype=bool)
    mask[::stride] = True
    
    # Add some randomness: randomly remove 20% of selected positions
    selected_indices = np.where(mask)[0]
    n_to_remove = int(len(selected_indices) * 0.2)
    if n_to_remove > 0:
        remove_indices = np.random.choice(selected_indices, size=n_to_remove, replace=False)
        mask[remove_indices] = False
    
    # Randomly add some positions to reach target ratio
    current_ratio = mask.sum() / n_positions
    if current_ratio < sampling_ratio:
        n_to_add = int((sampling_ratio - current_ratio) * n_positions)
        unselected_indices = np.where(~mask)[0]
        if len(unselected_indices) > 0 and n_to_add > 0:
            add_indices = np.random.choice(unselected_indices, size=min(n_to_add, len(unselected_indices)), replace=False)
            mask[add_indices] = True
    
    return mask


def generate_variable_density_mask(n_positions: int, sampling_ratio: float, seed: int, 
                                   center_ratio: float = 0.8) -> np.ndarray:
    """
    Generate variable-density undersampling mask.
    
    This creates higher sampling density in the center and lower density at the edges.
    This is motivated by the fact that photoacoustic signals are typically stronger
    in the center of the scan region.
    
    Args:
        n_positions: Total number of scan positions
        sampling_ratio: Overall target fraction of positions to keep
        seed: Random seed for reproducibility
        center_ratio: Fraction of positions considered "center" (default: 80% center, 10% each side)
    
    Returns:
        Boolean array indicating which positions to keep
    """
    np.random.seed(seed)
    
    # Define center and edge regions
    center_start = int(n_positions * (1 - center_ratio) / 2)
    center_end = int(n_positions * (1 + center_ratio) / 2)
    
    # Sample more densely in center
    center_sampling_ratio = min(1.0, sampling_ratio * 1.5)
    edge_sampling_ratio = max(0.1, sampling_ratio * 0.5)
    
    mask = np.zeros(n_positions, dtype=bool)
    
    # Center region
    center_mask = np.random.binomial(1, center_sampling_ratio, size=center_end - center_start) == 1
    mask[center_start:center_end] = center_mask
    
    # Left edge
    left_mask = np.random.binomial(1, edge_sampling_ratio, size=center_start) == 1
    mask[:center_start] = left_mask
    
    # Right edge
    right_mask = np.random.binomial(1, edge_sampling_ratio, size=n_positions - center_end) == 1
    mask[center_end:] = right_mask
    
    return mask


def get_mask_generator(mask_type: str):
    """Return the appropriate mask generation function."""
    mask_generators = {
        'random': generate_random_mask,
        'semistructured': generate_semistructured_mask,
        'variabledensity': generate_variable_density_mask,
    }
    return mask_generators[mask_type]


# =============================================================================
# RECONSTRUCTION WRAPPER
# =============================================================================

def reconstruct_from_raw(mat_data: Dict, sensitivity: SensitivityField, 
                         temp_mat_path: str, verbose: bool = False) -> np.ndarray:
    """
    Wrapper for reconstruction that handles temporary file creation.
    
    Args:
        mat_data: Dictionary containing 'S' and 'positionXY' arrays
        sensitivity: SensitivityField object for reconstruction
        temp_mat_path: Path to save temporary .mat file
        verbose: Whether to print reconstruction progress
    
    Returns:
        Reconstructed volume as numpy array (before RGB conversion)
    
    Note: Your original reconstruction requires .mat file input, so we need
    to save the (potentially masked) data to a temporary file first.
    """
    # Save data to temporary .mat file
    h5.savemat(temp_mat_path, mat_data)
    
    # Perform reconstruction using your original code
    recon = saft_matfile_adapter(temp_mat_path, sensitivity, verbose=verbose)
    
    return recon


# =============================================================================
# IMAGE PROCESSING FUNCTIONS
# =============================================================================

def create_mip(volume: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Create Maximum Intensity Projection (MIP) along specified axis.
    
    Args:
        volume: 4D array (x, y, z, channels) or 3D array (x, y, z)
        axis: Axis along which to compute MIP (default: 1 = Y-axis)
    
    Returns:
        MIP image with one dimension removed
    
    Note: Axis 1 (Y-axis) is commonly used for cross-sectional views
    in photoacoustic imaging.
    """
    return volume.max(axis=axis)


def crop_center_region(image: np.ndarray, crop_fraction: float = 0.25) -> np.ndarray:
    """
    Crop image to remove side regions with weak signals.
    
    Args:
        image: 2D or 3D image array
        crop_fraction: Fraction to crop from EACH side (left and right)
    
    Returns:
        Cropped image
    
    Note: In photoacoustic imaging, the strongest signals are typically in the
    center of the scan. Side regions often contain weak or empty signals.
    
    Example: If image width is 1245 and crop_fraction=0.25:
        - Remove 311 pixels from left (1245 * 0.25)
        - Remove 311 pixels from right
        - Keep center 623 pixels
    """
    width = image.shape[1]

    left_crop = int(width * crop_fraction)
    right_crop = int(width * (1 - crop_fraction))

    cropped = image[:, left_crop:right_crop]

    return cropped
    # if image.ndim < 2:
    #     raise ValueError("Image must be at least 2D")
    
    # # Crop along the last spatial dimension (typically Z-axis in photoacoustic data)
    # width = image.shape[1]
    # crop_pixels = int(width * crop_fraction)
    
    # if image.ndim == 2:
    #     cropped = image[:, crop_pixels:-crop_pixels]
    # elif image.ndim == 3:
    #     cropped = image[:, :, crop_pixels:-crop_pixels]
    # else:
    #     raise ValueError(f"Unexpected image dimension: {image.ndim}")
    
    # return cropped


def normalize_percentile(ref_image: np.ndarray,target_image: np.ndarray, lower_percentile: float = 1.0, 
                         upper_percentile: float = 99.0) -> np.ndarray:
    """
    Normalize image using percentile clipping.
    
    This is preferred over min-max normalization for medical imaging because
    it's robust to outliers and preserves the relative intensity relationships.
    
    Args:
        ref_image: Reference image array for determining percentile values
        target_image: Image to be normalized
        lower_percentile: Lower percentile for clipping (default: 1%)
        upper_percentile: Upper percentile for clipping (default: 99%)
    
    Returns:
        Normalized image in range [0, 1]
    """
    p_low = np.percentile(ref_image, lower_percentile)
    p_high = np.percentile(ref_image, upper_percentile)
    
    # Clip and normalize
    clipped = np.clip(target_image, p_low, p_high)
    normalized = (clipped - p_low) / (p_high - p_low + 1e-8)
    
    return normalized.astype(np.float32)


# =============================================================================
# DATASET SPLITTING
# =============================================================================

def create_volume_splits(volume_paths: List[str], train_ratio: float = 0.7, 
                        val_ratio: float = 0.15, seed: int = 42) -> Dict[str, List[str]]:
    """
    Split volumes into train/val/test sets.
    
    CRITICAL: Splitting is done at VOLUME level, not at sample level.
    This prevents data leakage, since multiple samples (different masks/ratios)
    are generated from each volume.
    
    Args:
        volume_paths: List of paths to all .mat files
        train_ratio: Fraction of volumes for training
        val_ratio: Fraction of volumes for validation
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with keys 'train', 'val', 'test', each containing list of paths
    """
    np.random.seed(seed)
    
    # Shuffle volumes
    volume_paths = list(volume_paths)
    np.random.shuffle(volume_paths)
    
    n_volumes = len(volume_paths)
    n_train = int(round(n_volumes * train_ratio))
    n_val = int(round(n_volumes * val_ratio))

    splits = {
        'train': volume_paths[:n_train],
        'val': volume_paths[n_train:n_train + n_val],
        'test': volume_paths[n_train + n_val:]
    }
    
    print(f"Dataset split: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    
    return splits

def match_shapes(img1: np.ndarray, img2: np.ndarray):
    """
    Crop both images to smallest common spatial size.
    """
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])

    def crop_center(img, target_h, target_w):
        start_h = (img.shape[0] - target_h) // 2
        start_w = (img.shape[1] - target_w) // 2

        return img[
            start_h:start_h + target_h,
            start_w:start_w + target_w
        ]

    img1 = crop_center(img1, h, w)
    img2 = crop_center(img2, h, w)

    return img1, img2
    
# =============================================================================
# MAIN PIPELINE
# =============================================================================

class PhotoacousticDatasetPipeline:
    """
    Main pipeline for creating photoacoustic restoration dataset.
    """
    
    def __init__(self, 
                 output_dir: str,
                 mask_types: List[str] = None,
                 sampling_ratios: List[float] = None,
                 n_masks_per_ratio: int = 3,
                 crop_fraction: float = 0.25,
                 mip_axis: int = 1,
                 temp_dir: str = '/tmp/pa_reconstruction'):
        """
        Initialize pipeline.
        
        Args:
            output_dir: Root directory for dataset output
            mask_types: List of mask types to generate (default: all types)
            sampling_ratios: List of sampling ratios (default: [0.3, 0.4, 0.5])
            n_masks_per_ratio: Number of different masks per ratio (default: 3)
            crop_fraction: Fraction to crop from each side (default: 0.25)
            mip_axis: Axis for MIP projection (default: 1)
            temp_dir: Directory for temporary files during reconstruction
        """
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        
        # Default mask types and ratios
        self.mask_types = mask_types or ['random', 'semistructured', 'variabledensity']
        self.sampling_ratios = sampling_ratios or [0.3, 0.4, 0.5]
        self.n_masks_per_ratio = n_masks_per_ratio
        
        # Processing parameters
        self.crop_fraction = crop_fraction
        self.mip_axis = mip_axis
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize sensitivity field (used for all reconstructions)
        self.sensitivity = SensitivityField()
        
        print(f"Pipeline initialized:")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Mask types: {self.mask_types}")
        print(f"  Sampling ratios: {self.sampling_ratios}")
        print(f"  Masks per ratio: {self.n_masks_per_ratio}")
    
    
    def process_volume(self, mat_path: str, volume_id: str, split: str, 
                       save_visualizations: bool = False) -> int:
        """
        Process a single volume: create HQ and multiple LQ samples.
        
        Args:
            mat_path: Path to .mat file containing raw data
            volume_id: Unique identifier for this volume (e.g., 'volume001')
            split: Dataset split ('train', 'val', or 'test')
            save_visualizations: Whether to save PNG visualizations
        
        Returns:
            Number of samples created from this volume
        """
        print(f"\nProcessing {volume_id} ({split})...")
        start_time = time()
        
        # Load raw data
        print("  Loading raw data...")
        raw_data = h5.loadmat(mat_path)
        
        # Verify required fields exist
        if 'S' not in raw_data or 'positionXY' not in raw_data:
            raise ValueError(f"Raw data missing required fields. Found keys: {raw_data.keys()}")
        
        n_positions = raw_data['S'].shape[0]
        print(f"  Raw data shape: S={raw_data['S'].shape}, positionXY={raw_data['positionXY'].shape}")
        
        # =================================================================
        # STEP 1: HQ Reconstruction (full data)
        # =================================================================
        print("  Reconstructing HQ (full data)...")
        temp_hq_path = str(self.temp_dir / f"{volume_id}_hq.mat")
        
        # Reconstruct from full raw data
        recon_hq = reconstruct_from_raw(raw_data, self.sensitivity, temp_hq_path, verbose=False)
        
        # Convert to RGB if needed (your original code does this)
        recon_hq_rgb = recon2rgb(recon_hq).get()  # .get() converts CuPy to NumPy if needed
        print(f"  HQ reconstruction shape: {recon_hq_rgb.shape}")
        
        # Create MIP
        mip_hq = create_mip(recon_hq_rgb, axis=self.mip_axis)
        print("mip_hq shape:", mip_hq.shape)
        print("mip_hq size:", mip_hq.size)
        # Crop center region
        mip_hq_cropped = crop_center_region(mip_hq, self.crop_fraction)
        print("mip_hq_cropped shape:", mip_hq_cropped.shape)
        print("mip_hq_cropped size:", mip_hq_cropped.size)
        # Normalize
        mip_hq_normalized = normalize_percentile(mip_hq_cropped, mip_hq_cropped)  # Use HQ itself for percentile calculation

        
        
        print(f"  HQ MIP shape after processing: {mip_hq_normalized.shape}")
        
        # =================================================================
        # STEP 2: Generate multiple LQ samples with different masks
        # =================================================================
        sample_count = 0
        
        for mask_type in self.mask_types:
            for ratio in self.sampling_ratios:
                for mask_idx in range(self.n_masks_per_ratio):
                    
                    # Generate unique seed for this sample
                    seed_str = f"{volume_id}_{mask_type}_{ratio}_{mask_idx}"
                    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
                    
                    # Create sample name
                    sample_name = f"{volume_id}_{mask_type}_ratio{int(ratio*100)}_mask{mask_idx:02d}"
                    
                    print(f"    Creating sample: {sample_name}")
                    
                    # =====================================================
                    # CRITICAL: Generate mask and apply to RAW data
                    # =====================================================
                    mask_generator = get_mask_generator(mask_type)
                    mask = mask_generator(n_positions, ratio, seed)
                    
                    actual_ratio = mask.sum() / n_positions
                    print(f"      Mask: {mask.sum()}/{n_positions} positions ({actual_ratio:.2%})")
                    
                    # Apply mask to raw acquisition data BEFORE reconstruction
                    masked_data = {
                        'S': raw_data['S'][mask],
                        'positionXY': raw_data['positionXY'][mask]
                    }
                    
                    # Copy other fields if they exist
                    for key in raw_data.keys():
                        if key not in ['S', 'positionXY']:
                            masked_data[key] = raw_data[key]
                    
                    # =====================================================
                    # Reconstruct LQ from undersampled raw data
                    # =====================================================
                    temp_lq_path = str(self.temp_dir / f"{sample_name}_lq.mat")
                    recon_lq = reconstruct_from_raw(masked_data, self.sensitivity, temp_lq_path, verbose=False)
                    
                    # Convert to RGB
                    recon_lq_rgb = recon2rgb(recon_lq).get()
                    
                    # Create MIP
                    mip_lq = create_mip(recon_lq_rgb, axis=self.mip_axis)
                    
                    # Crop center region (SAME crop as HQ)
                    mip_lq_cropped = crop_center_region(mip_lq, self.crop_fraction)
                    
                    # Normalize
                    mip_lq_normalized = normalize_percentile(mip_hq_cropped, mip_lq_cropped)

                    # Ensure identical shape with HQ
                    mip_hq_aligned, mip_lq_aligned = match_shapes(
                        mip_hq_normalized,
                        mip_lq_normalized
                    )
                    
                    # =====================================================
                    # Save sample
                    # =====================================================
                    sample_dir = self.output_dir / split / sample_name
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save arrays
                    np.save(sample_dir / 'HQ.npy', mip_hq_aligned)
                    np.save(sample_dir / 'LQ.npy', mip_lq_aligned)
                    np.save(sample_dir / 'mask.npy', mask)
                    
                    # Save metadata
                    metadata = {
                        'volume_id': volume_id,
                        'sample_name': sample_name,
                        'split': split,
                        'mask_type': mask_type,
                        'target_sampling_ratio': ratio,
                        'actual_sampling_ratio': float(actual_ratio),
                        'n_positions_total': int(n_positions),
                        'n_positions_sampled': int(mask.sum()),
                        'original_mat_path': mat_path,
                        'hq_shape': list(mip_hq_aligned.shape),
                        'lq_shape': list(mip_lq_aligned.shape),
                        'crop_fraction': self.crop_fraction,
                        'mip_axis': self.mip_axis,
                        'seed': int(seed)
                    }
                    
                    with open(sample_dir / 'metadata.json', 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    # Optional: Save visualizations
                    sample_count += 1
                    if save_visualizations:
                        try:
                            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                            axes[0].imshow(mip_hq_aligned, cmap='hot')
                            axes[0].set_title('HQ')
                            axes[0].axis('off')
                            
                            axes[1].imshow(mip_lq_aligned, cmap='hot')
                            axes[1].set_title(f'LQ ({actual_ratio:.1%})')
                            axes[1].axis('off')
                            
                            axes[2].imshow(mip_hq_aligned - mip_lq_aligned, cmap='seismic')
                            axes[2].set_title('Difference')
                            axes[2].axis('off')
                            
                            plt.tight_layout()
                            plt.savefig(sample_dir / 'comparison.png', dpi=150, bbox_inches='tight')
                            plt.close()
                    
                        except Exception as e:
                            print(f"      WARNING: Failed to save visualization for {sample_name}: {e}")
                    
                    # Clean up temporary files
                    if os.path.exists(temp_lq_path):
                        os.remove(temp_lq_path)
        
        # Clean up HQ temporary file
        if os.path.exists(temp_hq_path):
            os.remove(temp_hq_path)
        
        elapsed = time() - start_time
        print(f"  Completed {volume_id}: {sample_count} samples created in {elapsed:.1f}s")
        
        return sample_count
    
    def create_dataset(self, volume_paths: List[str], 
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      save_visualizations: bool = False):
        """
        Create complete dataset from list of volume paths.
        
        Args:
            volume_paths: List of paths to .mat files
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set
            save_visualizations: Whether to save comparison images
        """
        print("="*80)
        print("PHOTOACOUSTIC DATASET CREATION PIPELINE")
        print("="*80)
        
        # Split volumes into train/val/test
        splits = create_volume_splits(volume_paths, train_ratio, val_ratio)
        
        # Process each split
        total_samples = 0
        split_stats = {}
        
        for split_name, split_paths in splits.items():
            print(f"\n{'='*80}")
            print(f"Processing {split_name.upper()} split ({len(split_paths)} volumes)")
            print(f"{'='*80}")
            
            split_samples = 0
            
            for idx, mat_path in enumerate(split_paths, 1):
                # Generate volume ID from filename
                volume_name = Path(mat_path).stem
                volume_id = f"vol{idx:03d}"
                
                n_samples = self.process_volume(
                    mat_path, 
                    volume_id, 
                    split_name,
                    save_visualizations=save_visualizations
                )
                
                split_samples += n_samples
            
            split_stats[split_name] = split_samples
            total_samples += split_samples
        
        # Save dataset summary
        summary = {
            'total_volumes': len(volume_paths),
            'total_samples': total_samples,
            'splits': {
                'train': {'n_volumes': len(splits['train']), 'n_samples': split_stats['train']},
                'val': {'n_volumes': len(splits['val']), 'n_samples': split_stats['val']},
                'test': {'n_volumes': len(splits['test']), 'n_samples': split_stats['test']}
            },
            'mask_types': self.mask_types,
            'sampling_ratios': self.sampling_ratios,
            'n_masks_per_ratio': self.n_masks_per_ratio,
            'crop_fraction': self.crop_fraction,
            'mip_axis': self.mip_axis
        }
        
        with open(self.output_dir / 'dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*80)
        print("DATASET CREATION COMPLETE")
        print("="*80)
        print(f"Total volumes: {len(volume_paths)}")
        print(f"Total samples: {total_samples}")
        print(f"  Train: {split_stats['train']} samples from {len(splits['train'])} volumes")
        print(f"  Val: {split_stats['val']} samples from {len(splits['val'])} volumes")
        print(f"  Test: {split_stats['test']} samples from {len(splits['test'])} volumes")
        print(f"\nDataset saved to: {self.output_dir}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def collect_mat_files(root_dir: str, pattern: str = '*.mat') -> List[str]:
    """
    Recursively collect all .mat files from directory.
    
    Args:
        root_dir: Root directory to search
        pattern: File pattern to match (default: '*.mat')
    
    Returns:
        List of full paths to .mat files
    """
    root_path = Path(root_dir)
    mat_files = list(root_path.rglob(pattern))
    return [str(f) for f in mat_files]


def verify_dataset_structure(dataset_dir: str):
    """
    Verify that dataset was created correctly.
    
    Args:
        dataset_dir: Path to dataset root directory
    """
    dataset_path = Path(dataset_dir)
    
    print("Verifying dataset structure...")
    
    for split in ['train', 'val', 'test']:
        split_dir = dataset_path / split
        if not split_dir.exists():
            print(f"  WARNING: {split} directory not found")
            continue
        
        sample_dirs = list(split_dir.iterdir())
        print(f"  {split}: {len(sample_dirs)} samples")
        
        # Check first sample
        if sample_dirs:
            sample_dir = sample_dirs[0]
            required_files = ['HQ.npy', 'LQ.npy', 'mask.npy', 'metadata.json']
            
            for req_file in required_files:
                if not (sample_dir / req_file).exists():
                    print(f"    WARNING: {req_file} missing in {sample_dir.name}")
            
            # Load and check shapes
            hq = np.load(sample_dir / 'HQ.npy')
            lq = np.load(sample_dir / 'LQ.npy')
            mask = np.load(sample_dir / 'mask.npy')
            
            print(f"    Sample shape verification:")
            print(f"      HQ: {hq.shape}, dtype: {hq.dtype}")
            print(f"      LQ: {lq.shape}, dtype: {lq.dtype}")
            print(f"      Mask: {mask.shape}, dtype: {mask.dtype}")
            
            with open(sample_dir / 'metadata.json', 'r') as f:
                metadata = json.load(f)
            print(f"      Sampling ratio: {metadata['actual_sampling_ratio']:.2%}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    # =========================================================================
    # CONFIGURATION - MODIFY THESE PATHS AND PARAMETERS
    # =========================================================================
    
    # Path to directory containing your .mat files
    # This should contain ~150 .mat files organized in 3 categories
    RAW_DATA_DIR = '/home/v207e/E130-Projekte/Photoacoustics/RawData/20250404_rsom_invivo_fabian/Biederstein_Device'
    
    # Output directory for dataset
    OUTPUT_DIR = '/home/v207e/GitLab/v207e/v207e/SupreRes/Data/RSOM/processed_new'
    
    # Pipeline parameters
    MASK_TYPES = ['random', 'semistructured', 'variabledensity']
    SAMPLING_RATIOS = [0.3, 0.4, 0.5]
    N_MASKS_PER_RATIO = 3  # Generate 3 different masks per ratio
    CROP_FRACTION = 0.20   # Crop 20% from each side
    MIP_AXIS = 1           # Y-axis for MIP
    
    # Dataset split ratios
    TRAIN_RATIO = 0.7      # 70% for training
    VAL_RATIO = 0.15       # 15% for validation, 15% for test
    
    # Whether to save visualization images
    SAVE_VISUALIZATIONS = True  # Set to False for faster processing
    
    # =========================================================================
    # PIPELINE EXECUTION
    # =========================================================================
    start = time()
    print("Collecting .mat files...")
    volume_paths = collect_mat_files(RAW_DATA_DIR)
    # import ipdb; ipdb.set_trace()
    print(f"Found {len(volume_paths)} .mat files")
    
    if len(volume_paths) == 0:
        raise ValueError(f"No .mat files found in {RAW_DATA_DIR}")
    
    # Initialize pipeline
    pipeline = PhotoacousticDatasetPipeline(
        output_dir=OUTPUT_DIR,
        mask_types=MASK_TYPES,
        sampling_ratios=SAMPLING_RATIOS,
        n_masks_per_ratio=N_MASKS_PER_RATIO,
        crop_fraction=CROP_FRACTION,
        mip_axis=MIP_AXIS
    )
    
    # Create dataset
    pipeline.create_dataset(
        volume_paths=volume_paths,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        save_visualizations=SAVE_VISUALIZATIONS
    )
    
    # Verify dataset
    verify_dataset_structure(OUTPUT_DIR)
    
    print("\nPipeline complete!")
    print(f"Dataset ready for training at: {OUTPUT_DIR}")
    print('Total Elapsed time:', time() - start)


if __name__ == '__main__':
    main()