"""
Device Detection and Management
Supports CUDA, MPS (Apple Silicon), and CPU
"""

import torch
import platform
from typing import Tuple


def get_available_device(preferred: str = 'auto') -> Tuple[torch.device, str]:
    """
    Get the best available device for training/inference
    
    Args:
        preferred: 'auto', 'cuda', 'mps', or 'cpu'
    
    Returns:
        device: torch.device object
        device_name: String description of device
    """
    
    if preferred == 'auto':
        # Auto-detect best available device
        if torch.cuda.is_available():
            device = torch.device('cuda')
            device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            device_name = "MPS (Apple Silicon)"
        else:
            device = torch.device('cpu')
            device_name = "CPU"
    
    elif preferred == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
        else:
            raise RuntimeError("CUDA requested but not available")
    
    elif preferred == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            device_name = "MPS (Apple Silicon)"
        else:
            raise RuntimeError("MPS requested but not available")
    
    elif preferred == 'cpu':
        device = torch.device('cpu')
        device_name = "CPU"
    
    else:
        raise ValueError(f"Unknown device: {preferred}. Use 'auto', 'cuda', 'mps', or 'cpu'")
    
    return device, device_name


def get_device_info() -> dict:
    """
    Get comprehensive device information
    
    Returns:
        info: Dictionary with device details
    """
    info = {
        'platform': platform.system(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
    }
    
    # CUDA information
    info['cuda_available'] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_devices'] = [
            torch.cuda.get_device_name(i) 
            for i in range(torch.cuda.device_count())
        ]
        info['cuda_memory'] = {
            f'device_{i}': f"{torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB"
            for i in range(torch.cuda.device_count())
        }
    
    # MPS information
    info['mps_available'] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if info['mps_available']:
        info['mps_built'] = torch.backends.mps.is_built()
    
    # Get recommended device
    device, device_name = get_available_device('auto')
    info['recommended_device'] = str(device)
    info['recommended_device_name'] = device_name
    
    return info


def print_device_info():
    """Print device information in a readable format"""
    info = get_device_info()
    
    print("\n" + "="*60)
    print("  Device Information")
    print("="*60)
    
    print(f"\nPlatform: {info['platform']}")
    print(f"Processor: {info['processor']}")
    print(f"Python: {info['python_version']}")
    print(f"PyTorch: {info['pytorch_version']}")
    
    print(f"\n{'='*60}")
    print("  Available Devices")
    print("="*60)
    
    # CUDA
    if info['cuda_available']:
        print(f"\nCUDA: Available")
        print(f"   Version: {info['cuda_version']}")
        print(f"   Devices: {info['cuda_device_count']}")
        for i, name in enumerate(info['cuda_devices']):
            memory = info['cuda_memory'][f'device_{i}']
            print(f"     [{i}] {name} ({memory})")
    else:
        print(f"\nCUDA: Not available")
    
    # MPS
    if info['mps_available']:
        print(f"\nMPS (Apple Silicon): Available")
        print(f"   Built: {info.get('mps_built', 'N/A')}")
    else:
        print(f"\nMPS: Not available")
    
    # CPU
    print(f"\nCPU: Always available")
    
    print(f"\n{'='*60}")
    print(f"Recommended Device: {info['recommended_device_name']}")
    print("="*60 + "\n")


def get_optimal_device_settings(device: torch.device) -> dict:
    """
    Get optimal settings for a given device
    
    Args:
        device: torch.device object
    
    Returns:
        settings: Dictionary with optimal settings
    """
    settings = {
        'device': device,
        'device_type': device.type,
        'pin_memory': False,
        'num_workers': 4,
        'persistent_workers': False,
    }
    
    if device.type == 'cuda':
        # CUDA optimizations
        settings['pin_memory'] = True
        settings['persistent_workers'] = True
        
        # Check VRAM for batch size recommendation
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if total_memory < 8:
                settings['recommended_batch_size'] = 8
            elif total_memory < 12:
                settings['recommended_batch_size'] = 16
            else:
                settings['recommended_batch_size'] = 32
    
    elif device.type == 'mps':
        # MPS optimizations
        settings['pin_memory'] = False # Not needed for MPS
        settings['num_workers'] = 4
        settings['persistent_workers'] = False
        settings['recommended_batch_size'] = 32
    
    elif device.type == 'cpu':
        # CPU optimizations
        settings['pin_memory'] = False
        settings['num_workers'] = 4
        settings['persistent_workers'] = False
        settings['recommended_batch_size'] = 4 # CPU is slow, use small batches
    
    return settings


def is_mps_compatible_operation(operation: str) -> bool:
    """
    Check if an operation is compatible with MPS
    
    Some PyTorch operations may not be fully supported on MPS yet
    """
    # List of operations known to have issues on MPS
    incompatible_ops = [
        'group_norm', # May have issues in some PyTorch versions
        'upsample_bilinear2d', # May fallback to CPU
    ]
    
    return operation not in incompatible_ops


def setup_device_for_training(config: dict) -> Tuple[torch.device, dict]:
    """
    Setup device based on config and return optimal settings
    
    Args:
        config: Configuration dictionary
    
    Returns:
        device: Selected device
        settings: Optimal settings for the device
    """
    # Get device preference from config
    device_pref = config.get('train', {}).get('device', 'auto')
    
    # Get available device
    device, device_name = get_available_device(device_pref)
    
    # Get optimal settings
    settings = get_optimal_device_settings(device)
    
    print(f"\n{'='*60}")
    print(f"  Training Device: {device_name}")
    print(f"{'='*60}")
    
    if device.type == 'cuda':
        print(f"Using CUDA GPU acceleration")
        print(f"   Recommended batch size: {settings.get('recommended_batch_size', 'N/A')}")
    elif device.type == 'mps':
        print(f"Using Apple Silicon (MPS) acceleration")
        print(f"   Recommended batch size: {settings.get('recommended_batch_size', 'N/A')}")
    else:
        print(f"Using CPU (training will be slow)")
        print(f"   Recommended batch size: {settings.get('recommended_batch_size', 'N/A')}")
    
    print(f"{'='*60}\n")
    
    return device, settings


if __name__ == '__main__':
    print_device_info()