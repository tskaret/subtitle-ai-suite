import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_basic_imports():
    """Test that basic modules can be imported"""
    try:
        from utils.device_manager import DeviceManager
        assert DeviceManager is not None
    except ImportError as e:
        pytest.fail(f"Failed to import DeviceManager: {e}")

def test_torch_import():
    """Test torch import and basic functionality"""
    try:
        import torch
        device = torch.device('cpu')  # Always use CPU for testing
        assert device.type == 'cpu'
    except ImportError as e:
        pytest.fail(f"Failed to import torch: {e}")

def test_project_structure():
    """Test that project structure is correct"""
    project_root = Path(__file__).parent.parent
    
    # Check essential directories exist
    assert (project_root / 'src').exists()
    assert (project_root / 'src' / 'core').exists()
    assert (project_root / 'src' / 'utils').exists()
    
    # Check essential files exist
    assert (project_root / 'src' / 'main.py').exists()
    assert (project_root / 'requirements.txt').exists()
    assert (project_root / 'README.md').exists()