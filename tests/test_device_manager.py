import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.device_manager import DeviceManager

class TestDeviceManager:
    """Test suite for device manager"""
    
    def test_get_optimal_device(self):
        """Test optimal device selection"""
        device = DeviceManager.get_optimal_device()
        assert device is not None
        assert hasattr(device, 'type')
    
    def test_print_device_info(self):
        """Test device info printing (should not raise errors)"""
        try:
            DeviceManager.print_device_info()
        except Exception as e:
            pytest.fail(f"print_device_info raised an exception: {e}")
    
    def test_move_model_to_device(self):
        """Test model device movement"""
        # Create a mock model-like object
        class MockModel:
            def to(self, device):
                self.device = device
                return self
        
        model = MockModel()
        result = DeviceManager.move_model_to_device(model)
        assert result is not None
        assert hasattr(result, 'device')