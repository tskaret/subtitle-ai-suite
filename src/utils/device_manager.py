import torch
import logging

class DeviceManager:
    """
    Centralized device management for AI processing
    """
    @staticmethod
    def get_optimal_device() -> torch.device:
        """
        Determine the optimal processing device
        
        Returns:
            torch.device: Preferred device (GPU if available, else CPU)
        """
        if torch.cuda.is_available():
            logging.info(f"GPU Available: {torch.cuda.get_device_name(0)}")
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            logging.info("Apple Metal Performance Shaders (MPS) GPU Available")
            return torch.device('mps')
        else:
            logging.info("No GPU available. Falling back to CPU.")
            return torch.device('cpu')

    @staticmethod
    def print_device_info():
        """
        Print detailed device information
        """
        device = DeviceManager.get_optimal_device()
        
        if device.type == 'cuda':
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
        elif device.type == 'mps':
            print("Using Apple Metal Performance Shaders (MPS) GPU")
        else:
            print("Using CPU for processing")
        
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Optimal Device: {device}")

    @staticmethod
    def move_model_to_device(model):
        """
        Move a model to the optimal device
        
        Args:
            model: PyTorch model to move
        
        Returns:
            Model moved to optimal device
        """
        device = DeviceManager.get_optimal_device()
        return model.to(device)