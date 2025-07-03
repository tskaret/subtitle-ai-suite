#!/usr/bin/env python3
"""
Test runner for Subtitle AI Suite
"""

import subprocess
import sys
import os
from pathlib import Path

def run_tests():
    """Run the test suite"""
    print("🧪 Running Subtitle AI Suite Test Suite")
    print("=" * 50)
    
    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Run pytest
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/', 
            '-v',  # verbose output
            '--tb=short',  # shorter traceback format
        ], check=True)
        
        print("\n✅ All tests passed!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code: {e.returncode}")
        return False
    except FileNotFoundError:
        print("❌ pytest not found. Please install it with: pip install pytest")
        return False

def run_quick_test():
    """Run a quick smoke test"""
    print("🚀 Running quick smoke test...")
    
    try:
        # Test imports
        sys.path.insert(0, str(Path(__file__).parent / 'src'))
        
        from utils.device_manager import DeviceManager
        print("✓ Device manager import successful")
        
        device = DeviceManager.get_optimal_device()
        print(f"✓ Device detection successful: {device}")
        
        print("✓ Quick test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Subtitle AI Suite tests")
    parser.add_argument('--quick', action='store_true', help='Run quick smoke test only')
    args = parser.parse_args()
    
    if args.quick:
        success = run_quick_test()
    else:
        success = run_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()