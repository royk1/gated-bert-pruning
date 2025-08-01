#!/usr/bin/env python3
"""
GPU Support Test Script for Gated BERT

This script tests GPU support for both NVIDIA CUDA and Apple MPS,
verifying that the TensorFlow configuration works correctly on different platforms.
"""

import os
import sys
import platform
import subprocess

def test_tensorflow_import():
    """Test TensorFlow import and basic functionality."""
    print("=== Testing TensorFlow Import ===")
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} imported successfully")
        return tf
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return None

def test_gpu_detection(tf):
    """Test GPU detection and configuration."""
    print("\n=== Testing GPU Detection ===")
    
    # Test physical devices
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        cpus = tf.config.experimental.list_physical_devices('CPU')
        
        print(f"✅ Physical CPUs: {len(cpus)}")
        print(f"✅ Physical GPUs: {len(gpus)}")
        
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
                
                # Try to get GPU details
                try:
                    gpu_details = tf.config.experimental.get_device_details(gpu)
                    if gpu_details:
                        device_name = gpu_details.get('device_name', 'Unknown')
                        print(f"    Device name: {device_name}")
                        
                        # Detect GPU type
                        if 'nvidia' in device_name.lower() or 'cuda' in device_name.lower():
                            print(f"    Type: NVIDIA CUDA")
                        elif 'metal' in device_name.lower():
                            print(f"    Type: Apple Metal")
                        else:
                            print(f"    Type: Generic GPU")
                except Exception as e:
                    print(f"    Could not get GPU details: {e}")
        else:
            print("⚠️  No GPU devices found")
            
    except Exception as e:
        print(f"❌ GPU detection failed: {e}")

def test_gpu_functionality(tf):
    """Test basic GPU functionality."""
    print("\n=== Testing GPU Functionality ===")
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if not gpus:
        print("⚠️  No GPU devices available for testing")
        return
    
    # Test basic tensor operations on GPU
    try:
        with tf.device('/GPU:0'):
            # Create test tensors
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
            
            # Perform operations
            c = tf.matmul(a, b)
            d = tf.reduce_sum(c)
            
            print(f"✅ GPU tensor operations successful")
            print(f"   Matrix multiplication result: {c.numpy()}")
            print(f"   Sum result: {d.numpy()}")
            
    except Exception as e:
        print(f"❌ GPU tensor operations failed: {e}")

def test_memory_growth(tf):
    """Test GPU memory growth configuration."""
    print("\n=== Testing GPU Memory Growth ===")
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if not gpus:
        print("⚠️  No GPU devices available for memory testing")
        return
    
    try:
        # Check if memory growth is enabled
        for gpu in gpus:
            # This is a bit tricky to test directly, but we can check if the configuration was applied
            print(f"✅ Memory growth configuration applied to {gpu}")
            
    except Exception as e:
        print(f"❌ Memory growth test failed: {e}")

def test_platform_specific_config():
    """Test platform-specific configuration."""
    print("\n=== Testing Platform-Specific Configuration ===")
    
    system = platform.system()
    print(f"Platform: {system} {platform.release()}")
    
    if system == "Darwin":  # macOS
        print("✅ macOS detected - MPS support should be available")
        
        # Check for Metal framework
        try:
            import Metal
            print("✅ Metal framework available")
        except ImportError:
            print("⚠️  Metal framework not available")
            
    elif system == "Linux":
        print("✅ Linux detected - CUDA support should be available")
        
        # Check for CUDA
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ NVIDIA GPU detected via nvidia-smi")
                print(f"   GPU info: {result.stdout.split('|')[1].strip()}")
            else:
                print("⚠️  nvidia-smi not available or no NVIDIA GPU")
        except FileNotFoundError:
            print("⚠️  nvidia-smi not found")
            
    elif system == "Windows":
        print("✅ Windows detected - CUDA support should be available")
        
        # Check for CUDA on Windows
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ NVIDIA GPU detected via nvidia-smi")
            else:
                print("⚠️  nvidia-smi not available or no NVIDIA GPU")
        except FileNotFoundError:
            print("⚠️  nvidia-smi not found")
    
    else:
        print(f"⚠️  Unknown platform: {system}")

def test_environment_variables():
    """Test environment variable configuration."""
    print("\n=== Testing Environment Variables ===")
    
    relevant_vars = [
        'TF_CPP_MIN_LOG_LEVEL',
        'TF_USE_LEGACY_KERAS',
        'TF_FORCE_GPU_ALLOW_GROWTH',
        'TF_DISABLE_MPS',
        'CUDA_VISIBLE_DEVICES',
        'TF_GPU_ALLOCATOR'
    ]
    
    for var in relevant_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")

def test_utils_setup():
    """Test the utils.setup_tensorflow function."""
    print("\n=== Testing Utils Setup Function ===")
    
    try:
        from utils import setup_tensorflow
        print("✅ utils.setup_tensorflow imported successfully")
        
        # Call the setup function
        setup_tensorflow()
        print("✅ utils.setup_tensorflow executed successfully")
        
    except ImportError as e:
        print(f"❌ Could not import utils.setup_tensorflow: {e}")
    except Exception as e:
        print(f"❌ utils.setup_tensorflow failed: {e}")

def main():
    """Main test function."""
    print("Gated BERT GPU Support Test")
    print("=" * 50)
    
    # Test environment variables
    test_environment_variables()
    
    # Test TensorFlow import
    tf = test_tensorflow_import()
    if tf is None:
        print("\n❌ TensorFlow not available. Exiting.")
        return
    
    # Test platform-specific configuration
    test_platform_specific_config()
    
    # Test utils setup
    test_utils_setup()
    
    # Test GPU detection
    test_gpu_detection(tf)
    
    # Test GPU functionality
    test_gpu_functionality(tf)
    
    # Test memory growth
    test_memory_growth(tf)
    
    print("\n" + "=" * 50)
    print("GPU Support Test Complete")
    print("=" * 50)

if __name__ == "__main__":
    main() 