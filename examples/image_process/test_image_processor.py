"""
Quick test script for image_processor module
Simple functionality tests without extensive benchmarking

Usage:
    python test_image_processor.py
"""

import numpy as np
import sys

def test_import():
    """Test if module can be imported"""
    print("="*60)
    print("Test 1: Module Import")
    print("="*60)
    try:
        import image_processor
        print("✓ Successfully imported image_processor")
        return image_processor
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        print("\nCompile the module first with:")
        print("  c++ -O3 -Wall -shared -std=c++11 -fPIC \\")
        print("      $(python3 -m pybind11 --includes) \\")
        print("      image_processor.cpp \\")
        print("      -o image_processor$(python3-config --extension-suffix)")
        sys.exit(1)


def test_instantiation(image_processor):
    """Test creating ImageProcessor instance"""
    print("\n" + "="*60)
    print("Test 2: Create ImageProcessor Instance")
    print("="*60)
    try:
        processor = image_processor.ImageProcessor()
        print("✓ Successfully created ImageProcessor instance")
        return processor
    except Exception as e:
        print(f"✗ Failed to create instance: {e}")
        sys.exit(1)


def test_gaussian_blur(processor):
    """Test Gaussian blur functionality"""
    print("\n" + "="*60)
    print("Test 3: Gaussian Blur")
    print("="*60)
    
    # Create test image
    image = np.random.rand(100, 100)
    print(f"  Input image shape: {image.shape}")
    print(f"  Input image dtype: {image.dtype}")
    print(f"  Input range: [{image.min():.3f}, {image.max():.3f}]")
    
    try:
        result = processor.gaussianBlur(image, sigma=1.0)
        print(f"  Output image shape: {result.shape}")
        print(f"  Output image dtype: {result.dtype}")
        print(f"  Output range: [{result.min():.3f}, {result.max():.3f}]")
        
        # Verify shape is preserved
        assert result.shape == image.shape, "Shape mismatch!"
        print("✓ Gaussian blur works correctly")
        return True
    except Exception as e:
        print(f"✗ Gaussian blur failed: {e}")
        return False


def test_sobel_edge_detection(processor):
    """Test Sobel edge detection"""
    print("\n" + "="*60)
    print("Test 4: Sobel Edge Detection")
    print("="*60)
    
    # Create test image with an edge
    image = np.zeros((100, 100))
    image[:, 50:] = 1.0  # Vertical edge in the middle
    print(f"  Input image shape: {image.shape}")
    print(f"  Created image with vertical edge")
    
    try:
        result = processor.sobelEdgeDetection(image)
        print(f"  Output image shape: {result.shape}")
        print(f"  Output range: [{result.min():.3f}, {result.max():.3f}]")
        
        # Verify shape is preserved
        assert result.shape == image.shape, "Shape mismatch!"
        
        # Verify edge was detected (should have high values near column 50)
        edge_region = result[:, 48:52].max()
        print(f"  Max value in edge region: {edge_region:.3f}")
        assert edge_region > 0.5, "Edge not detected!"
        
        print("✓ Sobel edge detection works correctly")
        return True
    except Exception as e:
        print(f"✗ Sobel edge detection failed: {e}")
        return False


def test_histogram(processor):
    """Test histogram computation"""
    print("\n" + "="*60)
    print("Test 5: Histogram Computation")
    print("="*60)
    
    # Create test image
    image = np.random.rand(100, 100)
    bins = 256
    print(f"  Input image shape: {image.shape}")
    print(f"  Computing histogram with {bins} bins")
    
    try:
        result = processor.computeHistogram(image, bins)
        print(f"  Histogram length: {len(result)}")
        print(f"  Total count: {sum(result)}")
        print(f"  Expected count: {image.size}")
        
        # Verify
        assert len(result) == bins, f"Expected {bins} bins, got {len(result)}"
        assert sum(result) == image.size, "Histogram counts don't match image size!"
        
        print("✓ Histogram computation works correctly")
        return True
    except Exception as e:
        print(f"✗ Histogram computation failed: {e}")
        return False


def test_batch_processing(processor):
    """Test batch processing"""
    print("\n" + "="*60)
    print("Test 6: Batch Processing")
    print("="*60)
    
    # Create multiple test images
    num_images = 5
    images = [np.random.rand(50, 50) for _ in range(num_images)]
    print(f"  Created {num_images} test images of shape (50, 50)")
    
    try:
        # Test blur operation
        print("  Testing batch blur...")
        results_blur = processor.batchProcess(images, "blur")
        assert len(results_blur) == num_images, "Wrong number of results!"
        print(f"    ✓ Processed {len(results_blur)} images with blur")
        
        # Test edge operation
        print("  Testing batch edge detection...")
        results_edge = processor.batchProcess(images, "edge")
        assert len(results_edge) == num_images, "Wrong number of results!"
        print(f"    ✓ Processed {len(results_edge)} images with edge detection")
        
        print("✓ Batch processing works correctly")
        return True
    except Exception as e:
        print(f"✗ Batch processing failed: {e}")
        return False


def test_error_handling(processor):
    """Test error handling with invalid inputs"""
    print("\n" + "="*60)
    print("Test 7: Error Handling")
    print("="*60)
    
    passed = 0
    total = 0
    
    # Test 1: Invalid dimension (3D instead of 2D)
    total += 1
    try:
        invalid_image = np.random.rand(10, 10, 3)
        processor.gaussianBlur(invalid_image)
        print("  ✗ Should have rejected 3D array")
    except:
        print("  ✓ Correctly rejected 3D array")
        passed += 1
    
    # Test 2: Invalid operation in batch processing
    total += 1
    try:
        images = [np.random.rand(10, 10)]
        processor.batchProcess(images, "invalid_operation")
        print("  ✗ Should have rejected invalid operation")
    except:
        print("  ✓ Correctly rejected invalid operation")
        passed += 1
    
    print(f"\n✓ Error handling: {passed}/{total} tests passed")
    return passed == total


def quick_performance_check(processor):
    """Quick performance check"""
    print("\n" + "="*60)
    print("Test 8: Quick Performance Check")
    print("="*60)
    
    import time
    
    sizes = [(64, 64), (128, 128), (256, 256)]
    
    for height, width in sizes:
        image = np.random.rand(height, width)
        
        start = time.perf_counter()
        for _ in range(10):
            _ = processor.gaussianBlur(image)
        end = time.perf_counter()
        
        avg_time = (end - start) / 10
        if avg_time < 1e-3:
            time_str = f"{avg_time * 1e6:.2f} μs"
        else:
            time_str = f"{avg_time * 1e3:.2f} ms"
        
        print(f"  {height}x{width}: {time_str} per blur")
    
    print("✓ Performance check completed")


def main():
    print("\n" + "="*60)
    print("IMAGE PROCESSOR - QUICK TEST SUITE")
    print("="*60)
    
    # Run all tests
    module = test_import()
    processor = test_instantiation(module)
    
    results = []
    results.append(("Gaussian Blur", test_gaussian_blur(processor)))
    results.append(("Sobel Edge Detection", test_sobel_edge_detection(processor)))
    results.append(("Histogram", test_histogram(processor)))
    results.append(("Batch Processing", test_batch_processing(processor)))
    results.append(("Error Handling", test_error_handling(processor)))
    
    # Performance check
    quick_performance_check(processor)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name:<25} {status}")
    
    print("-"*60)
    print(f"  Total: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n🎉 All tests passed! The module is working correctly.\n")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())