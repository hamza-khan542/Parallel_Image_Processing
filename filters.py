from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
from scipy.ndimage import gaussian_filter, sobel
import multiprocessing as mp

# Create a persistent process pool to avoid startup overhead
_process_pool = None

def get_process_pool(num_workers=4):
    """Get or create a persistent process pool"""
    global _process_pool
    if _process_pool is None:
        _process_pool = ProcessPoolExecutor(max_workers=num_workers)
    return _process_pool

def apply_filter_sequential(img, filter_type):
    """Sequential processing using numpy for actual computation"""
    start = time.time()
    img_array = np.array(img)
    
    if filter_type == "Grayscale":
        # Manual grayscale conversion with weights
        result_array = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        result_array = np.stack([result_array] * 3, axis=-1).astype(np.uint8)
    elif filter_type == "Blur":
        # Apply Gaussian blur to each channel
        result_array = np.zeros_like(img_array)
        for i in range(3):
            result_array[:, :, i] = gaussian_filter(img_array[:, :, i], sigma=2)
        result_array = result_array.astype(np.uint8)
    elif filter_type == "Edge Detection":
        # Sobel edge detection on each channel
        result_array = np.zeros_like(img_array)
        for i in range(3):
            sx = sobel(img_array[:, :, i], axis=0, mode='constant')
            sy = sobel(img_array[:, :, i], axis=1, mode='constant')
            result_array[:, :, i] = np.hypot(sx, sy)
        result_array = np.clip(result_array, 0, 255).astype(np.uint8)
    else:
        raise ValueError("Unknown filter type")
    
    result = Image.fromarray(result_array)
    end = time.time()
    return result, end - start

def process_chunk_numpy(args):
    """Process a chunk using numpy operations"""
    chunk_data, filter_type = args
    
    if filter_type == "Grayscale":
        result = np.dot(chunk_data[..., :3], [0.2989, 0.5870, 0.1140])
        result = np.stack([result] * 3, axis=-1).astype(np.uint8)
    elif filter_type == "Blur":
        result = np.zeros_like(chunk_data)
        for i in range(3):
            result[:, :, i] = gaussian_filter(chunk_data[:, :, i], sigma=2)
        result = result.astype(np.uint8)
    elif filter_type == "Edge Detection":
        result = np.zeros_like(chunk_data)
        for i in range(3):
            sx = sobel(chunk_data[:, :, i], axis=0, mode='constant')
            sy = sobel(chunk_data[:, :, i], axis=1, mode='constant')
            result[:, :, i] = np.hypot(sx, sy)
        result = np.clip(result, 0, 255).astype(np.uint8)
    else:
        result = chunk_data
    
    return result

def apply_filter_parallel(img, filter_type, num_threads=4):
    """Optimized parallel processing with minimal overhead"""
    start = time.time()
    
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    
    # Optimize thread count: larger chunks = less overhead
    # Use fewer, larger chunks to minimize process communication
    optimal_threads = min(num_threads, max(2, height // 200))
    
    if optimal_threads < 2:
        # Too small for parallelization
        return apply_filter_sequential(img, filter_type)
    
    chunk_height = height // optimal_threads
    
    # Split into chunks (contiguous memory is faster)
    chunks = []
    for i in range(optimal_threads):
        start_row = i * chunk_height
        end_row = (i + 1) * chunk_height if i < optimal_threads - 1 else height
        chunk = img_array[start_row:end_row].copy()  # Copy for better pickling
        chunks.append((chunk, filter_type))
    
    # Use persistent pool to avoid process creation overhead
    pool = get_process_pool(optimal_threads)
    
    # Process chunks in parallel
    processed_chunks = list(pool.map(process_chunk_numpy, chunks))
    
    # Combine results
    result_array = np.vstack(processed_chunks)
    result = Image.fromarray(result_array)
    
    end = time.time()
    return result, end - start
