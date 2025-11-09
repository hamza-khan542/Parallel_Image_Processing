from PIL import Image, ImageFilter
from concurrent.futures import ThreadPoolExecutor
import time

def apply_filter_sequential(img, filter_type):
    start = time.time()
    if filter_type == "Grayscale":
        result = img.convert("L").convert("RGB")
    elif filter_type == "Blur":
        result = img.filter(ImageFilter.GaussianBlur(2))
    elif filter_type == "Edge Detection":
        result = img.filter(ImageFilter.FIND_EDGES)
    else:
        raise ValueError("Unknown filter type")
    end = time.time()
    return result, end - start

def process_chunk(img, box, filter_type):
    region = img.crop(box)
    if filter_type == "Grayscale":
        region = region.convert("L").convert("RGB")
    elif filter_type == "Blur":
        region = region.filter(ImageFilter.GaussianBlur(2))
    elif filter_type == "Edge Detection":
        region = region.filter(ImageFilter.FIND_EDGES)
    return box, region

def apply_filter_parallel(img, filter_type, num_threads=4):
    width, height = img.size
    
    num_threads = min(num_threads, max(1, height))
    chunk_height = height // num_threads
    boxes = []
    for i in range(num_threads):
        top = i * chunk_height
        bottom = (i + 1) * chunk_height if i < num_threads - 1 else height
        boxes.append((0, top, width, bottom))

    start = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = executor.map(lambda box: process_chunk(img, box, filter_type), boxes)

    result_img = Image.new("RGB", (width, height))
    for box, region in results:
        result_img.paste(region, box)

    end = time.time()
    return result_img, end - start
