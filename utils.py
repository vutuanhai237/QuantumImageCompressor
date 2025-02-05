import numpy as np

def divide_image(img, k):
    # Add padding if needed
    if int(np.log2(k**2)) != np.log2(k**2):
        raise ValueError("k must be a power of 2")
    h, w = img.shape
    if h % k != 0 or w % k != 0:
        new_h = h + (k - h % k) if h % k != 0 else h
        new_w = w + (k - w % k) if w % k != 0 else w
        padded_img = np.zeros((new_h, new_w), dtype=img.dtype)
        padded_img[:h, :w] = img
        img = padded_img

    blocks = []
    scales = []
    h, w = img.shape
    for i in range(0, h, k):
        for j in range(0, w, k):
            block = img[i:i+k, j:j+k].flatten()
            norm = np.linalg.norm(block)
            if norm != 0:
                block = block / norm
            blocks.append(block)
            scales.append(norm)
    return blocks, scales