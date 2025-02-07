from compressor.image import Image
import cv2
k = 4
img = cv2.imread('./images/lenna.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (12, 12))
image_obj = Image(img, k)
print(image_obj.blocks[0][1])
image_obj.blocks[0][1].find_thetas(init_thetas = None, num_layers = 2)
print(image_obj.blocks[0][1])