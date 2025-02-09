import cv2
import matplotlib.pyplot as plt
import numpy as np
from compressor.image import Image


sizes = [8, 16, 32, 64, 128, 256]

ks = [2]
for k in ks:
    for image in ['lenna.png', 'cameraman.jpg']:
        for size in sizes:
            num_qubits = np.floor(np.log2(k**2)).astype(int)
            num_layers = num_qubits
            img = cv2.imread(f'./images/{image}', cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (size, size))
            image_obj = Image(img, k)
            image_obj.find_thetas_naive(num_layers = num_layers)
            print(image_obj.get_min_stepss())
            np.savetxt(f'./data/{image}_k={k}_size={size}_num_steps_naive.txt', image_obj.get_min_stepss(), fmt='%d')
            print(image_obj.get_costs())
            np.savetxt(f'./data/{image}_k={k}_size={size}_costs_naive.txt', image_obj.get_costs())
            np.savetxt(f'./data/{image}_k={k}_size={size}_percent_transfer_naive.txt', [image_obj.get_num_transfered()])
