import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh đầu vào
image = cv2.imread('thap_eiffel.jpg', cv2.IMREAD_GRAYSCALE)    

# Bước 1: Làm mờ ảnh với Gaussian Blur
blurred_image = cv2.GaussianBlur(image, (5, 5), 1)

# Bước 2: Phát hiện cạnh với Canny (ngưỡng thấp = 55, ngưỡng cao = 200)
edges = cv2.Canny(blurred_image, 0, 255)

# Hiển thị kết quả
# plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Ảnh gốc')
# plt.subplot(1, 3, 2), plt.imshow(blurred_image, cmap='gray'), plt.title('Ảnh làm mờ')
#plt.subplot(1, 3, 3), 
plt.imshow(edges, cmap='gray'), plt.title('Cạnh phát hiện với Canny')
plt.show()