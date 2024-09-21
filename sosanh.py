import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh đầu vào
image = cv2.imread('thap_eiffel.jpg', cv2.IMREAD_GRAYSCALE)

# Phương pháp Canny
blurred_image = cv2.GaussianBlur(image, (5, 5), 1)
edges_canny = cv2.Canny(blurred_image, 50, 150)

# Phương pháp Prewitt
prewitt_kernel_x = np.array([[1, 0, -1], 
                             [1, 0, -1], 
                             [1, 0, -1]])

prewitt_kernel_y = np.array([[1, 1, 1], 
                             [0, 0, 0], 
                             [-1, -1, -1]])

prewitt_x = cv2.filter2D(image.astype(np.float64), -1, prewitt_kernel_x)
prewitt_y = cv2.filter2D(image.astype(np.float64), -1, prewitt_kernel_y)

prewitt_combined = np.sqrt(prewitt_x**2 + prewitt_y**2)
prewitt_combined = cv2.normalize(prewitt_combined, None, 0, 255, cv2.NORM_MINMAX)

# Hiển thị cả hai kết quả để so sánh
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Ảnh gốc')
plt.axis('off')

# Hiển thị kết quả của phương pháp Canny
plt.subplot(1, 3, 2)
plt.imshow(edges_canny, cmap='gray')
plt.title('Phương pháp Canny')
plt.axis('off')

# Hiển thị kết quả của phương pháp Prewitt
plt.subplot(1, 3, 3)
plt.imshow(prewitt_combined, cmap='gray')
plt.title('Kỹ thuật Prewitt')
plt.axis('off')

# Hiển thị hình ảnh
plt.tight_layout()
plt.show()
