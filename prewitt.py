import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh đầu vào
image = cv2.imread('thap_eiffel.jpg', cv2.IMREAD_GRAYSCALE)

# Tạo kernel Prewitt cho hướng x và y
prewitt_kernel_x = np.array([[1, 0, -1], 
                             [1, 0, -1], 
                             [1, 0, -1]])

prewitt_kernel_y = np.array([[1, 1, 1], 
                             [0, 0, 0], 
                             [-1, -1, -1]])

# Áp dụng bộ lọc Prewitt cho hướng x và y
# Chuyển đổi ảnh sang kiểu float64 để thực hiện phép tính
prewitt_x = cv2.filter2D(image.astype(np.float64), -1, prewitt_kernel_x)
prewitt_y = cv2.filter2D(image.astype(np.float64), -1, prewitt_kernel_y)

# Tổng hợp gradient theo cả hai hướng
prewitt_combined = np.sqrt(prewitt_x**2 + prewitt_y**2)

# Chuẩn hóa kết quả 
prewitt_combined = cv2.normalize(prewitt_combined, None, 0, 255, cv2.NORM_MINMAX)

# Hiển thị kết quả
# plt.subplot(1, 4, 1), plt.imshow(image, cmap='gray'), plt.title('Ảnh gốc')
# plt.subplot(1, 4, 2), plt.imshow(prewitt_x, cmap='gray'), plt.title('Prewitt X')
# plt.subplot(1, 4, 3), plt.imshow(prewitt_y, cmap='gray'), plt.title('Prewitt y')
# plt.subplot(1, 4, 4), 
plt.imshow(prewitt_combined, cmap='gray'), plt.title('Prewitt tổng hợp')
plt.tight_layout()
plt.show()
