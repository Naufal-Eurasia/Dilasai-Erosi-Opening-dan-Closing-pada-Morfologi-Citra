# Dilasai-Erosi-Opening-dan-Closing-pada-Morfologi-Citra
Dilasai, Erosi, Opening dan Closing pada Morfologi Citra
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Baca gambar
img = cv2.imread('/content/s_Prasasti.jpg')

# 2. Konversi ke grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Tingkatkan kontras dengan CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)

# 4. Threshold adaptif
thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 15, 10)

# 5. Operasi morfologi (dilasi untuk mempertebal)
kernel = np.ones((2, 2), np.uint8)
dilated = cv2.dilate(thresh, kernel, iterations=1)

# 6. Tampilkan hasil
titles = ['Asli', 'Grayscale + CLAHE', 'Thresholding', 'Setelah Dilasi']
images = [img, enhanced, thresh, dilated]

plt.figure(figsize=(12, 6))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(images[i], cmap='gray' if i > 0 else None)
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
```
Program ini menghasilkan gambar dengan metode Dilasi,Erosi,Opening,dan Closing
![image](https://github.com/user-attachments/assets/92c82b1c-8120-47c6-a803-089ad17239cb)

