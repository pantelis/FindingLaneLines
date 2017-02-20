import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2  #bringing in OpenCV libraries

image = mpimg.imread('exit-ramp.jpg')

# display original image
plt.figure("Original Image")
plt.imshow(image)
plt.show()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #grayscale conversion

plt.figure("Grayscale Image")
plt.imshow(gray, cmap='gray')
plt.show()

# Define a kernel size for Gaussian smoothing / blurring
# Note: this step is optional as cv2.Canny() applies a 5x5 Gaussian internally
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

# Define parameters for Canny and run it
# NOTE: if you try running this code you might want to change these!
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Display the image
plt.figure("Canny Detector Output")
plt.imshow(edges, cmap='Greys_r')
plt.show()
