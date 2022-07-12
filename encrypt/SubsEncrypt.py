import LogisticKey as key   # Importing the key generating function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import imageio
from PIL import Image
import time

# Accepting an image
path = "../flatcams/001.png"
image = img.imread(path)
imageio.imwrite('orig_im.png', image)
print(image.max())

pil_image = Image.open('../flatcams/001.png')
pil_array = np.asarray(pil_image)/(2**16 -1)
print(pil_array.max())
pil_image = Image.fromarray(pil_array.astype(np.uint8))
pil_image.save('pil_array.png')


# Generating dimensions of the image
height = image.shape[0]
width = image.shape[1]
print(height, width)

# Generating keys
# Calling logistic_key and providing r value such that the keys are pseudo-random
# and generating a key for every pixel of the image
t0 = time.time()
generatedKey = key.logistic_key(0.01, 3.95, height*width)
t1 = time.time()
# Encryption using XOR
z = 0

# Initializing the encrypted image
encryptedImage = np.zeros(shape=[height, width], dtype=np.uint8)

# Substituting all the pixels in original image with nested for
for i in range(height):
    for j in range(width):
        # USing the XOR operation between image pixels and keys
        encryptedImage[i, j] = pil_array[i, j].astype(int) ^ generatedKey[z]
        z += 1
t2 = time.time()
# Displaying the encrypted image
imageio.imwrite('sub_log_key_en.png', encryptedImage)
print('Encrypt time ', t2-t0)
# Decryption using XOR
z = 0
t3 = time.time()
# Initializing the decrypted image
decryptedImage = np.zeros(shape=[height, width], dtype=np.uint8)

# Substituting all the pixels in encrypted image with nested for
for i in range(height):
    for j in range(width):
        # USing the XOR operation between encrypted image pixels and keys
        decryptedImage[i, j] = encryptedImage[i, j].astype(int) ^ generatedKey[z]
        z += 1

# Displaying the decrypted image
t4 = time.time()
print('decryptedImage time ', t4-t3 + t1-t0)
imageio.imwrite('sub_log_key_de.png', decryptedImage)