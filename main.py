import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
import os

# Load and preprocess the image
image = io.imread('/content/biodivercity_bjarkeingelsgroup_00-1-768x432.webp')  # Replace 'path_to_image.jpg' with the actual path to your image
image = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Define the autoencoder architecture
input_img = Input(shape=(image.shape[1], image.shape[2], image.shape[3]))

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # Output layer

# Autoencoder model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(image, image, epochs=500, batch_size=1, verbose=1)

# Compress and reconstruct the image
encoded_img = autoencoder.predict(image)

# Save the original and reconstructed images to compare sizes
original_image_path = 'original_image.jpg'
compressed_image_path = 'compressed_image.jpg'

# Remove batch dimension and save images
original_image = np.squeeze(image) * 255  # Convert back to 0-255 range
reconstructed_image = np.squeeze(encoded_img) * 255  # Convert back to 0-255 range

# Convert arrays to images
original_img = Image.fromarray(original_image.astype('uint8'))
compressed_img = Image.fromarray(reconstructed_image.astype('uint8'))

# Save images
original_img.save(original_image_path)
compressed_img.save(compressed_image_path)

# Get file sizes
original_size = os.path.getsize(original_image_path)
compressed_size = os.path.getsize(compressed_image_path)

# Display file sizes
print(f"Original Image Size: {original_size / 1024:.2f} KB")
print(f"Compressed Image Size: {compressed_size / 1024:.2f} KB")

# Display original and compressed images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(original_img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Compressed & Reconstructed Image')
plt.imshow(compressed_img)
plt.axis('off')

plt.show()
