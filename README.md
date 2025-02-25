


# 🖼️ Image Compression Using Autoencoder  

This project demonstrates **image compression and reconstruction** using a **Convolutional Autoencoder** built with TensorFlow and Keras. The autoencoder reduces the image size while preserving key features, making it useful for deep learning-based compression techniques.  

## 📌 Features  
- Loads and preprocesses an input image.  
- Builds an autoencoder with **CNN layers** for encoding and decoding.  
- Trains the model using **Mean Squared Error (MSE) loss** and **Adam optimizer**.  
- Compresses and reconstructs the image.  
- Compares original and compressed image sizes.  
- Displays the original vs. reconstructed images.  

## 🛠️ Installation & Usage  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-username/image-compression-autoencoder.git
cd image-compression-autoencoder
```

### 2️⃣ Install Dependencies  
```bash
pip install tensorflow numpy matplotlib scikit-image pillow
```

### 3️⃣ Run the Script  
```bash
python autoencoder.py
```

## 📊 Results  
- The **autoencoder** learns a compressed representation of the image.  
- The **reconstructed image** closely resembles the original while having a smaller file size.  
- **File size reduction** is achieved through deep learning-based compression.  

## 📷 Sample Output  
| Original Image | Reconstructed Image |  
|---------------|-------------------|  
| ![Original Image](original_image.jpg) | ![Reconstructed Image](compressed_image.jpg) |  

## 🚀 Future Improvements  
- Train on a **dataset of multiple images** instead of a single image.  
- Implement **Variational Autoencoders (VAEs)** for improved compression.  
- Add **PSNR (Peak Signal-to-Noise Ratio)** as a performance metric.  

## 🤝 Contributing  
Contributions are welcome! If you’d like to improve this project, feel free to fork the repository, create a branch, and submit a pull request.  

## 📜 License  
This project is licensed under the **MIT License**.  

---

⭐ **If you found this project useful, please give it a star!** ⭐  
```

You can **copy and paste** this directly into your `README.md` file. Let me know if you need any changes! 🚀
