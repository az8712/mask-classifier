# **Facial Recognition with Masks using Transfer Learning**

This project uses **transfer learning** with pre-trained models (Xception, VGG16, DenseNet121) for facial recognition to detect whether individuals are wearing masks. The models are fine-tuned for binary classification (with mask, without mask) using TensorFlow and Keras.

## **Project Overview**

This project aims to build a deep learning model to detect masks on faces using a dataset of images of people with and without masks. The models leverage pre-trained CNN architectures, such as **Xception**, **VGG16**, and **DenseNet121**, for transfer learning, significantly improving performance without needing to train from scratch.

## **Features**
- **Data Preprocessing**: Images are loaded, resized, and normalized for model input.
- **Transfer Learning**: Pre-trained models (Xception, VGG16, DenseNet121) are used with fine-tuning on the mask detection task.
- **Custom Training Loop**: A custom training function is implemented to handle data in batches and prevent memory overload.
- **Model Evaluation**: Performance is evaluated on test datasets with accuracy and other metrics.


## **Model Architectures**

This project uses three well-known CNN architectures for transfer learning:

- **Xception**: A deep CNN architecture with depthwise separable convolutions.
- **VGG16**: A 16-layer CNN architecture known for its simplicity and depth.
- **DenseNet121**: A densely connected network that connects each layer to every other layer, improving gradient flow and feature reuse.

Each model is trained with frozen early layers (to prevent re-training) and fine-tuned with a custom dense layer for binary classification.

## **Results**

| Model         | Test Accuracy |
|---------------|---------------|
| Xception      | 95%           |
| VGG16         | 93%           |
| DenseNet121   | 94%           |
