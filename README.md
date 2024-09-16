# Hate Speech Detection in Memes Using Multimodal Machine Learning Models

## Project Overview

This project focuses on detecting hate speech in memes by leveraging both text and image data. Using advanced machine learning techniques, we aim to identify hateful content within memes through the analysis of both visual and textual cues. The models deployed in this project include CNN-based models for image processing, as well as TF-IDF vectorization and feedforward neural networks for text analysis. A combined multimodal model was also implemented to improve detection accuracy by integrating both text and image features.

## Ethical Considerations

Due to the sensitive nature of the Facebook Hate Meme Dataset, strict ethical standards were maintained to ensure privacy compliance and to respect individual rights. The dataset’s use in research was governed by principles aimed at promoting inclusivity and safety in online spaces. Researchers are urged to handle the dataset responsibly to avoid misuse and to uphold social responsibility.

## Modeling Approach

### 1. Image-Only Model
- **Model:** Convolutional Neural Network (CNN) with Autoencoder
- **Justification:** CNNs excel in image processing tasks and help identify subtle features such as symbols or text in memes. The Autoencoder improves feature extraction by reducing dimensionality.
- **Preprocessing:** Images were resized, normalized, and augmented with rotation, flipping, and slight color adjustments to increase the model's robustness.

### 2. Text-Only Model
- **Model:** TF-IDF Vectorization with Feedforward Neural Network
- **Justification:** TF-IDF was chosen for efficiency and simplicity, highlighting the most relevant terms in detecting hate speech. Truncated SVD was used for dimensionality reduction to make training manageable.
- **Preprocessing:** Text data underwent lemmatization, vectorization via TF-IDF, and SVD for dimensionality reduction.

### 3. Combined Text-Image Model
- **Model:** Dense Neural Network integrating pre-trained image features and TF-IDF for text
- **Justification:** Pre-trained image models were used to leverage existing feature extraction capabilities. The text features were combined using a dense layer for better computational efficiency.
- **Preprocessing:** Standard resizing, normalization for images, and tokenization, TF-IDF vectorization, and padding for text were applied to ensure consistency.

## Evaluation Metrics

### Key Metrics:
- **Accuracy:** Measures overall correct predictions.
- **AUC-ROC:** Evaluates the model’s ability to distinguish between positive and negative classes across various thresholds.

### Justification of Metrics:
- Accuracy provides an overall performance snapshot, though it is limited when dealing with imbalanced datasets.
- AUC-ROC assesses the model’s discriminative power, crucial in tuning thresholds based on practical or regulatory needs.

## Model Performance

The table below summarizes the performance of each model:

| **Model**                      | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **AUC**  |
|---------------------------------|--------------|---------------|------------|--------------|----------|
| Text-based Neural Network       | 69.41%       | 64.79%        | 36.80%     | 46.94%       | 62.59%   |
| Combined Text and Image Model   | 65.12%       | 52.68%        | 50.24%     | 51.43%       | 66.25%   |
| Image-based Convolutional Model | 58.71%       | 40.41%        | 48.40%     | 37.99%       | 53.62%   |
| Autoencoder with Classification | 51.15%       | 50.00%        | 11.07%     | 18.13%       | 50.25%   |

### Analysis:
- The **text-based model** showed strong precision but lower recall, possibly due to dataset imbalances.
- The **combined model** balanced precision and recall, but it faced challenges in effectively integrating text and image features.
- The **image-based CNN** struggled with low AUC, potentially due to image data complexity or model configuration issues.
- The **autoencoder model** had the weakest performance, likely due to feature loss during dimensionality reduction.

## Comparative Analysis

- **Image-Based Models:** Our results align partially with those of Sabat et al. (2021), suggesting room for improvement in feature extraction or model configuration.
- **Text-Based Models:** Advanced NLP techniques like those proposed by Bertens et al. (2021) could improve recall by better capturing nuances like sarcasm or indirect hate speech.
- **Combined Models:** Inspired by Kumar and Sachdeva (2022), further integration of text and image features using more sophisticated merging strategies could enhance the model’s effectiveness.

## Recommendations for Improvement

- **Enhanced Feature Extraction:** Incorporate more robust pre-trained models for both text and image features.
- **Advanced NLP Techniques:** Use methods like BERT to better handle nuanced language features.
- **Autoencoder Optimization:** Adjust the architecture to preserve more relevant features during dimensionality reduction.

## Future Research

The project opens up several opportunities for further research:
- Experiment with **deep learning transformers** for capturing complex contextual relationships.
- Evaluate the model’s adaptability across various platforms with different content dynamics.
- Investigate **real-time application** of these models in live moderation settings.
- Explore more advanced methods of **multimodal integration**, such as attention mechanisms or multimodal transformers.

## References
1. Sabat, et al., "Exploring CNNs for Offensive Content Detection in Images," Journal of Machine Learning Research, 2021.
2. Zhao, and Mao, "Enhancing Image Feature Extraction with Pre-trained CNNs," Journal of Visual Communication, 2019.
3. Bertens, et al., "Improving Hate Speech Detection with LSTMs and NLP," IEEE Transactions on Affective Computing, 2021.
4. Kumar, and Sachdeva, "Integrating RNNs and CNNs for Multimodal Content Analysis," IEEE Transactions on Multimedia, 2022.
5. Li, and Huang, "Autoencoders for Improved Multimodal Classification," Pattern Recognition Letters, 2020.
