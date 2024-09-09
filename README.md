### Project Title: **Face Detection and Classification Using Deep Learning**


### Project Overview:
This project involves building a machine learning pipeline to classify images into two categories: **showing face** and **covering face**. The project utilizes deep learning models to perform classification and includes comprehensive preprocessing, training, validation, and performance logging mechanisms.

### Key Components:
1. **Data Preprocessing**:
   - Images are read from a dataset, resized to a uniform size, and preprocessed using Gaussian blur and other augmentations (rotation, flipping).
   - The dataset is split into training, validation, and test sets based on configurable ratios.

2. **Deep Learning Models**:
   - Multiple pre-trained models are fine-tuned for the task:
     - **GoogLeNet**
     - **MobileNetV2**
     - **ResNet**
   - These models are trained using `torchvision` and `torch` libraries, leveraging GPU acceleration where available.

3. **Training and Evaluation**:
   - A custom training loop is implemented to train models on the training dataset and validate their performance on a validation set.
   - The loop also tracks and logs key metrics such as loss, validation loss, accuracy, and the time taken per epoch.
   - During evaluation, metrics such as sensitivity, specificity, weighted error, and accuracy are calculated both per class and overall.

4. **Logging**:
   - A dedicated logging mechanism is used to record the process of training, validation, and other key operations. 
   - The logs provide insights into the model’s performance and capture errors during execution.

5. **Error Handling**:
   - The project implements error-handling mechanisms to manage potential issues such as missing folders, incorrect directory structures, and invalid file paths.

6. **Model Voting System**:
   - A voting classifier integrates the predictions of the three models (GoogLeNet, MobileNetV2, and ResNet) to improve classification performance.
   - This ensemble approach provides a more robust classification by averaging the predictions from the individual models.

### Project Outcome:
The result of this project is a robust face detection and classification system that can be used for real-world applications such as security systems, ATM user verification, and monitoring.

### Tools and Libraries:
- **Python**, **PyTorch**, **Torchvision**
- **OpenCV** for image processing
- **Logging** for detailed logging of training and evaluation
- **scikit-learn** for data splitting and potential model evaluation
