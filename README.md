#End to end ANN project

# Bank Customer Churn Prediction using Deep Learning

## Project Overview
This project demonstrates the development of an end-to-end deep learning pipeline using **TensorFlow** and **Keras**. The goal is to predict customer churn for a bank based on historical data, employing artificial neural networks (ANNs) for binary classification.

### Deployment Link
You can access the deployed application here: [Streamlit App](https://lxqyfwms5yivekxnmrts4u.streamlit.app/)

---

## Features
- **Data Preprocessing**:
  - Dropped irrelevant features like `RowNumber`, `CustomerID`, and `Surname`.
  - Converted categorical variables (`Geography` and `Gender`) using **One-Hot Encoding** and **Label Encoding**.
  - Standardized numerical features using **StandardScaler**.

- **Model Architecture**:
  - **Input Layer**: Processes 11 input features.
  - **Hidden Layers**: Includes multiple layers with ReLU activation functions.
  - **Output Layer**: Uses a sigmoid activation function for binary classification.

- **Training**:
  - Optimized using **Adam Optimizer** with binary cross-entropy as the loss function.
  - Early stopping and TensorBoard callbacks for efficient training and monitoring.

- **Model Saving**:
  - Saved the trained model in `.h5` format.
  - Pickled preprocessing encoders and scalers for deployment compatibility.

- **Deployment**:
  - Built a Streamlit web application for model inference.
  - Integrated pickled files for consistent preprocessing during predictions.

---

## Libraries Used
- **TensorFlow (2.x)** and **Keras**: For building and training the neural network.
- **Pandas**: For data manipulation.
- **NumPy**: For numerical computations.
- **scikit-learn**: For preprocessing, feature scaling, and train-test splitting.
- **Matplotlib**: For visualization.
- **Streamlit**: For deploying the web application.

---

## Steps to Reproduce
1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Set Up the Environment**:
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - (Optional) Create a virtual environment:
     ```bash
     conda create -n churn_env python=3.11
     conda activate churn_env
     ```

3. **Train the Model**:
   - Preprocess the dataset (`churn_modeling.csv`).
   - Train the neural network using the script:
     ```bash
     python train_model.py
     ```

4. **Deploy the App**:
   - Run the Streamlit application:
     ```bash
     streamlit run app.py
     ```

---

## Dataset
The dataset `churn_modeling.csv` includes the following features:
- `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`.
- Target: `Exited` (1 if the customer left the bank, 0 otherwise).

---



---


## Contributors
This project was developed as part of a hands-on deep learning tutorial series.

---

If there are additional details or refinements you'd like, let me know! 😊
