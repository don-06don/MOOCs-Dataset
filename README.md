# 📚 MOOCs Urgency Prediction Project

This project focuses on predicting the urgency of posts in MOOCs (Massive Open Online Courses) forums based on text content, metadata, and post statistics. It includes data preprocessing, correlation analysis, encoding, scaling, and training multiple machine learning models (KNN, Random Forest, SVM).

## 🗂️ Dataset Overview

The dataset contains the following columns:
  Text: The content of the post
  
  Urgency(1-7): Original urgency rating (1 = low, 7 = high)
  
  CourseType: Category of the course
  
  course_display_name: Course name
  
  post_type: Type of post (e.g., question, announcement)
  
  up_count: Number of upvotes
  
  reads: Number of times the post was read


## ✂️ Data Preprocessing

Rows without text were removed since text is the core feature.

Text, CourseType, and post_type were merged into a single column called merged to be used for embeddings.

The Urgency(1-7) column was converted into a binary target called Urgency_binary: 0 for low urgency (<4) and 1 for high urgency (>=4).

Categorical columns (CourseType, post_type, course_display_name) were encoded into numeric values for modeling.


## 📊 Correlation Analysis

We analyzed numeric relationships between features using correlation heatmaps.

Observations: Urgency(1-7) has a strong positive correlation with post_type and moderate correlation with CourseType. 

The encoded course_display_name has weak correlations but contributes as a categorical feature.

Scaling using StandardScaler does not affect correlations, only feature magnitudes.


## ⚙️ Feature Scaling

StandardScaler was applied to numeric features and embeddings.

Scaling is important for distance-based models such as KNN and SVM to ensure features are comparable.


## 🤖 Text Embedding

We used the all-MiniLM-L6-v2 model from sentence-transformers to convert text into 384-dimensional embeddings.

This model is lightweight yet effective for short feedback text.

Embeddings were normalized to improve performance in machine learning models and to capture semantic meaning.


## 🔀 Train-Test Split & SMOTE

The dataset was split into training (70%) and testing (30%) sets with stratification to maintain class balance.

SMOTE was applied on the training set to handle class imbalance and improve performance on high-urgency posts.


## 🌟 Models

-> K-Nearest Neighbors (KNN): Trained using combined numeric features and embeddings. Hyperparameters were tuned using GridSearchCV. Performance was moderate and struggled with the minority class.

-> Random Forest (RF): Trained on resampled data with class_weight='balanced'. Provided a strong baseline and handled class imbalance better than KNN.

-> Support Vector Machine (SVM): Used scaled embeddings as input. Hyperparameters tuned with GridSearchCV optimizing F1-score. Performed well on high-dimensional embeddings and offered robust comparison with Random Forest.


## 📈 Model Evaluation

Evaluation metrics included Accuracy, Precision, Recall, and F1-score.

High-urgency posts remain challenging due to imbalance.

Embeddings improved predictive performance by capturing semantic information.

Random Forest and SVM complement each other, while KNN is simpler but less effective on imbalanced data.


## 📝 Notes

The project is modular and fully reproducible in Google Colab.

All steps, including text preprocessing, feature engineering, scaling, and model training, are self-contained.

The pipeline can be adapted for other text datasets with similar structure.

Embeddings, scaling, and model parameters can be modified to experiment with different models and improve performance.
