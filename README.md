# **Cosmic_Classifier**
Hi everyone! This project of ours focuses on classifying planets based on features like gravity, temperature, and water content (Classification Model). I and my team built nine models—three each with LightGBM, Neural Networks, and XGBoost—and combined them into a single soft-voting ensemble model to improve prediction accuracy.

# 🚀 **Cosmic Classifier Ensemble Project** 🚀
**Galactic Planet Classification - Hackathon Submission**

Hey everyone! So, this is my submission for the Galactic Planet Classification hackathon. Basically, I built 9 different machine learning models and combined them into one ensemble to make the predictions better. I'll try my best to explain how I did it step-by-step, especially the code part.

# 📚**About the Models (All 9 of Them)**

I made three different kinds of models (LightGBM, Neural Network, and XGBoost). Each type has three files—so, overall, that's 9 models.

# 🔦**LightGBM Models (3 Files):**

Did preprocessing by filling missing values with the average (mean) and added polynomial features to create new data.

Used Label Encoding (making labels numeric).
Trained the model using Optuna (just fancy hyperparameter tuning stuff).

**Files saved:**
```
final_model_lightgbm.joblib

lgb_preprocessing_pipeline.joblib

lgb_label_encoder.joblib

lightgbm_trainingcode.ipynb
```

# 🧠 **Neural Network Models (3 Files):**

Preprocessed the data using KNN (to fill missing values), added polynomial features (degree=3), and standardized it (mean=0, variance=1).

Used Label Encoder again to turn labels into numbers.

Built neural networks (with multiple layers, dropout layers, and batch normalization) and tuned their hyperparameters like dropout, learning rate, number of neurons, and activation functions with Optuna.

**Files saved:**
```
final_model_cosmicclassifier.h5

preprocessing_pipeline.pkl

label_encoder.pkl
```

# 🌳 **XGBoost Models (3 Files):**
Used SMOTE (this cool method to balance out classes) and scaled the features.
Did Label Encoding for labels.
Did hyperparameter tuning with Optuna for parameters like eta, max depth, gamma, subsample, and stuff like that.

**Files saved:**
```
final_model_xgboost_real.joblib

xgb_preprocessing_pipeline_real.joblib

xgb_label_encoder_real.joblib

xcb_trainingcode.ipynb
```

# **🛠️ The Main Thing: Ensemble Model (Soft Voting Classifier)**
Now, the main part of my project is combining the three different model types into one ensemble model. It's called a "Soft Voting Classifier" because it averages the predicted probabilities from each model.

# 🧐 **Step-by-step Explanation (the code part)**

**Loading Stuff (models and pipelines):**

First, loaded all three models, preprocessing pipelines, and label encoders from saved files.
Paths look something like this:
```
final_model_lightgbm = joblib.load("/kaggle/input/final_model_lightgbm/...etc")
final_model_nn = load_model("/kaggle/input/final_model_cosmicclassifier/...etc")
final_model_xgb = joblib.load("/kaggle/input/final_model_xgboost_real/...etc")
```
**Data Preparation (loading CSV):**

Loaded the test dataset (cosmicclassifierTest.csv) from Kaggle's input folder.
Also converted weird labels like "Category_3" into simple numeric values using a custom function:
```
def category_to_float(val):
    if isinstance(val, str) and val.startswith("Category_"):
        return float(val.replace("Category_", ""))
    return val
```
**Feature Processing (making data ready):**

Passed data through each model’s preprocessing pipeline separately.
```
X_lgb = lgb_preprocessing_pipeline.transform(X_new)
X_nn = nn_preprocessing_pipeline.transform(X_new)
X_xgb = xgb_preprocessing_pipeline.transform(X_new)
```

**Prediction Generation (getting probabilities):**

Each model gave their probability predictions separately.
Had this helper function called reorder_probability_columns() to align predictions correctly. It helps especially when there are NaN values or different orders of classes:
```
probs_lgb = reorder_probability_columns(probs_lgb_raw, lgb_label_encoder.classes_, nn_label_encoder.classes_)
```
(did the same for XGB model predictions)

**Soft Voting (combining probabilities):**

Averaged the probabilities from the three models to make final predictions:
```
ensemble_probs = (probs_lgb + probs_nn + probs_xgb) / 3.0
ensemble_preds_encoded = np.argmax(ensemble_probs, axis=1)
```
**Saving Predictions to CSV:**

Translated numeric predictions back to original class names and made a CSV file:

```
ensemble_preds_labels = nn_label_encoder.inverse_transform(ensemble_preds_encoded)
submission = pd.DataFrame({
    "Planet_ID": df_new["Planet_ID"],
    "Final_Ensemble_Prediction": ensemble_preds_labels
})
submission.to_csv("ensemble_submission.csv", index=False)
```
# ⚙️ **How to Run it on Kaggle (step-by-step):**

**Step 1:**
Upload all the saved model files (.joblib, .h5, .pkl) to Kaggle's input folder.
```
For example: /kaggle/input/final_model_lightgbm/...
```

**Step 2:** 
Upload your test data (cosmicclassifierTest.csv) also into Kaggle’s input folder: 
```
/kaggle/input/testing-data-cogni1/cosmicclassifierTest.csv
```

**Step 3:** Run the ensemble Python script under **finalfinalfinalcosmic.ipynb** provided in Kaggle notebook environment.

**Step 4:**
Download the ensemble_submission.csv file and submit it directly on Kaggle.

# 📈 **Checking How Well It Did (Evaluation):**
If your test data already has the correct labels in it, the script automatically shows accuracy and a classification report (precision, recall, F1-score, etc.) right on the console.

**Accuracy on testing data was 93.6 %**
