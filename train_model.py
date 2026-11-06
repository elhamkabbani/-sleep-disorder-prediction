import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, classification_report

sleep_data = pd.read_excel("Sleep_health_and_lifestyle_dataset.xlsx")
print(sleep_data.head())

# This section loads the Sleep Health and Lifestyle dataset from the Excel file
# and stores it in the variable sleep_data. The head() function is then used to
# display the first five rows in the terminal. This helps confirm that the dataset
# has been loaded successfully and allows the structure, column names, and example
# values to be checked before any cleaning or preprocessing is performed.

sleep_data["Sleep Disorder"] = sleep_data["Sleep Disorder"].fillna("No Disorder")
sleep_data = sleep_data.drop(columns=["Person ID"])

# This section prepares the dataset before it is used for machine learning.
# The blank values in the Sleep Disorder column represent people who do not
# have a recorded sleep disorder, so they are replaced with the clearer label
# "No Disorder". The Person ID column is then removed because it is only an
# identification number and does not provide useful information for predicting
# whether a person has Insomnia, Sleep Apnea, or No Disorder.

sleep_data[["Systolic Pressure", "Diastolic Pressure"]] = (
    sleep_data["Blood Pressure"].str.split("/", expand=True).astype(int)
)

sleep_data = sleep_data.drop(columns=["Blood Pressure"])

# This section separates the Blood Pressure column into two numerical features:
# Systolic Pressure and Diastolic Pressure. The original values are stored in a
# combined format such as "120/80", which is not directly suitable for most
# machine-learning models. Splitting the value allows each pressure measurement
# to be processed independently as numerical data. The original Blood Pressure
# column is then removed because its information is now represented by the two
# new columns.

X = sleep_data.drop(columns=["Sleep Disorder"])
y = sleep_data["Sleep Disorder"]

# This section separates the dataset into the information used for prediction
# and the value that the machine-learning models are trying to predict.
# X contains all input features, such as age, sleep duration, stress level,
# occupation, heart rate, daily steps, and the new blood pressure columns.
# y contains only the Sleep Disorder column, which is the target variable.
# The models will learn relationships between the values in X and the correct
# sleep disorder category stored in y. X = the clues
#y = the correct answer

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# This section divides the dataset into two parts. Eighty percent of the records
# are used as training data so the machine-learning models can learn patterns
# between the input features and the correct Sleep Disorder category. The remaining
# twenty percent are kept aside as testing data so the models can later be evaluated
# using records they have not seen during training. random_state=42 keeps the split
# consistent each time the program runs, while stratify=y helps preserve a similar
# proportion of No Disorder, Insomnia, and Sleep Apnea cases in both sets.

numeric_features = [
    "Age",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "Heart Rate",
    "Daily Steps",
    "Systolic Pressure",
    "Diastolic Pressure"
]

categorical_features = [
    "Gender",
    "Occupation",
    "BMI Category"
]

# This section separates the input features into numerical and categorical groups
# because the two types of data need to be prepared differently before model training.
# Numerical features contain measurable values such as age, sleep duration, heart rate,
# daily steps, and blood pressure. Categorical features contain text-based categories
# such as gender, occupation, and BMI category. Defining these groups allows the
# preprocessing stage to apply the correct transformation to each type of data.

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# This section creates the preprocessing system that prepares the input data before
# it is sent to the machine-learning models. The numerical features are processed
# using StandardScaler, which places numerical values on a similar scale so that
# features with larger values, such as Daily Steps, do not dominate features with
# smaller values, such as Sleep Duration. The categorical features are processed
# using OneHotEncoder, which converts text values such as Gender, Occupation, and
# BMI Category into numerical form that the models can understand. The option
# handle_unknown="ignore" prevents errors if the system later receives a category
# that was not present in the training data.

prediction_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbours": KNeighborsClassifier()
}

# This section creates the five classification models that will be compared during
# the predictive analysis. Each model uses a different machine-learning method to
# learn patterns between the lifestyle and health information and the Sleep Disorder
# target. The models are stored together in a dictionary called prediction_models,
# which allows the program to train and evaluate them using the same process instead
# of writing separate training code for every model. random_state=42 is used where
# supported to make the results consistent when the program is run again.

model_scores = {}

for model_name, model in prediction_models.items():

    model_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("classifier", model)
        ]
    )

    validation_scores = cross_val_score(
        model_pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    model_scores[model_name] = validation_scores.mean()

    print(
        model_name,
        "- Cross-validation accuracy:",
        round(validation_scores.mean(), 4)
    )

# This section evaluates each of the five machine-learning models using the same
# training data. A Pipeline is created for every model so that the preprocessing
# steps are automatically applied before classification. This means that numerical
# values are scaled and categorical values are encoded before the selected model
# receives the data.
#
# cross_val_score performs five-fold cross-validation. Instead of measuring a model
# using only one portion of the training data, Round 1:
#Train using Parts 1,2,3,4
#Check using Part 5
#Round 2:
#Train using Parts 1,2,3,5
#Check using Part 4

best_model_name = max(model_scores, key=model_scores.get)
selected_model = prediction_models[best_model_name]

print("\nBest model:", best_model_name)
print(
    "Best cross-validation accuracy:",
    round(model_scores[best_model_name], 4)
)

# This section identifies which of the five models achieved the highest average
# cross-validation accuracy. The max() function searches the model_scores dictionary
# and finds the model associated with the highest score. The name of that model is
# stored in best_model_name, while the actual machine-learning model is stored in
# selected_model. This model will then be trained again and used as the final model
# for the sleep disorder prediction system.

final_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", selected_model)
    ]
)

final_pipeline.fit(X_train, y_train)
test_predictions = final_pipeline.predict(X_test)
# This section creates the final prediction pipeline using the model that achieved
# the best cross-validation performance. The pipeline combines the same preprocessing
# steps with the selected classifier. The fit() function then trains the complete
# pipeline using the 80% training portion of the dataset. After training is complete,
# predict() is used with X_test to generate sleep disorder predictions for the 20%
# testing records that the model did not use during its final training.

test_accuracy = accuracy_score(y_test, test_predictions)

print("\nFinal Test Accuracy:", round(test_accuracy, 4))
print("\nClassification Report:")
print(classification_report(y_test, test_predictions))

# This section evaluates how accurately the selected model performs on the testing
# data. accuracy_score compares the model's predictions with the real Sleep Disorder
# values stored in y_test and calculates the proportion of correct predictions.
# classification_report provides additional information about how effectively the
# model predicted each individual category, including No Disorder, Insomnia, and
# Sleep Apnea. Using previously unseen testing data provides a more realistic check
# of whether the trained model can make useful predictions beyond the records it
# learned from.

joblib.dump(final_pipeline, "sleep_disorder_model.pkl")

print("\nThe trained model has been saved as sleep_disorder_model.pkl")

# This section saves the complete trained pipeline into a .pkl file using joblib.
# The saved file contains both the preprocessing steps and the selected prediction
# model, meaning that new user information can later be processed and predicted using
# the same system without training the models again. The Flask web application will
# load this file when a user submits information through the prediction webpage.