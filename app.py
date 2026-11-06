from flask import Flask, render_template, request
import pandas as pd
import joblib

# This section imports the libraries needed for the web application. Flask is used
# to create the web server and connect the Python prediction system to the HTML page.
# render_template is used to display the webpage, while request is used to collect
# the values entered by the user in the form. Pandas is used to organise the user's
# input into the same structure as the training data, and joblib is used to load
# the previously trained machine-learning model from the .pkl file.

app = Flask(__name__)
prediction_model = joblib.load("sleep_disorder_model.pkl")

# This section creates the Flask application and loads the trained prediction model.
# Flask(__name__) creates the web application that will handle requests from the user.
# joblib.load() opens the saved sleep_disorder_model.pkl file that was created by
# train_model.py. This means the web application can use the trained model directly
# without repeating the model training process every time the application is started.

@app.route("/", methods=["GET", "POST"])
def home():

    prediction_result = None

    if request.method == "POST":

        gender = request.form["gender"]
        age = float(request.form["age"])
        occupation = request.form["occupation"]
        sleep_duration = float(request.form["sleep_duration"])
        quality_of_sleep = float(request.form["quality_of_sleep"])
        physical_activity = float(request.form["physical_activity"])
        stress_level = float(request.form["stress_level"])
        bmi_category = request.form["bmi_category"]
        heart_rate = float(request.form["heart_rate"])
        daily_steps = float(request.form["daily_steps"])
        systolic_pressure = float(request.form["systolic_pressure"])
        diastolic_pressure = float(request.form["diastolic_pressure"])

        # This section creates the main route for the application. When the webpage
        # is first opened, the route displays the form using a normal GET request.
        # When the user presses the prediction button, Flask receives a POST request
        # containing all values entered in the HTML form. Text-based values such as
        # gender, occupation, and BMI category are stored as strings, while numerical
        # values are converted to float so that they can be processed correctly by
        # the machine-learning pipeline.

        user_input = pd.DataFrame([{
            "Gender": gender,
            "Age": age,
            "Occupation": occupation,
            "Sleep Duration": sleep_duration,
            "Quality of Sleep": quality_of_sleep,
            "Physical Activity Level": physical_activity,
            "Stress Level": stress_level,
            "BMI Category": bmi_category,
            "Heart Rate": heart_rate,
            "Daily Steps": daily_steps,
            "Systolic Pressure": systolic_pressure,
            "Diastolic Pressure": diastolic_pressure
        }])

        # This section places the information entered by the user into a Pandas
        # DataFrame. The column names are written exactly the same way as the features
        # used when the machine-learning model was trained. This is important because
        # the saved preprocessing pipeline expects the same input structure. Creating
        # a one-row DataFrame allows the new user information to be processed in the
        # same way as the original training records.

        prediction_result = prediction_model.predict(user_input)[0]

        # This section sends the prepared user information to the trained model.
        # The predict() function processes the new values using the same preprocessing
        # steps and classification model that were saved during training. Because only
        # one person's information is being submitted, [0] retrieves the first and only
        # predicted category. The result will be one of three possible values:
        # No Disorder, Insomnia, or Sleep Apnea.

    return render_template(
        "index.html",
        prediction=prediction_result
    )

    # This section sends the index.html page to the user's browser. The prediction
    # variable is also passed to the webpage so that the result can be displayed after
    # the user submits the form. Before a prediction is made, prediction_result remains
    # None, so the webpage can simply display the input form without a result.


if __name__ == "__main__":
    app.run(debug=True)

# This section starts the Flask development server when app.py is run directly.
# debug=True is useful during development because Flask automatically reloads the
# application when changes are made and provides clearer error information if
# something goes wrong while testing the webpage.