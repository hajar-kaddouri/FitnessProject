

# ## Step 3:  Flask API to Use the Preprocessor



from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load the model and preprocessor
model = joblib.load('model_work.pkl')
preprocessor = joblib.load('preprocessor.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.get_json()
        df = pd.DataFrame([json_data])
        # Preprocess the data using the loaded preprocessor
        df_preprocessed = preprocessor.transform(df)
        # Predict using the loaded model
        prediction = model.predict(df_preprocessed)
        # Convert prediction to response, possibly decoding labels
        response = {'prediction': prediction.tolist()}
        return jsonify(response)
    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    try:
        app.run(debug=False, use_reloader=False)
    except Exception as e:
        print("Caught an exception in Flask app:", e)
        traceback.print_exc()







