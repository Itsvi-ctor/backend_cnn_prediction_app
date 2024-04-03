# from flask import Flask, request, jsonify
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import cv2
# import numpy as np

# app = Flask(__name__)

# # Load the pre-trained model
# # C:\Users\Jimi2\OneDrive\Documents\400lvl\Project
# model_path = './model.keras'  # Update this to the path of your .keras file
# model = load_model(model_path)

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Receive the image file from the request
#     file = request.files['file']
#     # Convert the file to a numpy array
#     spectrogram = np.fromfile(file, np.uint8)
#     spectrogram = cv2.imdecode(spectrogram, cv2.IMREAD_UNCHANGED)
    
#     # Preprocess the image
#     preprocessed_image = preprocess_image(spectrogram)
    
#     # Make a prediction
#     prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
    
#     # Return the result
#     result = {'prediction': prediction.tolist()}
#     return jsonify(result)

# def preprocess_image(spectrogram):
#     # Your preprocessing code here
#     # Convert the grayscale spectrogram to RGBA by repeating the grayscale values across all four channels
#     spectrogram_rgba = cv2.cvtColor(spectrogram, cv2.COLOR_GRAY2RGBA)

#     # Resize the spectrogram
#     spectrogram_resized = cv2.resize(spectrogram_rgba, (1200, 600))

#     # Normalize pixel values
#     spectrogram_resized = spectrogram_resized.astype('float32') / 255
#     pass

# if __name__ == '__main__':
#     app.run(debug=True)

# !
# from flask import Flask, request, jsonify
# import librosa
# import numpy as np
# from tensorflow.keras.models import load_model
# import os
# import tempfile

# app = Flask(__name__)

# # Load your pre-trained model
# model = load_model('./new_model.keras')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Check if the POST request has the file part
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part in the request'}), 400
    
#     file = request.files['file']
    
#     # Save the file to a temporary file
#     temp = tempfile.NamedTemporaryFile(delete=False)
#     file.save(temp.name)
    
#     # Load the audio file and convert it to a spectrogram
#     audio, sample_rate = librosa.load(temp.name, sr=None)
#     spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    
#     # Preprocess the spectrogram for the model
# def preprocess_image(spectrogram):
#     # Your preprocessing code here
#     # Convert the grayscale spectrogram to RGBA by repeating the grayscale values across all four channels
#     spectrogram_rgba = cv2.cvtColor(spectrogram, cv2.COLOR_GRAY2RGBA)

#     # Resize the spectrogram
#     spectrogram_resized = cv2.resize(spectrogram_rgba, (1200, 600))

#     # Normalize pixel values
#     spectrogram_resized = spectrogram_resized.astype('float32') / 255
#     pass
    
#     # Make a prediction
#     prediction = model.predict(np.expand_dims(spectrogram, axis=0))
    
#     # Delete the temporary file
#     os.unlink(temp.name)
    
#     # Return the prediction result
#     return jsonify({'prediction': prediction.tolist()})

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import os
import tempfile
app = Flask(__name__)
model_path = './tensor.keras'
model = load_model(model_path)
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    # Save the file to a temporary file and ensure it is closed properly
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        file.save(temp.name)
        temp.close()  # Close the file explicitly
    
    # Load the audio file and convert it to a spectrogram
    audio, sample_rate = librosa.load(temp.name, sr=None)
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
    
    # Preprocess the spectrogram for the model
    preprocessed_image = preprocess_image(spectrogram_db)
    
    # Make a prediction
    prediction = predict_image(preprocessed_image)
    
    # Delete the temporary file
    os.unlink(temp.name)
    
    # Interpret the prediction
    result = "Healthy Chicken" if prediction[0][0] > 0.5 else "Unhealthy Chicken"
    
    # Return the prediction result
    return jsonify({'prediction': result})
def preprocess_image(spectrogram):
    if spectrogram.ndim == 3 and spectrogram.shape[2] == 1:
        spectrogram = spectrogram.reshape(spectrogram.shape[0], spectrogram.shape[1])
    spectrogram_rgba = np.stack((spectrogram,) * 4, axis=-1)
    spectrogram_resized = cv2.resize(spectrogram_rgba, (1200, 600))
    spectrogram_resized = spectrogram_resized.astype('float32') / 255
    return spectrogram_resized
def predict_image(spectrogram):
    preprocessed_image = np.expand_dims(spectrogram, axis=0)
    prediction = model.predict(preprocessed_image)
    return prediction
if __name__ == '__main__':
    app.run(debug=True)