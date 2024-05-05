from flask import Flask, render_template, request
import cv2
import numpy as np 
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.preprocessing import LabelEncoder

import google.generativeai as genai

app = Flask(__name__)   

model = load_model('./model/resnet.h5')
le = LabelEncoder()
label_names = ['Eczema', 'Warts Molluscum and other Viral Infections', 'Melanoma', 'Atopic Dermatitis', 'Basal Cell Carcinoma', 'Melanocytic Nevi', 'Benign Keratosis-like Lesions', 'Psoriasis pictures Lichen Planus ', 'Seborrheic Keratoses and other Benign Tumors ', 'Tinea Ringworm Candidiasis and other Fungal Infections']

genai.configure(api_key="AIzaSyBNx1NABZ6sqk1OBzvVoWrx2W8OgxFi3-U")  # Replace YOUR_API_KEY with your actual API key

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 0,
  "max_output_tokens": 8192,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'imagefile' not in request.files:
            return render_template('index.html', error='No file selected')
        
        imagefile = request.files['imagefile']
        if imagefile.filename == '':
            return render_template('index.html', error='No file selected')
        
        image_path = "./images/" + imagefile.filename
        imagefile.save(image_path)

        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        image = preprocess_input(np.array([image]))
        predictions = model.predict(image)
        predicted_class_index = np.argmax(predictions)

        # Generate response from Gemini API
        convo_model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)
        convo = convo_model.start_chat(history=[])
        convo.send_message("Suggest some basic cure for " + label_names[predicted_class_index] +"in less than 100 words")
        response = convo.last.text

        return render_template('index.html', predictions=label_names[predicted_class_index], response=response)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
