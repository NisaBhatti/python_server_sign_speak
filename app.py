from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64
import io
from PIL import Image
import tensorflow as tf
import os
import glob

app = Flask(__name__)
CORS(app)

# ============================================
# AUTO-LOAD ALL MODELS
# ============================================
MODELS = {}

# Find all *_robust.tflite files
model_files = glob.glob("*_robust.tflite")

for model_file in model_files:
    # Extract alphabet name from filename
    alphabet_name = model_file.replace("_robust.tflite", "")
    
    try:
        interpreter = tf.lite.Interpreter(model_path=model_file)
        interpreter.allocate_tensors()
        MODELS[alphabet_name] = {
            'interpreter': interpreter,
            'input_details': interpreter.get_input_details(),
            'output_details': interpreter.get_output_details()
        }
        print(f"✅ Loaded: {alphabet_name}")
    except Exception as e:
        print(f"❌ Failed to load {alphabet_name}: {e}")

print("=" * 60)
print(f"✅ TOTAL MODELS LOADED: {len(MODELS)}")
print(f"📚 Available: {list(MODELS.keys())}")
print("=" * 60)

# ============================================
# MEDIAPIPE HANDS
# ============================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("✅ MediaPipe Hands initialized!")

# ============================================
# ALPHABET MAPPING
# ============================================
ALPHABET_DISPLAY = {
    'alif': 'ا',
    'bay': 'ب',
    'tay': 'ت',
    'thay': 'ث',
    'seen': 'س',
    'sheen': 'ش',
    'suaad': 'ص',
    'zvad': 'ض',
    'toayn': 'ط',
    'zoyn': 'ظ',
    'ain': 'ع',
    'ghain': 'غ',
    'fe': 'ف',
    'quaaf': 'ق',
    'kaf': 'ك',
    'gaf': 'گ',
    'lam': 'ل',
    'mim': 'م',
    'noon': 'ن',
    'vao': 'و',
    'hamza': 'ء',
    'choti_ye': 'ی',
    'bari_ye': 'ے',
    'ray': 'ر',
    'rray': 'ڑ',
    'zay': 'ز',
    'dal': 'د',
    'daal': 'ڈ',
    'zal': 'ذ',
    'khay': 'خ',
    'rre': 'ڑ',
}

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'OK', 'models': list(MODELS.keys())})

@app.route('/models', methods=['GET'])
def get_models():
    models_info = []
    for name in MODELS.keys():
        models_info.append({
            'name': name,
            'display': ALPHABET_DISPLAY.get(name, name),
            'arabic': ALPHABET_DISPLAY.get(name, '?')
        })
    return jsonify({'models': models_info, 'count': len(models_info)})

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        image_data = data.get('image')
        alphabet = data.get('alphabet', 'alif')
        
        if not image_data:
            return jsonify({'error': 'No image data'}), 400

        # Decode image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Process with MediaPipe
        rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            return jsonify({
                'hasHand': False,
                'isAlphabet': False,
                'confidence': 0.0,
                'landmarks': [],
                'alphabet': alphabet,
                'display': ALPHABET_DISPLAY.get(alphabet, alphabet),
                'message': 'No hand detected'
            })

        # Extract landmarks
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

        features = np.array(landmarks, dtype=np.float32).reshape(1, -1)
        
        # Check if model exists
        if alphabet not in MODELS:
            return jsonify({
                'hasHand': True,
                'isAlphabet': False,
                'confidence': 0.0,
                'landmarks': landmarks,
                'alphabet': alphabet,
                'display': ALPHABET_DISPLAY.get(alphabet, alphabet),
                'message': f'Model for {alphabet} not found'
            })

        # Run model
        model_data = MODELS[alphabet]
        interpreter = model_data['interpreter']
        input_details = model_data['input_details']
        output_details = model_data['output_details']
        
        interpreter.set_tensor(input_details[0]['index'], features)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        score = float(prediction[0][0])
        is_alphabet = score > 0.5

        return jsonify({
            'hasHand': True,
            'isAlphabet': is_alphabet,
            'confidence': score,
            'landmarks': landmarks,
            'alphabet': alphabet,
            'display': ALPHABET_DISPLAY.get(alphabet, alphabet),
            'message': 'Success'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 ALPHABET DETECTION SERVER")
    print("=" * 60)
    print(f"📚 Loaded {len(MODELS)} alphabet models")
    print(f"📡 Server: http://0.0.0.0:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)