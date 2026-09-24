from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image
import tensorflow as tf
import os
import glob

# ============================================
# MEDIAPIPE HANDS - FIXED IMPORT FOR NEWER VERSIONS
# ============================================
import mediapipe as mp

# Try new import path (mediapipe >= 0.10.15)
try:
    from mediapipe.python.solutions import hands as mp_hands_module
    print("✅ Using mediapipe.python.solutions.hands")
except Exception as e1:
    print(f"⚠️ Primary import failed: {e1}")
    try:
        import mediapipe.python.solutions as mp_solutions
        mp_hands_module = mp_solutions.hands
        print("✅ Using mediapipe.python.solutions fallback")
    except Exception as e2:
        print(f"⚠️ Secondary import failed: {e2}")
        mp_hands_module = mp.solutions.hands
        print("✅ Using legacy mp.solutions.hands")

app = Flask(__name__)
CORS(app)

# ============================================
# AUTO-LOAD ALL MODELS
# ============================================
MODELS = {}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_files = glob.glob(os.path.join(BASE_DIR, "*_robust.tflite"))

print("=" * 60)
print(f"🔍 Model dir: {BASE_DIR}")
print(f"📦 Found {len(model_files)} models")
print("=" * 60)

for model_file in model_files:
    alphabet_name = os.path.basename(model_file).replace("_robust.tflite", "")
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
        print(f"❌ Failed: {alphabet_name}: {e}")

print("=" * 60)
print(f"✅ TOTAL: {len(MODELS)}")
print("=" * 60)

# ============================================
# INITIALIZE HANDS
# ============================================
hands = mp_hands_module.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("✅ MediaPipe Hands initialized!")

ALPHABET_DISPLAY = {
    'alif': 'ا', 'bay': 'ب', 'tay': 'ت', 'thay': 'ث',
    'seen': 'س', 'sheen': 'ش', 'suaad': 'ص', 'zvad': 'ض',
    'toayn': 'ط', 'zoyn': 'ظ', 'ain': 'ع', 'ghain': 'غ',
    'fe': 'ف', 'quaaf': 'ق', 'kaf': 'ك', 'gaf': 'گ',
    'lam': 'ل', 'mim': 'م', 'noon': 'ن', 'vao': 'و',
    'hamza': 'ء', 'choti_ye': 'ی', 'bari_ye': 'ے', 'ray': 'ر',
    'rray': 'ڑ', 'zay': 'ز', 'dal': 'د', 'daal': 'ڈ',
    'zal': 'ذ', 'khay': 'خ', 'rre': 'ڑ',
}

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'OK', 'models': list(MODELS.keys())})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'models_count': len(MODELS)})

@app.route('/models', methods=['GET'])
def get_models():
    return jsonify({
        'models': [
            {'name': n, 'display': ALPHABET_DISPLAY.get(n, n), 'arabic': ALPHABET_DISPLAY.get(n, '?')}
            for n in MODELS.keys()
        ],
        'count': len(MODELS)
    })

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        image_data = data.get('image')
        alphabet = data.get('alphabet', 'alif')

        if not image_data:
            return jsonify({'error': 'No image'}), 400

        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            return jsonify({
                'hasHand': False, 'isAlphabet': False, 'confidence': 0.0,
                'landmarks': [], 'alphabet': alphabet,
                'display': ALPHABET_DISPLAY.get(alphabet, alphabet),
                'message': 'No hand'
            })

        landmarks = []
        for hl in results.multi_hand_landmarks:
            for lm in hl.landmark:
                landmarks.extend([lm.x, lm.y])

        if alphabet not in MODELS:
            return jsonify({
                'hasHand': True, 'isAlphabet': False, 'confidence': 0.0,
                'landmarks': landmarks, 'alphabet': alphabet,
                'display': ALPHABET_DISPLAY.get(alphabet, alphabet),
                'message': f'Model {alphabet} not found'
            })

        features = np.array(landmarks, dtype=np.float32).reshape(1, -1)
        m = MODELS[alphabet]
        m['interpreter'].set_tensor(m['input_details'][0]['index'], features)
        m['interpreter'].invoke()
        pred = m['interpreter'].get_tensor(m['output_details'][0]['index'])
        score = float(pred[0][0])

        return jsonify({
            'hasHand': True, 'isAlphabet': score > 0.5, 'confidence': score,
            'landmarks': landmarks, 'alphabet': alphabet,
            'display': ALPHABET_DISPLAY.get(alphabet, alphabet),
            'message': 'Success'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
