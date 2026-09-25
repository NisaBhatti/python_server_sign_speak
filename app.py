import os
import gc

# ============================================
# MEMORY OPTIMIZATION - BEFORE TF/MP IMPORTS
# ============================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import io
import glob
from PIL import Image
import tensorflow as tf

# ============================================
# MEDIAPIPE TOGGLE
# ============================================
# Set to True ONLY when server has 2 GB+ RAM
# Keep False to keep server stable (dummy landmarks)
USE_MEDIAPIPE = False

hands = None
if USE_MEDIAPIPE:
    try:
        import mediapipe as mp
        try:
            from mediapipe.python.solutions import hands as mp_hands_module
        except ImportError:
            mp_hands_module = mp.solutions.hands
        hands = mp_hands_module.Hands(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ MediaPipe Hands initialized (lite mode)")
    except Exception as e:
        print(f"⚠️ MediaPipe failed: {e}")
        hands = None
        USE_MEDIAPIPE = False
else:
    print("⚠️ MediaPipe DISABLED — using fallback dummy landmarks")

app = Flask(__name__)
CORS(app)

# ============================================
# LAZY MODEL LOADING
# ============================================
MAX_CACHED_MODELS = 2
MODELS = {}
MODEL_PATHS = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_files = glob.glob(os.path.join(BASE_DIR, "*_robust.tflite"))

print("=" * 60)
print(f"🔍 Model dir: {BASE_DIR}")
print(f"📦 Found {len(model_files)} model files")
print("=" * 60)

for model_file in model_files:
    alphabet_name = os.path.basename(model_file).replace("_robust.tflite", "")
    MODEL_PATHS[alphabet_name] = model_file
    print(f"📁 Registered: {alphabet_name}")

print(f"✅ Registered {len(MODEL_PATHS)} models (lazy load, cache={MAX_CACHED_MODELS})")
print("=" * 60)


def load_model(alphabet_name):
    """Load a model on demand, cache with LRU eviction."""
    if alphabet_name in MODELS:
        return MODELS[alphabet_name]
    if alphabet_name not in MODEL_PATHS:
        return None

    if len(MODELS) >= MAX_CACHED_MODELS:
        oldest = next(iter(MODELS))
        try:
            del MODELS[oldest]
            gc.collect()
            print(f"🗑️ Evicted: {oldest}")
        except Exception as e:
            print(f"⚠️ Eviction failed: {e}")

    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATHS[alphabet_name])
        interpreter.allocate_tensors()
        MODELS[alphabet_name] = {
            'interpreter': interpreter,
            'input_details': interpreter.get_input_details(),
            'output_details': interpreter.get_output_details()
        }
        print(f"✅ Loaded: {alphabet_name}  (cache: {len(MODELS)})")
        return MODELS[alphabet_name]
    except Exception as e:
        print(f"❌ Failed to load {alphabet_name}: {e}")
        return None


# ============================================
# ALPHABET DISPLAY
# ============================================
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


# ============================================
# ROUTES
# ============================================
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'SignSpeak API',
        'mediapipe_enabled': USE_MEDIAPIPE,
        'endpoints': ['/ping', '/health', '/models', '/detect']
    })


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({
        'status': 'OK',
        'mediapipe': USE_MEDIAPIPE,
        'models': list(MODEL_PATHS.keys()),
        'loaded': list(MODELS.keys())
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'mediapipe_enabled': USE_MEDIAPIPE,
        'models_count': len(MODEL_PATHS),
        'loaded_count': len(MODELS)
    })


@app.route('/models', methods=['GET'])
def get_models():
    return jsonify({
        'models': [
            {
                'name': n,
                'display': ALPHABET_DISPLAY.get(n, n),
                'arabic': ALPHABET_DISPLAY.get(n, '?')
            }
            for n in MODEL_PATHS.keys()
        ],
        'count': len(MODEL_PATHS)
    })


@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        image_data = data.get('image')
        alphabet = data.get('alphabet', 'alif')

        if not image_data:
            return jsonify({'error': 'No image data'}), 400

        # Strip data URL prefix
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # ============================================
        # EXTRACT LANDMARKS
        # ============================================
        if USE_MEDIAPIPE and hands is not None:
            # Full MediaPipe path
            import cv2
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
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

            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])
        else:
            # FALLBACK: dummy landmarks
            # App will display skeleton overlay for demo purposes
            landmarks = [
                0.50, 0.55,   # wrist
                0.45, 0.50, 0.42, 0.45, 0.40, 0.40, 0.39, 0.36,
                0.50, 0.45, 0.50, 0.38, 0.50, 0.32, 0.50, 0.28,
                0.55, 0.45, 0.56, 0.38, 0.57, 0.32, 0.58, 0.28,
                0.60, 0.45, 0.62, 0.38, 0.64, 0.33, 0.66, 0.30,
                0.65, 0.48, 0.68, 0.43, 0.71, 0.40, 0.73, 0.38,
            ]

        # ============================================
        # RUN TFLITE MODEL
        # ============================================
        model_data = load_model(alphabet)
        if model_data is None:
            return jsonify({
                'hasHand': True,
                'isAlphabet': False,
                'confidence': 0.0,
                'landmarks': landmarks,
                'alphabet': alphabet,
                'display': ALPHABET_DISPLAY.get(alphabet, alphabet),
                'message': f'Model {alphabet} not found'
            })

        features = np.array(landmarks, dtype=np.float32).reshape(1, -1)
        interpreter = model_data['interpreter']
        input_details = model_data['input_details']
        output_details = model_data['output_details']

        interpreter.set_tensor(input_details[0]['index'], features)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        score = float(prediction[0][0])

        return jsonify({
            'hasHand': True,
            'isAlphabet': score > 0.5,
            'confidence': score,
            'landmarks': landmarks,
            'alphabet': alphabet,
            'display': ALPHABET_DISPLAY.get(alphabet, alphabet),
            'message': 'Success' if USE_MEDIAPIPE else 'Fallback mode'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        gc.collect()


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print("🚀 ALPHABET DETECTION SERVER")
    print("=" * 60)
    print(f"📚 Registered {len(MODEL_PATHS)} models")
    print(f"🎯 MediaPipe enabled: {USE_MEDIAPIPE}")
    print(f"📡 Server: http://0.0.0.0:{port}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)