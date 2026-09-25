import os
import gc
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
import glob
from PIL import Image
import tensorflow as tf
import mediapipe as mp

# ============================================
# MEDIAPIPE HANDS — TRACKING MODE (faster)
# ============================================
hands = mp.solutions.hands.Hands(
    static_image_mode=False,        # tracking = faster
    max_num_hands=1,
    model_complexity=0,             # lite = less memory
    min_detection_confidence=0.4,
    min_tracking_confidence=0.3
)
print("✅ MediaPipe Hands (tracking mode, lite)")

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

for model_file in model_files:
    alphabet_name = os.path.basename(model_file).replace("_robust.tflite", "")
    MODEL_PATHS[alphabet_name] = model_file

print(f"✅ Registered {len(MODEL_PATHS)} models (lazy load, cache={MAX_CACHED_MODELS})")


def load_model(alphabet_name):
    if alphabet_name in MODELS:
        return MODELS[alphabet_name]
    if alphabet_name not in MODEL_PATHS:
        return None
    if len(MODELS) >= MAX_CACHED_MODELS:
        oldest = next(iter(MODELS))
        try:
            del MODELS[oldest]
            gc.collect()
        except Exception:
            pass
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATHS[alphabet_name])
        interpreter.allocate_tensors()
        MODELS[alphabet_name] = {
            'interpreter': interpreter,
            'input_details': interpreter.get_input_details(),
            'output_details': interpreter.get_output_details()
        }
        print(f"✅ Loaded: {alphabet_name}")
        return MODELS[alphabet_name]
    except Exception as e:
        print(f"❌ Failed: {alphabet_name}: {e}")
        return None


ALPHABET_DISPLAY = {
    'alif': 'ا', 'bay': 'ب', 'tay': 'ت', 'thay': 'ث', 'seen': 'س',
    'sheen': 'ش', 'suaad': 'ص', 'zvad': 'ض', 'toayn': 'ط', 'zoyn': 'ظ',
    'ain': 'ع', 'ghain': 'غ', 'fe': 'ف', 'quaaf': 'ق', 'kaf': 'ك',
    'gaf': 'گ', 'lam': 'ل', 'mim': 'م', 'noon': 'ن', 'vao': 'و',
    'hamza': 'ء', 'choti_ye': 'ی', 'bari_ye': 'ے', 'ray': 'ر',
    'rray': 'ڑ', 'zay': 'ز', 'dal': 'د', 'daal': 'ڈ', 'zal': 'ذ',
    'khay': 'خ', 'rre': 'ڑ',
}


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'SignSpeak API',
        'endpoints': ['/ping', '/health', '/models', '/warmup/<alphabet>', '/detect']
    })


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({
        'status': 'OK',
        'models': list(MODEL_PATHS.keys()),
        'loaded': list(MODELS.keys())
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'models_count': len(MODEL_PATHS),
        'loaded_count': len(MODELS)
    })


@app.route('/models', methods=['GET'])
def get_models():
    return jsonify({
        'models': [
            {'name': n, 'display': ALPHABET_DISPLAY.get(n, n), 'arabic': ALPHABET_DISPLAY.get(n, '?')}
            for n in MODEL_PATHS.keys()
        ],
        'count': len(MODEL_PATHS)
    })


@app.route('/warmup/<alphabet>', methods=['GET'])
def warmup(alphabet):
    """Preload model so first detection is instant."""
    t0 = time.time()
    m = load_model(alphabet)
    elapsed = time.time() - t0
    if m:
        print(f"🔥 Warmup {alphabet} took {elapsed:.2f}s")
        return jsonify({'status': 'ready', 'alphabet': alphabet, 'time': round(elapsed, 2)})
    return jsonify({'status': 'not_found', 'alphabet': alphabet}), 404


@app.route('/detect', methods=['POST'])
def detect():
    t0 = time.time()
    try:
        data = request.json
        image_data = data.get('image')
        alphabet = data.get('alphabet', 'alif')

        if not image_data:
            return jsonify({'error': 'No image data'}), 400

        if len(image_data) > 500_000:
            return jsonify({
                'hasHand': False, 'isAlphabet': False, 'confidence': 0.0,
                'landmarks': [], 'alphabet': alphabet,
                'display': ALPHABET_DISPLAY.get(alphabet, alphabet),
                'message': 'Image too large'
            })

        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            elapsed = time.time() - t0
            print(f"⏱️ /detect no hand ({elapsed:.2f}s)")
            return jsonify({
                'hasHand': False, 'isAlphabet': False, 'confidence': 0.0,
                'landmarks': [], 'alphabet': alphabet,
                'display': ALPHABET_DISPLAY.get(alphabet, alphabet),
                'message': 'No hand detected'
            })

        landmarks = []
        for hl in results.multi_hand_landmarks:
            for lm in hl.landmark:
                landmarks.extend([lm.x, lm.y])

        model_data = load_model(alphabet)
        if model_data is None:
            return jsonify({
                'hasHand': True, 'isAlphabet': False, 'confidence': 0.0,
                'landmarks': landmarks, 'alphabet': alphabet,
                'display': ALPHABET_DISPLAY.get(alphabet, alphabet),
                'message': f'Model {alphabet} not found'
            })

        features = np.array(landmarks, dtype=np.float32).reshape(1, -1)
        m = model_data
        m['interpreter'].set_tensor(m['input_details'][0]['index'], features)
        m['interpreter'].invoke()
        pred = m['interpreter'].get_tensor(m['output_details'][0]['index'])
        score = float(pred[0][0])

        elapsed = time.time() - t0
        print(f"⏱️ /detect ({alphabet}) took {elapsed:.2f}s, score={score:.2f}")

        return jsonify({
            'hasHand': True,
            'isAlphabet': score > 0.5,
            'confidence': score,
            'landmarks': landmarks,
            'alphabet': alphabet,
            'display': ALPHABET_DISPLAY.get(alphabet, alphabet),
            'message': 'Success'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        gc.collect()


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
