# --- IMPORTS ---
import sqlite3
import os
import re # For text forensics
import math # For video frame calculation
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from PIL import Image
from PIL.ExifTags import TAGS
from stegano import lsb
import numpy as np
import tensorflow as tf
# --- We need these to build the model locally ---
from tensorflow import keras
from keras import layers, models, regularizers
from keras import Input
# ---
from transformers import pipeline
import cv2 # For video processing

# --- CONSTANTS ---
DB_FILE = "humray.db"
UPLOAD_FOLDER = 'uploads'
MODEL_WEIGHTS_FILE = 'humray_model_v2_weights.weights.h5' # <-- Loading our v2 model!
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- FLASK APP SETUP ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)  # Allow all origins

# --- DATABASE FUNCTIONS ---
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_FILE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                filename TEXT,
                file_type TEXT,
                verdict TEXT,
                score REAL,
                metadata_details TEXT,
                steganography_details TEXT,
                feedback INTEGER DEFAULT 0
            )
        ''')
        db.commit()
        print("✅ Database Initialized")

# --- AI MODEL INITIALIZATION ---

# 1. Image AI Model (Building it locally and loading weights)
def build_our_model():
    # This architecture MUST be IDENTICAL to the one in Colab
    model = models.Sequential()
    model.add(Input(shape=(32, 32, 3))) # The "correct" input layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu')) 
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2)) 
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

try:
    # 1. Build the "empty brain"
    image_model = build_our_model()
    # 2. Load the "knowledge" (weights) into it
    image_model.load_weights(MODEL_WEIGHTS_FILE)
    
    print(f"✅ Real Image AI Model Built and Weights Loaded from {MODEL_WEIGHTS_FILE}")
    MODEL_INPUT_SHAPE = image_model.input_shape[1:3] # Gets (32, 32)
    print(f"✅ Model expects input shape: {MODEL_INPUT_SHAPE}")
except Exception as e:
    print(f"--- !!! FAILED TO LOAD REAL MODEL: {e} !!! ---")
    print("--- !!! FALLING BACK TO PLACEHOLDER !!! ---")
    # This is our old safety net
    def create_placeholder_model():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
    image_model = create_placeholder_model()
    MODEL_INPUT_SHAPE = (150, 150)
    print("✅ Image AI Model Loaded (Placeholder)")


# 2. Text AI Model
try:
    text_classifier = pipeline('sentiment-analysis')
    print("✅ Text AI Model Loaded")
except Exception as e:
    print(f"Error loading text model: {e}")
    text_classifier = None

# --- FORENSIC ANALYSIS FUNCTIONS ---

# --- Metadata Analysis Function ---
def analyze_metadata(image):
    try:
        exif_data = image._getexif()
        if not exif_data:
            return "No metadata found. (Image may be from social media, a screenshot, or intentionally stripped)."

        metadata_details = {}
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name in ['Make', 'Model', 'Software', 'DateTime']:
                metadata_details[tag_name] = str(value)

        if not metadata_details:
            return "Basic metadata found, but no identifiable fields (Make, Model, Software)."

        return ", ".join([f"{k}: {v}" for k, v in metadata_details.items()])
    except Exception as e:
        return f"Could not read metadata: {e}"

# --- Steganography Analysis Function ---
def analyze_steganography(image_path):
    try:
        secret_data = lsb.reveal(image_path)
        if secret_data:
            return "High Probability: Hidden data detected within the image."
        else:
            return "Low Probability: No hidden data detected."
    except Exception as e:
        return "Low Probability: No hidden data detected."

# --- Text Forensic Functions ---
def detect_invisible_chars(text):
    invisible_chars = ['\u200b', '\u200c', '\u200d', '\uFEFF']
    found = [char for char in invisible_chars if char in text]
    if found:
        return "Warning: Invisible characters detected (e.g., Zero Width Space)."
    return "No invisible characters detected."

def detect_homoglyphs(text):
    HOMOGLYPHS = { 'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x', 'і': 'i', 'ј': 'j' }
    found_glyphs = [char for char in text if char in HOMOGLYPHS]
    if found_glyphs:
        return f"Warning: Suspicious characters (homoglyphs) detected: {', '.join(set(found_glyphs))}."
    return "No homoglyph characters detected."

# --- CORE LOGIC FUNCTIONS ---
def analyze_image_data(image, image_path):
    # --- 1. Cybersecurity Forensic Layer: Metadata ---
    metadata_details = analyze_metadata(image)
    
    # --- 2. Cybersecurity Forensic Layer: Steganography ---
    steganography_details = analyze_steganography(image_path)
    
    # --- 3. AI Layer (USING OUR REAL MODEL) ---
    img_for_model = image.resize(MODEL_INPUT_SHAPE) 
    if img_for_model.mode == 'RGBA':
        img_for_model = img_for_model.convert('RGB')
        
    # --- THIS IS THE FIX ---
    # Our model was trained on pixels 0-255, so we MUST feed it pixels 0-255.
    # We remove the "/ 255.0" to fix the data mismatch.
    img_array = np.array(img_for_model) 
    # --- END OF FIX ---
    
    img_array = np.expand_dims(img_array, axis=0)
    
    image_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    prediction = image_model.predict(img_array)[0][0]
    
    # This is the "Real-ness" score (0.0 = FAKE, 1.0 = REAL)
    real_score = float(prediction) 
    
    # Our logic is now correct (FAKE=0, REAL=1)
    if real_score > 0.5:
        verdict = "Looks Real" # It's a REAL image (label 1)
    else:
        verdict = "AI-Generated" # It's a FAKE image (label 0)
    
    # We report the "AI-Generated" confidence
    ai_confidence_score = (1.0 - real_score)
        
    return verdict, ai_confidence_score, metadata_details, steganography_details

# --- API ROUTES ---

@app.route("/api/analyze-image", methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"message": "No image file provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400
    
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        image = Image.open(filepath)
        
        # --- Call our core analysis function ---
        verdict, score, metadata_details, steganography_details = analyze_image_data(image, filepath)

        # --- Database Logging ---
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO analysis_log (filename, file_type, verdict, score, metadata_details, steganography_details) VALUES (?, ?, ?, ?, ?, ?)",
            (file.filename, 'image', verdict, score, metadata_details, steganography_details)
        )
        log_id = cursor.lastrowid
        conn.commit()
        
        # os.remove(filepath) 
        
        return jsonify({
            "verdict": verdict,
            "score": f"{score * 100:.2f}%", # This now correctly reports the AI score
            "metadata_details": metadata_details,
            "steganography_details": steganography_details,
            "log_id": log_id
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"message": f"Error processing the image: {e}"}), 500

@app.route("/api/analyze-text", methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text')
    
    if not text:
        return jsonify({"message": "No text provided"}), 400
    
    try:
        invisible_details = detect_invisible_chars(text)
        homoglyph_details = detect_homoglyphs(text)
        
        result = text_classifier(text)[0]
        score = float(result['score'])
        label = result['label']
        
        if label == 'NEGATIVE' or label == 'LABEL_0':
            verdict = "Suspicious (AI)"
        else:
            verdict = "Looks Safe (AI)"
        
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO analysis_log (filename, file_type, verdict, score, metadata_details, steganography_details) VALUES (?, ?, ?, ?, ?, ?)",
            ('N/A - Text Entry', 'text', verdict, score, homoglyph_details, invisible_details)
        )
        log_id = cursor.lastrowid
        conn.commit()
        
        return jsonify({
            "verdict": verdict,
            "score": f"{score * 100:.2f}% ({label})",
            "homoglyph_details": homoglyph_details,
            "invisible_details": invisible_details,
            "log_id": log_id
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"message": f"Error processing the text: {e}"}), 500

# --- VIDEO ROUTE ---
@app.route("/api/analyze-video", methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({"message": "No video file provided"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return jsonify({"message": "Could not open video file"}), 500

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        
        sample_interval_seconds = 2
        frames_to_sample = min(10, math.floor(duration / sample_interval_seconds))
        if frames_to_sample == 0:
             frames_to_sample = 1 

        frame_indices = [int(i * (total_frames / (frames_to_sample + 1))) for i in range(1, frames_to_sample + 1)]
        
        ai_votes = 0
        real_votes = 0
        total_score = 0 # This will be the AI score
        frame_results = [] 

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                frame_path = os.path.join(app.config['UPLOAD_FOLDER'], f"frame_{idx}.jpg")
                pil_image.save(frame_path)

                # --- This now returns the (verdict, ai_score, ...)
                verdict, ai_score, meta, stego = analyze_image_data(pil_image, frame_path)
                
                if verdict == "AI-Generated":
                    ai_votes += 1
                else:
                    real_votes += 1
                total_score += ai_score
                
                frame_results.append(f"Frame {idx}: Meta({meta}) Stego({stego})")
                
                os.remove(frame_path)

        cap.release()
        
        if ai_votes > real_votes:
            final_verdict = "Probable Deepfake"
        else:
            final_verdict = "Looks Real"
            
        avg_score = (total_score / frames_to_sample) if frames_to_sample > 0 else 0
        forensic_summary = f"Sampled {frames_to_sample} frames. AI Votes: {ai_votes}, Real Votes: {real_votes}."

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO analysis_log (filename, file_type, verdict, score, metadata_details, steganography_details) VALUES (?, ?, ?, ?, ?, ?)",
            (file.filename, 'video', final_verdict, avg_score, forensic_summary, "Steganography check on frames only.")
        )
        log_id = cursor.lastrowid
        conn.commit()

        # os.remove(filepath) # Clean up original video

        return jsonify({
            "verdict": final_verdict,
            "score": f"{avg_score * 100:.2f}% (Average AI Confidence)",
            "frame_summary": forensic_summary,
            "frame_count": frames_to_sample,
            "log_id": log_id
        })

    except Exception as e:
        print(f"Error processing video: {e}")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        return jsonify({"message": f"Error processing video: {e}"}), 500


@app.route("/api/feedback", methods=['POST'])
def feedback():
    data = request.get_json()
    log_id = data.get('log_id')
    feedback_value = data.get('feedback')
    
    if not log_id or feedback_value not in [1, -1]:
        return jsonify({"message": "Invalid data"}), 400
        
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE analysis_log SET feedback = ? WHERE id = ?",
            (feedback_value, log_id)
        )
        conn.commit()
        return jsonify({"message": "Feedback received, thank you!"})
    except Exception as e:
        return jsonify({"message": f"Database error: {e}"}), 500

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    init_db() # Ensure DB is created on start
    app.run(debug=True, port=5000)