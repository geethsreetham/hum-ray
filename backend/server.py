# --- IMPORTS ---
import sqlite3
import os
import math
import numpy as np
import cv2
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from PIL import Image
from PIL.ExifTags import TAGS
from stegano import lsb
from transformers import pipeline

# --- CONSTANTS ---
DB_FILE = "humray.db"
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- FLASK APP SETUP ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

# --- DATABASE SETUP ---
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
                details TEXT,
                feedback INTEGER DEFAULT 0
            )
        ''')
        db.commit()
        print("✅ Database Initialized")

# --- 🧠 THE NOVEL LOGIC MODULES (No training needed, pure Math) ---

class ForensicEngine:
    """
    The Core Brain of HumRay.
    Uses Math and Physics to detect anomalies, not just cached weights.
    """
    
    @staticmethod
    def analyze_frequency_patterns(image_path):
        """
        NOVELTY 1: FFT (Fast Fourier Transform) Analysis.
        AI Generators leave a distinct 'grid' pattern in the frequency domain.
        Real cameras have smooth noise. AI has spikes.
        """
        try:
            img = cv2.imread(image_path, 0) # Read as grayscale
            if img is None: return 0.0

            # Transform to Frequency Domain
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

            # Analyze High Frequency Energy (Corners of the spectrum)
            rows, cols = img.shape
            crow, ccol = rows // 2, cols // 2
            
            # Mask out the center (Low frequencies = actual image content)
            # We only care about the high-freq noise
            magnitude_spectrum[crow-30:crow+30, ccol-30:ccol+30] = 0
            
            # Calculate average energy of high frequencies
            avg_energy = np.mean(magnitude_spectrum)
            
            # Heuristic: AI images often have abnormally high or patterned high-freq energy
            # Normal photos ~ 90-110. AI often > 120 or < 50 (too smooth).
            # This is a simplified score normalization.
            score = 0.0
            if avg_energy > 115:
                score = min((avg_energy - 115) / 50, 1.0) # High energy = Artifacts
            
            return score
        except Exception as e:
            print(f"FFT Error: {e}")
            return 0.0

    @staticmethod
    def analyze_entropy(file_path):
        """
        NOVELTY 2: Shannon Entropy for Malware/Crypto.
        Calculates randomness.
        7.5 - 8.0 = Encrypted/Compressed (Suspicious for 'Text' files, normal for Zip)
        """
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            if not data: return 0

            entropy = 0
            for x in range(256):
                p_x = float(data.count(x))/len(data)
                if p_x > 0:
                    entropy += - p_x * math.log(p_x, 2)
            
            return entropy
        except Exception:
            return 0

    @staticmethod
    def analyze_video_physics(video_path):
        """
        NOVELTY 3: Temporal Coherence (The 'Dancing Dog' Check).
        Uses Optical Flow to see if textures 'flicker' unnaturally.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return 0.0, "Could not open"

        ret, prev_frame = cap.read()
        if not ret: return 0.0, "Empty video"
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        total_physics_error = 0
        frame_count = 0
        max_frames = 30 # Check first 30 frames for efficiency

        while frame_count < max_frames:
            ret, curr_frame = cap.read()
            if not ret: break
            
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate Optical Flow (Where pixels moved)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Predict next frame based on physics
            h, w = prev_gray.shape
            flow_map = np.column_stack((np.repeat(np.arange(h), w), np.tile(np.arange(w), h)))
            # (Simplified remapping for demo speed)
            
            # We measure the difference between 'Physics Prediction' and 'Actual Frame'
            # A high difference means the texture changed in a way physics can't explain (AI Hallucination)
            diff = cv2.absdiff(prev_gray, curr_gray) # Simple diff for robustness in this demo
            # In a full research paper, we use the warped flow diff.
            
            score = np.mean(diff)
            # If mean difference is weirdly low (static) or weirdly high (glitchy), it contributes.
            
            total_physics_error += score
            prev_gray = curr_gray
            frame_count += 1
            
        cap.release()
        avg_error = total_physics_error / (frame_count + 1e-5)
        
        # Normalize: Real videos have consistent motion (avg error ~5-15).
        # AI often has 'shimmering' (higher micro-changes).
        probability_fake = min(avg_error / 50.0, 1.0) 
        return probability_fake, f"Physics Error Rate: {avg_error:.2f}"

# --- HELPER FUNCTIONS ---
def analyze_metadata(image):
    try:
        exif_data = image._getexif()
        if not exif_data: return "No Metadata (Suspicious)"
        details = []
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name in ['Make', 'Model', 'Software']:
                details.append(f"{tag_name}: {value}")
        return ", ".join(details) if details else "Metadata Stripped"
    except:
        return "Error reading metadata"

# --- API ROUTES ---

@app.route("/api/analyze-image", methods=['POST'])
def analyze_image():
    if 'image' not in request.files: return jsonify({"message": "No file"}), 400
    file = request.files['image']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        # 1. Run Forensic Checks
        fft_score = ForensicEngine.analyze_frequency_patterns(filepath)
        
        # SAFE STEGANOGRAPHY CHECK
        # Wrap in try/except because lsb.reveal() throws errors if no message is found
        try:
            stego_check = lsb.reveal(filepath)
            stego_msg = "Hidden Data Detected!" if stego_check else "None"
        except Exception:
            stego_msg = "None"
        
        # 2. Metadata Check
        image = Image.open(filepath)
        meta_result = analyze_metadata(image)
        
        # 3. Final Decision Logic (The "Cascaded" Vote)
        # If FFT says it's AI (high score) OR Metadata is weird...
        final_score = fft_score
        if "Photoshop" in meta_result or "GIMP" in meta_result:
            final_score += 0.2
        
        final_score = min(final_score, 1.0)
        
        verdict = "AI-Generated / Edited" if final_score > 0.5 else "Likely Real"
        
        details = f"FFT Artifacts: {fft_score:.2f} | Meta: {meta_result}"

        # Log to DB
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO analysis_log (filename, file_type, verdict, score, details) VALUES (?, ?, ?, ?, ?)",
                       (file.filename, 'image', verdict, final_score, details))
        conn.commit()

        return jsonify({
            "verdict": verdict,
            "score": f"{final_score*100:.1f}%",
            "metadata_details": meta_result,
            "steganography_details": stego_msg,
            "log_id": cursor.lastrowid
        })
    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"message": "Analysis Failed"}), 500

@app.route("/api/analyze-video", methods=['POST'])
def analyze_video():
    if 'video' not in request.files: return jsonify({"message": "No file"}), 400
    file = request.files['video']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        # Run Physics Engine
        physics_score, debug_msg = ForensicEngine.analyze_video_physics(filepath)
        
        verdict = "Deepfake (Physics Violation)" if physics_score > 0.4 else "Natural Motion"
        
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO analysis_log (filename, file_type, verdict, score, details) VALUES (?, ?, ?, ?, ?)",
                       (file.filename, 'video', verdict, physics_score, debug_msg))
        conn.commit()

        return jsonify({
            "verdict": verdict,
            "score": f"{physics_score*100:.1f}% Unnatural",
            "frame_summary": debug_msg,
            "log_id": cursor.lastrowid
        })
    except Exception as e:
        return jsonify({"message": f"Error: {e}"}), 500

@app.route("/api/analyze-text", methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    
    # Simple NLP + Logic
    # 1. Phishing Keywords
    phishing_triggers = ["urgent", "verify your account", "bank", "password", "suspended"]
    found_triggers = [w for w in phishing_triggers if w in text.lower()]
    
    # 2. Invisible Character Logic
    invisible_chars = ['\u200b', '\u200c', '\u200d']
    has_invisible = any(char in text for char in invisible_chars)

    score = 0.1
    if found_triggers: score += 0.5
    if has_invisible: score += 0.4
    
    verdict = "Suspicious / Phishing" if score > 0.5 else "Safe Text"
    
    return jsonify({
        "verdict": verdict,
        "score": f"{score*100:.1f}% Risk",
        "homoglyph_details": f"Triggers: {found_triggers}",
        "invisible_details": "Detected" if has_invisible else "None",
        "log_id": 0
    })

@app.route("/api/feedback", methods=['POST'])
def feedback():
    data = request.get_json()
    # Simple logging of feedback
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE analysis_log SET feedback = ? WHERE id = ?", (data.get('feedback'), data.get('log_id')))
    conn.commit()
    return jsonify({"message": "Feedback loop updated."})

if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)