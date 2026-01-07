# --- HUMRAY: UNIFIED DIGITAL IMMUNE SYSTEM (FINAL WORKING VERSION) ---
# Features: Novel Forensics + Adaptive RL + Auto-DB Fix + Tuned Thresholds

import sqlite3
import os
import math
import numpy as np
import cv2
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from PIL import Image, ImageChops
from PIL.ExifTags import TAGS
from stegano import lsb

# --- CONFIGURATION ---
DB_FILE = "humray.db"
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

# --- GLOBAL ADAPTIVE PARAMETERS ---
# FIXED: Entropy raised to 7.99 to ignore normal JPEGs.
SYSTEM_PARAMS = {
    "FFT_THRESHOLD": 125.0,     
    "ENTROPY_THRESHOLD": 7.99,  
    "WATERMARK_SENSITIVITY": 60.0 
}

# --- DATABASE MANAGEMENT ---
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
    """ Initializes DB and Auto-Fixes missing columns """
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        
        # Create table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                filename TEXT,
                file_type TEXT,
                verdict TEXT,
                score REAL,
                fft_val REAL,
                details TEXT,
                feedback INTEGER DEFAULT 0
            )
        ''')
        
        # AUTO-FIX: Add 'fft_val' if missing (Prevents "no such column" error)
        try:
            cursor.execute("SELECT fft_val FROM analysis_log LIMIT 1")
        except sqlite3.OperationalError:
            print("⚠️ Upgrading Database: Adding missing 'fft_val' column...")
            cursor.execute("ALTER TABLE analysis_log ADD COLUMN fft_val REAL DEFAULT 0.0")
            
        db.commit()
        
        # Trigger RL on startup
        optimize_thresholds(db)
        print("✅ HumRay Database Initialized")

# --- 🧠 REINFORCEMENT LEARNING ---
def optimize_thresholds(db):
    global SYSTEM_PARAMS
    cursor = db.cursor()
    # Find False Positives (We said AI, User said Real)
    cursor.execute("SELECT fft_val FROM analysis_log WHERE verdict LIKE '%AI%' AND feedback = -1")
    false_positives = cursor.fetchall()
    
    # Filter valid scores
    valid_scores = [x[0] for x in false_positives if x[0] and x[0] > 0]
    
    if valid_scores:
        avg_fp = sum(valid_scores) / len(valid_scores)
        # If user feedback suggests our threshold is too low, raise it.
        if avg_fp > SYSTEM_PARAMS["FFT_THRESHOLD"]:
            SYSTEM_PARAMS["FFT_THRESHOLD"] = min(avg_fp + 2.0, 140.0)
            print(f"   [RL] 🧠 Adapted FFT Threshold to {SYSTEM_PARAMS['FFT_THRESHOLD']:.2f}")

# ==============================================================================
# 🧠 FORENSIC ENGINE (NOVEL ALGORITHMS)
# ==============================================================================
class ForensicEngine:
    
    # 1. Watermark Detection
    @staticmethod
    def detect_watermarks(image_path):
        try:
            img = cv2.imread(image_path)
            if img is None: return 0.0, ""
            h, w, _ = img.shape
            # Check bottom-right corner for logos
            corner = img[h-30:h, w-80:w]
            if corner.size == 0: return 0.0, ""
            (mean, std) = cv2.meanStdDev(corner)
            if np.mean(std) > SYSTEM_PARAMS["WATERMARK_SENSITIVITY"]:
                return 0.85, "Visual signature detected (Logo/Watermark)"
            return 0.0, ""
        except: return 0.0, ""

    # 2. FFT Analysis (The "Grid" Check)
    @staticmethod
    def analyze_frequency_patterns(image_path):
        try:
            img = cv2.imread(image_path, 0)
            if img is None: return 0.0, "", 0.0
            
            # FFT Math
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
            
            # Mask center
            rows, cols = img.shape
            crow, ccol = rows // 2, cols // 2
            magnitude_spectrum[crow-30:crow+30, ccol-30:ccol+30] = 0
            
            avg_energy = np.mean(magnitude_spectrum)
            
            thresh = SYSTEM_PARAMS["FFT_THRESHOLD"]
            if avg_energy > thresh:
                score = min((avg_energy - thresh) / 40, 1.0)
                return score, f"Unnatural pixel grid patterns (FFT: {avg_energy:.1f})", avg_energy
            return 0.0, "", avg_energy
        except: return 0.0, "", 0.0

    # 3. LSB Variance (Hidden Crypto)
    @staticmethod
    def analyze_lsb_variance(image_path):
        try:
            img = cv2.imread(image_path)
            if img is None: return 0.0, ""
            blue = img[:,:,0]
            lsb = blue & 1
            avg_val = np.mean(lsb)
            # FIXED: Drastically tighter tolerance (0.0001) to ignore sensor noise
            dist_from_random = abs(avg_val - 0.5)
            if dist_from_random < 0.0001: 
                return 0.6, "LSB Statistical Anomaly (Potential Encrypted Data)"
            return 0.0, ""
        except: return 0.0, ""

    # 4. Shannon Entropy (Malware)
    @staticmethod
    def analyze_entropy(file_path):
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            if not data: return 0, ""
            if len(data) > 1000000: data = data[:1000000]
            entropy = 0
            for x in range(256):
                p_x = float(data.count(x))/len(data)
                if p_x > 0: entropy += - p_x * math.log(p_x, 2)
            
            # FIXED: Uses 7.99 threshold
            if entropy > SYSTEM_PARAMS["ENTROPY_THRESHOLD"]:
                 return 0.9, f"Abnormal File Randomness (Entropy: {entropy:.2f})"
            return 0.0, ""
        except: return 0.0, ""

    # 5. ELA Proxy (Editing Traces)
    @staticmethod
    def detect_editing_traces(image_path):
        try:
            temp_path = image_path + ".temp.jpg"
            img = Image.open(image_path).convert('RGB')
            img.save(temp_path, 'JPEG', quality=90)
            temp_img = Image.open(temp_path)
            ela_img = ImageChops.difference(img, temp_img)
            extrema = ela_img.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            os.remove(temp_path)
            if max_diff > 50:
                 return 0.6, "Inconsistent Compression (Likely Edited)"
            return 0.0, ""
        except: return 0.0, ""

    # 6. Video Physics
    @staticmethod
    def analyze_video_physics(video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return 0.0, "Error"
        ret, prev_frame = cap.read()
        if not ret: return 0.0, "Empty"
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        total_error = 0
        frame_count = 0
        while frame_count < 20: 
            ret, curr_frame = cap.read()
            if not ret: break
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, curr_gray)
            total_error += np.mean(diff)
            prev_gray = curr_gray
            frame_count += 1
        cap.release()
        avg_diff = total_error / (frame_count + 1e-5)
        
        prob = 0.0
        if avg_diff < 2.0: prob = 0.9
        elif avg_diff > 35.0: prob = 0.7
        return prob, f"Motion Score: {avg_diff:.2f}"

# --- HELPER FUNCTIONS ---
def get_human_explanation(flags):
    if not flags:
        return "No significant anomalies found. The file structure resembles a standard camera capture."
    explanation = "We flagged this because: "
    for flag in flags:
        explanation += f"{flag}. "
    return explanation

def analyze_metadata(image):
    try:
        exif_data = image._getexif()
        if not exif_data: return "No Metadata found (Web/Screenshot)", 0.0 
        details = []
        suspicious_score = 0.0
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name in ['Make', 'Model', 'Software']:
                val_str = str(value).lower()
                details.append(f"{tag_name}: {value}")
                if "photoshop" in val_str or "gimp" in val_str:
                    suspicious_score += 0.3
                if "stable diffusion" in val_str or "midjourney" in val_str:
                    suspicious_score += 1.0 
        return ", ".join(details), suspicious_score
    except:
        return "Error reading metadata", 0.0

# ==============================================================================
# 🚀 API ROUTES
# ==============================================================================

@app.route("/api/analyze-image", methods=['POST'])
def analyze_image():
    if 'image' not in request.files: return jsonify({"message": "No file"}), 400
    file = request.files['image']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    explanation_flags = []

    try:
        # Run Forensic Checks
        wm_score, wm_msg = ForensicEngine.detect_watermarks(filepath)
        if wm_msg: explanation_flags.append(wm_msg)
        
        fft_score, fft_msg, raw_fft = ForensicEngine.analyze_frequency_patterns(filepath)
        if fft_msg: explanation_flags.append(fft_msg)
        
        edit_score, edit_msg = ForensicEngine.detect_editing_traces(filepath)
        if edit_msg: explanation_flags.append(edit_msg)

        lsb_score, lsb_msg = ForensicEngine.analyze_lsb_variance(filepath)
        if lsb_msg: explanation_flags.append(lsb_msg)
        
        # Standard Stego Check
        stego_found = False
        try:
            if lsb.reveal(filepath): 
                stego_found = True
                explanation_flags.append("Standard Steganography detected")
        except: pass

        ent_score, ent_msg = ForensicEngine.analyze_entropy(filepath)
        if ent_msg: explanation_flags.append(ent_msg)

        # Metadata Check
        try:
            image = Image.open(filepath)
            meta_text, meta_score = analyze_metadata(image)
            if meta_score > 0: explanation_flags.append("Metadata indicates editing software")
        except: 
            meta_text = "Error"
            meta_score = 0.0

        # --- SCORING ---
        # AI/Edit Risks vs Malware Risks
        ai_risk = max(fft_score, wm_score, meta_score)
        malware_risk = max(ent_score, lsb_score if stego_found else 0.0)

        # Priority of Verdicts
        if malware_risk > 0.8:
            verdict = "Hidden Data / Malware"
            final_score = malware_risk
        elif ai_risk > 0.8:
            verdict = "AI Generated"
            final_score = ai_risk
        elif ai_risk > 0.5:
            verdict = "Probable AI"
            final_score = ai_risk
        elif edit_score > 0.4:
            verdict = "Real but Edited"
            final_score = 0.3
        elif ai_risk > 0.2:
            verdict = "Likely Authentic"
            final_score = ai_risk
        else:
            verdict = "Authentic / Real"
            final_score = 0.1

        human_explanation = get_human_explanation(explanation_flags)

        # DB Log
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO analysis_log (filename, file_type, verdict, score, fft_val, details) VALUES (?, ?, ?, ?, ?, ?)",
                       (file.filename, 'image', verdict, final_score, raw_fft, human_explanation))
        conn.commit()

        return jsonify({
            "verdict": verdict,
            "score": f"{final_score*100:.1f}% Risk",
            "metadata_details": meta_text,
            "steganography_details": human_explanation, 
            "log_id": cursor.lastrowid
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"message": "Analysis Failed"}), 500

@app.route("/api/analyze-video", methods=['POST'])
def analyze_video():
    if 'video' not in request.files: return jsonify({"message": "No file"}), 400
    file = request.files['video']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        physics_score, debug_msg = ForensicEngine.analyze_video_physics(filepath)
        verdict = "Deepfake (Unnatural Physics)" if physics_score > 0.6 else "Natural Motion"

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO analysis_log (filename, file_type, verdict, score, details) VALUES (?, ?, ?, ?, ?)",
                       (file.filename, 'video', verdict, physics_score, debug_msg))
        conn.commit()

        return jsonify({
            "verdict": verdict,
            "score": f"{physics_score*100:.1f}% Anomaly",
            "frame_summary": debug_msg,
            "log_id": cursor.lastrowid
        })
    except Exception as e:
        return jsonify({"message": f"Error: {e}"}), 500

@app.route("/api/analyze-text", methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', '')
    phishing_triggers = ["urgent", "verify", "bank", "password"]
    found_triggers = [w for w in triggers if w in text.lower()]
    score = 0.6 if found_triggers else 0.1
    verdict = "Suspicious / Phishing" if score > 0.5 else "Safe Text"
    
    return jsonify({
        "verdict": verdict,
        "score": f"{score*100:.1f}% Risk",
        "homoglyph_details": f"Triggers: {found_triggers}",
        "invisible_details": "None",
        "log_id": 0
    })

@app.route("/api/feedback", methods=['POST'])
def feedback():
    data = request.get_json()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE analysis_log SET feedback = ? WHERE id = ?", (data.get('feedback'), data.get('log_id')))
    conn.commit()
    return jsonify({"message": "Feedback recorded."})

if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)
