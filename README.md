# HUM-Ray: AI Threat Analyzer 🕵️‍♂️

**HUM-Ray** (Human vs. X-Ray) is a full-stack forensic tool designed to detect AI-generated content. It uses a custom-trained Deep Learning "brain" along with traditional cybersecurity forensics to analyze images, text, and video.

## 🚀 Features

* **Image Forensics:** Detects AI-generated images, analyzes EXIF metadata, and scans for steganography (hidden data).
* **Text & Email Scanner:** Detects hidden invisible characters and homoglyph attacks (fake letters used in phishing).
* **Video Analysis:** Scans video frames to detect deepfakes.
* **Database Logging:** Automatically saves every analysis result to a local database for review.
* **Feedback Loop:** Users can flag incorrect results to help train future AI versions.

---

## 🛠️ Prerequisites (What to Install First)

Before you start, you need these two free tools installed on your computer:

1. **Python** (For the backend server): [Download Python](https://www.python.org/downloads/)
* *Important:* During installation, check the box **"Add Python to PATH"**.


2. **Node.js** (For the frontend interface): [Download Node.js](https://nodejs.org/) (Choose the "LTS" version).

---

## 📥 Installation Guide

Follow these steps exactly to set up the project.

### Step 1: Download the Project

Open your terminal (Command Prompt or PowerShell) and run:

```bash
git clone https://github.com/YOUR_USERNAME/hum-ray.git
cd hum-ray

```

*(Replace `YOUR_USERNAME` with your actual GitHub username)*

### Step 2: Set Up the Backend (The Brain)

This sets up the Python server and installs the AI libraries.

1. Navigate to the backend folder:
```bash
cd backend

```


2. Create a virtual environment (a safe space for dependencies):
* **Windows:**
```bash
python -m venv venv

```


* **Mac/Linux:**
```bash
python3 -m venv venv

```




3. Activate the environment:
* **Windows:**
```bash
.\venv\Scripts\activate

```


* **Mac/Linux:**
```bash
source venv/bin/activate

```




*(You should see `(venv)` appear at the start of your command line).*
4. Install the required libraries:
```bash
pip install flask flask-cors tensorflow pillow stegano opencv-python transformers

```



### Step 3: Set Up the Frontend (The Interface)

Open a **new** terminal window (keep the backend one open) and navigate to the project folder.

1. Go to the frontend folder:
```bash
cd frontend

```


2. Install the React dependencies:
```bash
npm install

```



---

## 🚦 How to Run the App

You need two terminals running at the same time.

### Terminal 1: Start the Backend Server

Make sure you are in the `backend` folder and your `(venv)` is active.

```bash
python server.py

```

*You will see a message: `✅ Real Image AI Model Built and Weights Loaded...*`
*The server is now running at `http://127.0.0.1:5000`.*

### Terminal 2: Start the Frontend Interface

Make sure you are in the `frontend` folder.

```bash
npm start

```

*This will automatically open your web browser to `http://localhost:3000`.*

---

## 🧪 How to Use It

1. **Image Analyzer:**
* Click "Choose File" and upload an image (JPG/PNG).
* Click "Analyze Image."
* **Result:** It will tell you if it's "AI-Generated" or "Looks Real" and show any hidden metadata.


2. **Text Analyzer:**
* Paste any suspicious text or email content.
* Click "Analyze Text."
* **Result:** It checks for "invisible characters" (often used in hacks) and fake letters (homoglyphs).


3. **Video Analyzer:**
* Upload a short video (MP4).
* **Result:** It extracts frames and checks if they appear to be deepfakes.



---

## 🐛 Troubleshooting

* **Error: `Module not found`?**
* Make sure you activated the virtual environment (`.\venv\Scripts\activate`) before running `server.py`.


* **Backend crashes on startup?**
* Ensure the file `humray_model_v2_weights.weights.h5` exists in the `backend` folder.


* **Frontend says "Network Error"?**
* Make sure the backend server (Terminal 1) is running.



---

**Built with ❤️ using Python (Flask) and React.**
