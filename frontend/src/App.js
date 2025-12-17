import React, { useState } from 'react';
import './App.css';

function App() {
  // States for Image Analyzer
  const [selectedFile, setSelectedFile] = useState(null);
  const [imageResult, setImageResult] = useState(null);
  const [isImageLoading, setIsImageLoading] = useState(false);
  const [imageFeedback, setImageFeedback] = useState(0); // 0=none, 1=correct, -1=incorrect

  // States for Text Analyzer
  const [textInput, setTextInput] = useState('');
  const [isTextLoading, setIsTextLoading] = useState(false);
  const [textResult, setTextResult] = useState(null);
  const [textFeedback, setTextFeedback] = useState(0);

  // States for Video Analyzer (NEW!)
  const [selectedVideoFile, setSelectedVideoFile] = useState(null);
  const [videoResult, setVideoResult] = useState(null);
  const [isVideoLoading, setIsVideoLoading] = useState(false);
  const [videoFeedback, setVideoFeedback] = useState(0);


  // --- Image Handlers ---
  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setImageResult(null); 
    setImageFeedback(0); 
  };

  const handleImageUpload = () => {
    if (!selectedFile) { alert("Please select a file first!"); return; }
    const formData = new FormData();
    formData.append('image', selectedFile);
    setIsImageLoading(true);
    setImageFeedback(0);
    fetch('http://localhost:5000/api/analyze-image', { method: 'POST', body: formData })
      .then(res => res.json())
      .then(data => { 
        if (data.verdict) {
          setImageResult(data);
        } else {
          setImageResult({ verdict: "Error", score: data.message, metadata_details: "", steganography_details: "" });
        }
      })
      .catch(err => setImageResult({ verdict: "Error", score: "Could not connect to server.", metadata_details: "", steganography_details: "" }))
      .finally(() => setIsImageLoading(false));
  };

  // --- Text Handlers ---
  const handleTextChange = (event) => {
    setTextInput(event.target.value);
  };

  const handleTextAnalysis = () => {
    if (!textInput.trim()) { alert("Please enter some text!"); return; }
    setIsTextLoading(true);
    setTextResult(null);
    setTextFeedback(0);
    fetch('http://localhost:5000/api/analyze-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: textInput }),
    })
    .then(res => res.json())
    .then(data => {
      if (data.verdict) {
        setTextResult(data);
      } else {
        setTextResult({ verdict: "Error", score: data.message, homoglyph_details: "", invisible_details: "" });
      }
    })
    .catch(err => setTextResult({ verdict: "Error", score: "Could not connect to server.", homoglyph_details: "", invisible_details: "" }))
    .finally(() => setIsTextLoading(false));
  };

  // --- Video Handlers (NEW!) ---
  const handleVideoFileChange = (event) => {
    setSelectedVideoFile(event.target.files[0]);
    setVideoResult(null);
    setVideoFeedback(0);
  };

  const handleVideoUpload = () => {
    if (!selectedVideoFile) { alert("Please select a video file first!"); return; }
    const formData = new FormData();
    formData.append('video', selectedVideoFile);
    setIsVideoLoading(true);
    setVideoFeedback(0);
    fetch('http://localhost:5000/api/analyze-video', { method: 'POST', body: formData })
      .then(res => res.json())
      .then(data => { 
        if (data.verdict) {
          setVideoResult(data);
        } else {
          setVideoResult({ verdict: "Error", score: data.message, frame_summary: "" });
        }
      })
      .catch(err => setVideoResult({ verdict: "Error", score: "Could not connect to server.", frame_summary: "" }))
      .finally(() => setIsVideoLoading(false));
  };


  // --- Feedback Handler (Updated for video) ---
  const handleFeedback = (log_id, feedback, type) => {
    fetch('http://localhost:5000/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ log_id: log_id, feedback: feedback }),
    })
    .then(res => res.json())
    .then(data => {
      console.log(data.message);
      if (type === 'image') {
        setImageFeedback(feedback);
      } else if (type === 'text') {
        setTextFeedback(feedback);
      } else if (type === 'video') {
        setVideoFeedback(feedback);
      }
    })
    .catch(err => console.error("Feedback error:", err));
  };


  return (
    <div className="App">
      <header className="App-header">
        <h1>HUM-Ray: AI Threat Analyzer</h1>
        <div className="analyzers-container">
          
          {/* Image Analyzer Box */}
          <div className="analyzer-box">
            <h2>Image Analyzer</h2>
            <p className="subtitle">Check images for AI generation, hidden data, and editing history.</p>
            <div className="uploader">
              <input type="file" onChange={handleFileChange} accept="image/png, image/jpeg" />
              <button onClick={handleImageUpload} disabled={isImageLoading}>
                {isImageLoading ? 'Analyzing...' : 'Analyze Image'}
              </button>
            </div>
            
            {imageResult && (
              <div className="result-container">
                <h3>Analysis Result</h3>
                <div className="result-grid">
                  <p><strong>Verdict:</strong></p>
                  <p><span className={imageResult.verdict === 'AI-Generated' ? 'ai' : 'real'}>{imageResult.verdict}</span></p>

                  <p><strong>AI Confidence:</strong></p>
                  <p><span>{imageResult.score}</span></p>

                  <p><strong>Metadata:</strong></p>
                  <p><span>{imageResult.metadata_details}</span></p>

                  <p><strong>Steganography:</strong></p>
                  <p><span>{imageResult.steganography_details}</span></p>
                </div>
                
                {imageResult.log_id && imageFeedback === 0 && (
                  <div className="feedback-section">
                    <p>Was this result correct?</p>
                    <button className="feedback-btn correct" onClick={() => handleFeedback(imageResult.log_id, 1, 'image')}>👍 Correct</button>
                    <button className="feedback-btn incorrect" onClick={() => handleFeedback(imageResult.log_id, -1, 'image')}>👎 Incorrect</button>
                  </div>
                )}
                
                {imageFeedback !== 0 && (
                  <p className="feedback-thanks">Thank you for your feedback!</p>
                )}
              </div>
            )}
          </div>

          {/* Text Analyzer Box */}
          <div className="analyzer-box">
            <h2>Text & Email Analyzer</h2>
            <p className="subtitle">Scan text for malicious links, phishing, and invisible characters.</p>
            <div className="uploader">
              <textarea
                value={textInput}
                onChange={handleTextChange}
                placeholder="Paste text or an email body here..."
                rows="5"
              />
              <button onClick={handleTextAnalysis} disabled={isTextLoading}>
                {isTextLoading ? 'Analyzing...' : 'Analyze Text'}
              </button>
            </div>

            {textResult && (
              <div className="result-container">
                <h3>Analysis Result</h3>
                <div className="result-grid">
                  <p><strong>Verdict:</strong></p>
                  <p><span className={textResult.verdict === 'Suspicious (AI)' ? 'ai' : 'real'}>{textResult.verdict}</span></p>

                  <p><strong>AI Confidence:</strong></p>
                  <p><span>{textResult.score}</span></p>

                  <p><strong>Phishing (Scan):</strong></p>
                  <p><span>{textResult.homoglyph_details}</span></p>

                  <p><strong>Hidden Chars:</strong></p>
                  <p><span>{textResult.invisible_details}</span></p>
                </div>
                
                {textResult.log_id && textFeedback === 0 && (
                  <div className="feedback-section">
                    <p>Was this result correct?</p>
                    <button className="feedback-btn correct" onClick={() => handleFeedback(textResult.log_id, 1, 'text')}>👍 Correct</button>
                    <button className="feedback-btn incorrect" onClick={() => handleFeedback(textResult.log_id, -1, 'text')}>👎 Incorrect</button>
                  </div>
                )}
                
                {textFeedback !== 0 && (
                  <p className="feedback-thanks">Thank you for your feedback!</p>
                )}
              </div>
            )}
          </div>

          {/* Video Analyzer Box (NOW ACTIVE!) */}
          <div className="analyzer-box">
            <h2>Video Analyzer</h2>
            <p className="subtitle">Analyze videos for deepfakes and manipulated content.</p>
            <div className="uploader">
              <input type="file" onChange={handleVideoFileChange} accept="video/mp4, video/quicktime" />
              <button onClick={handleVideoUpload} disabled={isVideoLoading}>
                {isVideoLoading ? 'Analyzing...' : 'Analyze Video'}
              </button>
            </div>
            
            {videoResult && (
              <div className="result-container">
                <h3>Analysis Result</h3>
                <div className="result-grid video-grid">
                  <p><strong>Verdict:</strong></p>
                  <p><span className={videoResult.verdict === 'Probable Deepfake' ? 'ai' : 'real'}>{videoResult.verdict}</span></p>

                  <p><strong>Avg. Confidence:</strong></p>
                  <p><span>{videoResult.score}</span></p>

                  <p><strong>Frame Summary:</strong></p>
                  <p><span>{videoResult.frame_summary}</span></p>
                </div>
                
                {videoResult.log_id && videoFeedback === 0 && (
                  <div className="feedback-section">
                    <p>Was this result correct?</p>
                    <button className="feedback-btn correct" onClick={() => handleFeedback(videoResult.log_id, 1, 'video')}>👍 Correct</button>
                    <button className="feedback-btn incorrect" onClick={() => handleFeedback(videoResult.log_id, -1, 'video')}>👎 Incorrect</button>
                  </div>
                )}
                
                {videoFeedback !== 0 && (
                  <p className="feedback-thanks">Thank you for your feedback!</p>
                )}
              </div>
            )}
          </div>

        </div>
      </header>
    </div>
  );
}

export default App;