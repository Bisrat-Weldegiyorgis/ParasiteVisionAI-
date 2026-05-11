# 🦠 ParasiteVisionAI

ParasiteVisionAI is a deep learning-based web application for detecting parasites from medical images. It uses a trained PyTorch model and provides an easy-to-use Streamlit interface for real-time predictions.

---

## 🚀 Features

- 🧠 Deep learning model for parasite detection
- 🖼️ Image upload and prediction interface
- ⚡ Fast inference using PyTorch
- 🌐 Web-based UI using Streamlit
- 📊 Clean and simple results display

---

## 📁 Project Structure
ParasiteVisionAI-/
│
├── app.py # Streamlit web application
├── model.py # Model loading and inference logic
├── dataset.py # Dataset class and preprocessing (training only)
├── parasite_model.pth # Trained PyTorch model
├── requirements.txt # Project dependencies
└── README.md # Project documentation


---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/ParasiteVisionAI-.git
cd ParasiteVisionAI-

🧠 Model Information
Framework: PyTorch
Input: Medical image (parasite sample)
Output: Predicted parasite class
File: parasite_model.pth
📌 Notes
Dataset files are not included in this repository.
The model must be present in the project root for the app to run.
dataset.py is used only for training, not for deployment.

🚀 Deployment

This project is designed to run on:

Streamlit Cloud
Local machine
Any Python-supported server
⚠️ Important
Do NOT upload raw dataset images to GitHub
Keep sensitive medical data private
Ensure parasite_model.pth is included or properly linked
👨‍💻 Developed by
Bisrat Weldegiyorgis

📜 License

This project is for educational and research purposes only.
