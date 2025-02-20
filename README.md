# Text Summarization Project

## Overview

This project is a **Text Summarization Web Application** that utilizes a fine-tuned **PEGASUS** model for text summarization. It consists of a **Flask-based backend** for handling summarization requests and a **React-based frontend** for user interaction.

## Features

- Summarizes input text using a fine-tuned PEGASUS model.
- Simple and intuitive UI for text input and summarization.
- Flask API backend for model inference.
- Supports GPU acceleration if available.

## Project Structure

```
Text Summarization Project
│── backend
│   │── app.py                 # Flask backend
│   │── requirements.txt       # Dependencies for backend
│── frontend
│   │── public
│   │   └── index.html         # Basic HTML template
│   │── src
│   │   ├── App.js             # Main React Component
│   │   ├── index.js           # React Entry Point
│   │   ├── package.json       # Frontend Dependencies
│   │   ├── package-lock.json  # Package Lock File
│── Text-Summarization-Fine-tuning-Transformers-model.ipynb # Model Fine-Tuning Notebook
```

## Setup and Installation

### Backend Setup

#### Prerequisites:

- Python 3.8+
- CUDA (Optional for GPU acceleration)
- Virtual environment (recommended)

#### Steps:

1. Navigate to the backend folder:
   ```sh
   cd backend
   ```
2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv venv

   # On Linux use:
   source venv/bin/activate  
   
   # On Windows use: 
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```


### Option 1: Use a Pretrained Model

If you don’t want to retrain the model, you can download a pretrained PEGASUS model along with the tokenizer from our Google Drive link and use it directly for inference.

#### Steps:

1. Download the pretrained model and tokenizer from [Google Drive](https://drive.google.com/drive/folders/17arjGx9-NPU-AdxjuZGrGkwvYNlzjZcn).
2. Put it in the main folder and extract the files.
3. Run the backend server in backend folder:
   ```sh
   # If you in the main folder
   cd backend 
   python app.py
   ```
   The backend will start at `http://localhost:5000/`



### Option 2: Retrain the Model

If you want to fine-tune the PEGASUS model yourself, you can either run the training commands directly in your terminal or execute the provided Jupyter Notebook (`Text-Summarization-Fine-tuning-Transformers-model.ipynb`). The notebook contains step-by-step training instructions, ensuring that you generate a new model and checkpoints for inference.

1. Make model file and download the models using the following terminal commands:
   ```sh
   mkdir backend/models && cd backend/models
   wget https://huggingface.co/google/pegasus-cnn_dailymail/resolve/main/pytorch_model.bin
   wget https://huggingface.co/google/pegasus-cnn_dailymail/resolve/main/config.json
   wget https://huggingface.co/google/pegasus-cnn_dailymail/resolve/main/spiece.model
   cd ../..
   ```

2. Key Libraries Install:

The **fine-tuning process** is documented in `Text-Summarization-Fine-tuning-Transformers-model.ipynb` using **Hugging Face Transformers** and the **SAMSum dataset**.

    ```sh
    pip install transformers[sentencepiece] datasets sacrebleu rouge_score py7zr
    ```

3. Run Jupyter Notebook :
    ```sh
    python Text-Summarization-Fine-tuning-Transformers-model.ipynb  
    ```

What it does:
- Load **SAMSum dataset** (`datasets.load_dataset("samsum")`)
- Tokenize input text and summaries.
- Fine-tune `google/pegasus-cnn_dailymail` using **Hugging Face Trainer**.
- Evaluate using **ROUGE metrics**.
- Save the model and tokenizer for deployment.

4. Run the backend server in backend folder:
   ```sh
   # If you in the main folder
   cd backend 
   python app.py
   ```
   The backend will start at `http://localhost:5000/`


#### API Endpoint

- **Endpoint:** `POST /summarize`
- **Request Body:**
  ```json
  {
    "text": "Your text to summarize here"
  }
  ```
- **Response:**
  ```json
  {
    "summary": "Generated summary here"
  }
  ```


### Frontend Setup
Open a new terminal

#### Prerequisites:
- Node.js and npm
   ```sh
      # Recommend install in the same environment as backend
      # On window 10/11:
      cd backend
      python -m venv venv
      venv\Scripts\activate

      # Install Node.js and npm
      winget install OpenJS.NodeJS

      # Check
      node -v
      npm -v

      # If got blocked by shell: try these code then reset code terminal if needed
      Get-Command node -ErrorAction SilentlyContinue
      Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```


#### Steps:
1. Navigate to the frontend folder:
   ```sh
   cd ..
   cd frontend
   ```
2. Install dependencies:
   ```sh
   npm install
   ```
3. Start the React frontend:
   ```sh
   npm start
   ```
   The frontend will be available at `http://localhost:3000/`



## References

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [SAMSum Dataset](https://huggingface.co/datasets/samsum)
- [PEGASUS Paper](https://arxiv.org/abs/1912.08777)

## Future Improvements

- Deploy on a cloud platform (AWS/GCP/Heroku).
- Add user authentication.
- Support multilingual summarization.
- Optimize model for faster inference.

---

**Author:** Group 21\
**Date:** February 2025