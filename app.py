# import pdfplumber
# import pytesseract
# from PIL import Image
# import io
# import os
# from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
# import pickle
# import torch

# # Load model, tokenizer, label encoder as before
# model = DistilBertForSequenceClassification.from_pretrained("./distilbert_doc_classifier")
# tokenizer = DistilBertTokenizerFast.from_pretrained("./distilbert_doc_classifier")

# with open("./distilbert_doc_classifier/label_encoder.pkl", "rb") as f:
#     label_encoder = pickle.load(f)

# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text + "\n"

#     # If extracted text is too small, fall back to OCR
#     if len(text.strip()) < 20:
#         print("Using OCR fallback...")
#         text = ""
#         with pdfplumber.open(pdf_path) as pdf:
#             for page in pdf.pages:
#                 # Convert page to image
#                 pil_image = page.to_image(resolution=300).original
#                 ocr_text = pytesseract.image_to_string(pil_image)
#                 text += ocr_text + "\n"

#     return text

# def predict(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
#     model.eval()
#     with torch.no_grad():
#         outputs = model(**inputs)
#     predicted_class_id = outputs.logits.argmax().item()
#     return label_encoder.inverse_transform([predicted_class_id])[0]

# # Example usage
# if __name__ == "__main__":
#     pdf_file_path = "medical.pdf"  # replace with your PDF file path
#     extracted_text = extract_text_from_pdf(pdf_file_path)
#     print(f"Extracted Text:\n{extracted_text[:500]}...")  # print first 500 chars
    
#     prediction = predict(extracted_text)
#     print(f"Predicted class: {prediction}")



import streamlit as st
import pdfplumber
import pytesseract
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import pickle
import torch
from PIL import Image
import io

# Load model, tokenizer, label encoder once
@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("./distilbert_doc_classifier")
    tokenizer = DistilBertTokenizerFast.from_pretrained("./distilbert_doc_classifier")
    with open("./distilbert_doc_classifier/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_model()

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    if len(text.strip()) < 20:
        # OCR fallback
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                pil_image = page.to_image(resolution=300).original
                ocr_text = pytesseract.image_to_string(pil_image)
                text += ocr_text + "\n"
    return text

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class_id = outputs.logits.argmax().item()
    return label_encoder.inverse_transform([predicted_class_id])[0]

st.title("Document Classification with OCR")

option = st.radio("Choose input type:", ("Upload PDF Document", "Paste Text"))

if option == "Upload PDF Document":
    uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)
        st.text_area("Extracted Text (first 1000 chars):", text[:1000], height=200)
        
        if st.button("Predict from extracted text"):
            with st.spinner("Predicting..."):
                prediction = predict(text)
            st.success(f"Predicted Class: {prediction}")

elif option == "Paste Text":
    user_text = st.text_area("Paste your text here", height=300)
    if user_text and st.button("Predict"):
        with st.spinner("Predicting..."):
            prediction = predict(user_text)
        st.success(f"Predicted Class: {prediction}")
