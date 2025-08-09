import requests
import time
import pdfplumber
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import pickle
import torch

AZURE_OCR_ENDPOINT = "https://<your-region>.api.cognitive.microsoft.com/"
AZURE_OCR_KEY = "<your-azure-computer-vision-key>"

model = DistilBertForSequenceClassification.from_pretrained("./distilbert_doc_classifier")
tokenizer = DistilBertTokenizerFast.from_pretrained("./distilbert_doc_classifier")

with open("./distilbert_doc_classifier/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

def azure_read_ocr(pdf_path):

    with open(pdf_path, "rb") as f:
        data_bytes = f.read()

    ocr_url = AZURE_OCR_ENDPOINT.rstrip("/") + "/vision/v3.2/read/analyze"

    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_OCR_KEY,
        "Content-Type": "application/pdf",
    }

    response = requests.post(ocr_url, headers=headers, data=data_bytes)
    if response.status_code != 202:
        raise Exception(f"Azure OCR failed: {response.status_code}, {response.text}")
    operation_url = response.headers["Operation-Location"]

    analysis = {}
    while True:
        result_response = requests.get(operation_url, headers={"Ocp-Apim-Subscription-Key": AZURE_OCR_KEY})
        analysis = result_response.json()
        status = analysis["status"]

        if status == "succeeded":
            break
        elif status == "failed":
            raise Exception("Azure OCR read operation failed")
        else:
            time.sleep(1)  

    extracted_text = ""
    for read_result in analysis["analyzeResult"]["readResults"]:
        for line in read_result["lines"]:
            extracted_text += line["text"] + "\n"

    return extracted_text

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class_id = outputs.logits.argmax().item()
    return label_encoder.inverse_transform([predicted_class_id])[0]

# Example usage
if __name__ == "__main__":
    pdf_file_path = "meidcal.pdf"  
    extracted_text = azure_read_ocr(pdf_file_path)
    print(f"Extracted Text:\n{extracted_text[:500]}...") 

    prediction = predict(extracted_text)
    print(f"Predicted class: {prediction}")