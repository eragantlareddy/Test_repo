import pandas as pd
from fpdf import FPDF
import fitz  
import streamlit as st
import json
from langchain_community.llms import Ollama
import os
import pytesseract
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import pdfkit
 
# Initialize Ollama model
llm = Ollama(model="llama3.2:latest", base_url="http://localhost:11434")
 
# Initialize output directory
output_dir = Path("documents")
output_dir.mkdir(exist_ok=True)
 
# Function for font and image analysis
def detect_fonts_with_ocr(pdf_file):
    # Open the PDF
    doc = fitz.open(pdf_file)
   
    # Initialize analysis containers
    font_analysis = {
        'total_pages': len(doc),
        'small_fonts': 0,
        'large_fonts': 0,
        'total_text_elements': 0,
        'pages_font_details': []
    }
   
    # Process each page
    for page_num in range(len(doc)):
        # Get page
        page = doc[page_num]
       
        # Convert page to image
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
       
        # Preprocess image
        opencv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
       
        # Perform OCR with detailed configuration
        details = pytesseract.image_to_data(
            thresh,
            output_type=pytesseract.Output.DICT,
            config='--psm 6 -c preserve_interword_spaces=1'
        )
       
        # Page-specific font details
        page_font_details = {
            'page_number': page_num + 1,
            'small_fonts': 0,
            'large_fonts': 0,
            'total_text_elements': 0
        }
       
        # Analyze font sizes from OCR results
        for font_size in details['height']:
            # Filter out noise and very small values
            if 5 < font_size < 100:
                page_font_details['total_text_elements'] += 1
                font_analysis['total_text_elements'] += 1
               
                # Categorize font sizes
                if font_size < 10:
                    page_font_details['small_fonts'] += 1
                    font_analysis['small_fonts'] += 1
                else:
                    page_font_details['large_fonts'] += 1
                    font_analysis['large_fonts'] += 1
       
        font_analysis['pages_font_details'].append(page_font_details)
   
    # Calculate percentages
    total_text_elements = font_analysis['total_text_elements']
    font_analysis['small_fonts_percentage'] = (font_analysis['small_fonts'] / total_text_elements * 100) if total_text_elements else 0
    font_analysis['large_fonts_percentage'] = (font_analysis['large_fonts'] / total_text_elements * 100) if total_text_elements else 0
   
    return font_analysis
 
def excel_to_pdf_content_only(excel_file, output_pdf):
    df = pd.read_excel(excel_file)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
   
    for index, row in df.iterrows():
        content_line = " ".join([str(cell) for cell in row if pd.notnull(cell)])
        pdf.multi_cell(0, 10, content_line)
        pdf.ln(5)
   
    pdf.output(output_pdf)
 
# Text extraction functions (as in original code)
def extract_pdf_text(pdf_file):
    doc = fitz.open(pdf_file)
    all_text = []
   
    for page_number, page in enumerate(doc):
        text = page.get_text("text")
        images = len(page.get_images(full=True))
       
        if not text:
            text = ocr_pdf_page(page)
       
        all_text.append({
            "page_number": page_number + 1,
            "content": text,
            "images": images
        })
   
    return all_text
 
def ocr_pdf_page(page):
    img = page.get_pixmap()
    pil_image = Image.frombytes("RGB", (img.width, img.height), img.samples)
    text = pytesseract.image_to_string(pil_image)
    return text
 
# JSON parsing and analysis functions
import json
 
def parse_pdf_to_json(all_text):
    if not all_text:
        return {"error": "No content found in PDF"}
 
    # Let's divide the content by pages and send one page at a time.
    json_data = {"pages": []}
 
    for page_number, page_content in enumerate(all_text):
        # Create a prompt for each page
        prompt = f"""
        You are helping convert a PDF document into a structured JSON format.
        Please output the following content in valid JSON format (without markdown or backticks):
        """
 
        try:
            # Log the current page's prompt for debugging
            print(f"\nSending the following prompt for page {page_number + 1} to Ollama:")
 
            # Send the prompt to Ollama
            response = llm.invoke(prompt)
 
            # Log the raw response for this page
            print(f"\nRaw response from Ollama for page {page_number + 1}:")
            print(response)
 
            if not response:
                # Handle empty response
                print(f"\nEmpty response for page {page_number + 1}.")
                continue
 
            # Clean the response (remove unwanted markdown, if any)
            cleaned_response = response.replace("```json", "").replace("```", "").strip()
 
            # Parse the cleaned response
            try:
                page_json = json.loads(cleaned_response)
                json_data["pages"].append(page_json)
            except json.JSONDecodeError as e:
                print(f"\nError parsing JSON for page {page_number + 1}: {str(e)}")
                continue
 
        except Exception as e:
            print(f"\nError processing page {page_number + 1}: {str(e)}")
            continue
 
    if not json_data["pages"]:
        return {"error": "Failed to parse any pages into JSON."}
 
    return json_data
 
def calculate_text_and_image_percentage_from_json(json_data, pdf_file):
    page_data_list = []
 
    # Perform font analysis
    font_analysis = detect_fonts_with_ocr(pdf_file)
 
    # Analyze pages
    for idx, page_data in enumerate(json_data.get('pages', [])):
        page_number = page_data.get('page_number')
        content = page_data.get('content', "")
        images = page_data.get('images', 0)
 
        text_length = len(content)
        image_count = images
       
        # Total content calculation
        total_content = text_length + image_count * 1000
        text_percentage = (text_length / total_content) * 100 if total_content else 0
        image_percentage = (image_count * 1000 / total_content) * 100 if total_content else 0
 
        # Get font details for this page
        page_font_details = font_analysis['pages_font_details'][idx] if idx < len(font_analysis['pages_font_details']) else {}
 
        page_data_list.append({
            "page_number": page_number,
            "text_percentage": text_percentage,
            "image_percentage": image_percentage,
            "image_count": image_count,
            "small_fonts_percentage": page_font_details.get('small_fonts', 0) / page_font_details.get('total_text_elements', 1) * 100,
            "large_fonts_percentage": page_font_details.get('large_fonts', 0) / page_font_details.get('total_text_elements', 1) * 100
        })
   
    return page_data_list
 
# Streamlit UI
def main():
    st.title("Smart Converter 🤖")
 
    uploaded_files = st.file_uploader(
        "Upload HTML, Excel, or PDF files (Maximum 5 files)",
        type=["html", "xlsx", "pdf"],
        accept_multiple_files=True
    )
 
    if uploaded_files:
        if len(uploaded_files) > 5:
            st.error("Please upload no more than 5 files.")
        else:
            pdf_files = []
            for uploaded_file in uploaded_files:
                pdf_file = output_dir / f"{uploaded_file.name.split('.')[0]}.pdf"
                if uploaded_file.type == "application/pdf":
                    with open(pdf_file, "wb") as f:
                        f.write(uploaded_file.read())
                elif uploaded_file.type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                    excel_to_pdf_content_only(uploaded_file, str(pdf_file))
                elif uploaded_file.type == "text/html":
                    html_file_path = output_dir / uploaded_file.name
                    with open(html_file_path, "wb") as f:
                        f.write(uploaded_file.read())
                   
                    html_to_pdf_with_pdfkit(str(html_file_path), str(pdf_file))
 
                pdf_files.append(pdf_file)
 
            st.write(f"Converted {len(pdf_files)} files to PDF.")
 
            if st.button("Convert PDFs to JSON and Analyze Content"):
                if not pdf_files:
                    st.error("No PDF files to analyze.")
                else:
                    json_outputs = []
                    for pdf_file in pdf_files:
                        all_text = extract_pdf_text(pdf_file)
                        json_output = parse_pdf_to_json(all_text)
                       
                        if "error" in json_output:
                            st.error(f"Error processing {pdf_file}: {json_output['error']}")
                        else:
                            page_data_list = calculate_text_and_image_percentage_from_json(json_output, str(pdf_file))
                            json_output["pages_info"] = page_data_list
                            json_outputs.append({pdf_file.name: json_output})
 
                    for json_output in json_outputs:
                        for pdf_file, json_data in json_output.items():
                            st.write(f"JSON for {pdf_file}:")
                            if isinstance(json_data, dict):
                                st.json(json_data)
                            else:
                                st.error(json_data)
 
                            json_file = output_dir / f"{pdf_file.split('.')[0]}_output.json"
                            with open(json_file, "w") as f:
                                json.dump(json_data, f, indent=4)
                            st.write(f"Saved JSON output for {pdf_file} at {json_file}")
    else:
        st.write("Upload HTML or Excel files to begin conversion.")
 
if __name__ == "__main__":
    main()