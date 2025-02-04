"""
Financial Data Ingestion Module
================================
This module processes PDF financial statements by extracting text and structured data using LLM-based methods.

Features include:
- Converts PDF pages into images for processing.
- Extracts structured financial data from images using OpenAI Vision.
- Saves extracted data in JSON format for further processing.
- Supports optional integration with S3 for cloud storage.
- Enables storage in PySpark for scalability in big data processing.

This module ensures accurate data extraction from financial statements while maintaining structured formatting for downstream analytics.
"""

import os
import io
import json
import base64
import logging
import shutil
import boto3
from pdf2image import convert_from_path
from openai import OpenAI
from pyspark.sql import SparkSession
from botocore.exceptions import NoCredentialsError
from config import (
    PDF_FOLDER,
    RAW_TEXT_FOLDER,
    S3_BUCKET_NAME,
    USE_SPARK,
    USE_S3,
    SPARK_OUTPUT_FOLDER,
    OPENAI_API_KEY,
    MODEL_NAME_PDF,
    SYSTEM_PROMPT_PDF,
    MODEL_TOKEN_PDF
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Spark session if enabled
if USE_SPARK:
    spark = SparkSession.builder.appName("FinancialDataProcessing").getOrCreate()

# Initialize S3 client if enabled
if USE_S3:
    s3_client = boto3.client("s3")


class FinancialDataIngestion:
    def __init__(self):
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)

    def convert_pdf_to_images(self, pdf_path):
        """Converts a PDF file into a list of images."""
        return convert_from_path(pdf_path)

    def encode_image_to_base64(self, img):
        """Encodes a PIL image to a Base64-encoded string."""
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

    def extract_text_from_images(self, images):
        """Uses OpenAI Vision to extract text from images."""
        img_data = [{"type": "image_url", "image_url": {"url": self.encode_image_to_base64(img)}} for img in images]
        response = self.openai_client.chat.completions.create(
            model=MODEL_NAME_PDF,
            messages=[{"role": "system", "content": SYSTEM_PROMPT_PDF}, {"role": "user", "content": img_data}],
            max_tokens=MODEL_TOKEN_PDF,
            temperature=0,
            top_p=0.1
        )
        if not response or not response.choices or not response.choices[0].message.content.strip():
            logging.error("OpenAI API returned an empty response")
            return "{}"
        extracted_text = response.choices[0].message.content.strip()
        if extracted_text.startswith("```json"):
            extracted_text = extracted_text[7:]
        if extracted_text.endswith("```"):
            extracted_text = extracted_text[:-3]
        return extracted_text

    def process_pdf(self, pdf_path):
        """Processes a single PDF and extracts structured JSON data."""
        images = self.convert_pdf_to_images(pdf_path)
        extracted_text = self.extract_text_from_images(images)

        try:
            structured_data = json.loads(extracted_text) if extracted_text else {}
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing failed: {e}, raw data: {extracted_text}")
            return

        output_file = os.path.join(RAW_TEXT_FOLDER, os.path.basename(pdf_path).replace(".pdf", ".json"))
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, indent=4, ensure_ascii=False)
        logging.info(f"Data saved locally: {output_file}")

        if USE_SPARK:
            self.save_to_spark(structured_data, output_file)
        if USE_S3:
            self.upload_to_s3(output_file)

    def save_to_spark(self, data, file_name):
        """Stores data into a Spark DataFrame."""
        df = spark.createDataFrame([(file_name, json.dumps(data))], ["pdf_name", "content"])
        df.write.mode("overwrite").parquet(os.path.join(SPARK_OUTPUT_FOLDER, f"{file_name}.parquet"))
        logging.info(f"Data stored in Spark: {SPARK_OUTPUT_FOLDER}/{file_name}.parquet")

    def upload_to_s3(self, file_path):
        """Uploads JSON data to S3."""
        try:
            s3_client.upload_file(file_path, S3_BUCKET_NAME, os.path.basename(file_path))
            logging.info(f"Successfully uploaded to S3: {S3_BUCKET_NAME}/{os.path.basename(file_path)}")
        except NoCredentialsError:
            logging.error("AWS S3 authentication failed!")

    def process_all_pdfs(self):
        """Processes all PDF files in the specified folder."""
        os.makedirs(RAW_TEXT_FOLDER, exist_ok=True)
        pdf_files = [os.path.join(PDF_FOLDER, f) for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
        for pdf in pdf_files:
            logging.info(f"Processing file: {pdf}")
            self.process_pdf(pdf)


if __name__ == "__main__":
    ingestion = FinancialDataIngestion()
    ingestion.process_all_pdfs()
