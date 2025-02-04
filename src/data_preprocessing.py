"""
Financial Data Processing Module
================================
This module processes structured JSON data extracted from financial statements,
ensuring consistency and preparing it for further analysis.

Features include:
- Validating the extracted JSON data structure to ensure completeness.
- Normalizing missing values and ensuring numerical consistency.
- Structuring data for downstream financial analysis and reporting.
- Storing processed data in Spark for scalability and optionally uploading it to S3 for cloud storage.

This module ensures that extracted financial data is cleaned, structured, and ready for advanced processing.
"""

import os
import json
import logging
import boto3
from pyspark.sql import SparkSession
from typing import Dict, Optional
from config import (
    RAW_TEXT_FOLDER,
    PROCESSED_DATA_FOLDER,
    USE_SPARK,
    USE_S3,
    S3_BUCKET_NAME,
    SPARK_OUTPUT_FOLDER
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Spark session if enabled
if USE_SPARK:
    spark = SparkSession.builder.appName("FinancialDataProcessing").getOrCreate()

# Initialize S3 client if enabled
if USE_S3:
    s3_client = boto3.client("s3")

class FinancialDataProcessing:
    """
    Handles JSON data validation, cleaning, and structuring.
    """
    def __init__(self):
        self.raw_text_folder = RAW_TEXT_FOLDER
        self.processed_data_folder = PROCESSED_DATA_FOLDER

    def validate_json_structure(self, data: Dict) -> bool:
        """
        Validates if the extracted JSON has the expected structure.
        Ensures that all required financial statement sections exist.
        """
        required_sections = [
            "Statement of Comprehensive Income",
            "Statement of Financial Position",
            "Statement of Changes in Equity",
            "Statement of Cash Flows"
        ]
        return all(section in data for section in required_sections)

    def normalize_data(self, data: Dict) -> Dict:
        """
        Normalizes missing values and ensures numerical consistency.
        Converts null values to zero to avoid calculation errors.
        """
        def clean_entry(entry):
            if isinstance(entry, dict):
                return {k: clean_entry(v) for k, v in entry.items()}
            elif entry is None:
                return 0  # Replace null values with zero for financial calculations
            return entry

        return {k: clean_entry(v) for k, v in data.items()}

    def process_json_file(self, file_path: str) -> Dict:
        """
        Loads, validates, and cleans a JSON file extracted from a financial statement.
        Returns a structured and normalized JSON object.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not self.validate_json_structure(data):
            logging.error(f"Invalid JSON structure: {file_path}")
            return {}

        return self.normalize_data(data)

    def process_all_json_files(self):
        """
        Processes all extracted JSON files in the raw text folder.
        Cleans and structures each file before saving it to the processed data folder.
        """
        os.makedirs(self.processed_data_folder, exist_ok=True)

        for file_name in os.listdir(self.raw_text_folder):
            if file_name.endswith(".json"):
                file_path = os.path.join(self.raw_text_folder, file_name)
                cleaned_data = self.process_json_file(file_path)
                output_file = os.path.join(self.processed_data_folder, file_name)

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(cleaned_data, f, indent=4)

                logging.info(f"Processed JSON saved: {output_file}")

                if USE_SPARK:
                    self.save_to_spark(cleaned_data, file_name)
                if USE_S3:
                    self.upload_to_s3(output_file)

    def save_to_spark(self, data, file_name):
        """
        Stores cleaned financial data in a PySpark DataFrame for scalable analytics.
        """
        df = spark.createDataFrame([(file_name, json.dumps(data))], ["file_name", "content"])
        df.write.mode("overwrite").parquet(os.path.join(SPARK_OUTPUT_FOLDER, f"{file_name}.parquet"))
        logging.info(f"Data stored in Spark: {SPARK_OUTPUT_FOLDER}/{file_name}.parquet")

    def upload_to_s3(self, file_path):
        """
        Uploads processed financial data to an S3 bucket for cloud storage.
        """
        try:
            s3_client.upload_file(file_path, S3_BUCKET_NAME, os.path.basename(file_path))
            logging.info(f"Uploaded to S3: {S3_BUCKET_NAME}/{os.path.basename(file_path)}")
        except Exception as e:
            logging.error(f"S3 Upload failed: {e}")

if __name__ == "__main__":
    processor = FinancialDataProcessing()
    processor.process_all_json_files()