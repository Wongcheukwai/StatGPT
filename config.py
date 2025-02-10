"""
config.py
=========
Holds configuration variables and constants for the application.

This script centralizes configuration for:
1. General settings (folder paths, API keys).
2. Data extraction settings (models, tokens, prompts).
3. Storage and scalability settings (S3, Spark).
4. Evaluation settings.

Environment variables are used for flexibility, with defaults provided for local development.
"""

import os

# ------------------------------------------------------------------------------
# General Settings
# ------------------------------------------------------------------------------
# Folder containing PDF files to process.
PDF_FOLDER = os.environ.get("PDF_FOLDER", "./financial_statements")

# Folder for storing initially extracted text from PDFs.
RAW_TEXT_FOLDER = os.environ.get("RAW_TEXT_FOLDER", "./extracted_data")

# Folder for saving processed structured JSON data.
PROCESSED_DATA_FOLDER = os.environ.get("PROCESSED_DATA_FOLDER", "./processed_data")

REPORT_FOLDER = os.environ.get("REPORT_FOLDER", "./generated_data")
# API key for accessing OpenAI services.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "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")# ------------------------------------------------------------------------------
# Data Extraction Settings
# ------------------------------------------------------------------------------
# Model for extracting text from PDF images.
MODEL_NAME_PDF = os.environ.get("MODEL_NAME_PDF", "gpt-4-turbo")

# Maximum number of tokens for a single API call.
MODEL_TOKEN_PDF = int(os.environ.get("MODEL_TOKEN_PDF", 4096))

# System prompt for the model during data extraction.
SYSTEM_PROMPT_PDF = """
You are a highly precise data extraction assistant responsible for parsing images of PDF-based financial statements. 
Your objective is to generate a well-structured JSON output containing all relevant financial details while adhering to the following guidelines:

Ensure that:
    1. **Preserve Key Statements**: Pay special attention to these financial statements:
       • Statement of Comprehensive Income
       • Statement of Financial Position
       • Statement of Changes in Equity
       • Statement of Cash Flows
    2. **Exclude Irrelevant Content**: Remove headers, footers, page numbers, and any generic text that does not contain meaningful financial data.
    3. **Merge Multi-Page Tables or Paragraphs**: If a table or paragraph spans multiple pages, integrate it seamlessly into a single section in the JSON output.
    4. **Maintain Data Structure**:
       • Keep the original table formats and field/column names.
       • Convert extracted numerical values from text to proper numeric formats.
       • If a “Notes” column has no content, include it but assign a null (or None) value.
    5. **Ensure Completeness**: Capture all lines and data points from the source documents accurately, ensuring nothing is omitted.
    6. **Output Format**:
       • Return the final output as a well-formed JSON structure.
       • Only include fields present in the original statements; do not add extra commentary or unrelated keys.

Process each document thoroughly, page by page and line by line, ensuring that no critical financial data is overlooked.
"""

# ------------------------------------------------------------------------------
# Storage and Scalability Settings
# ------------------------------------------------------------------------------
# AWS S3 Storage
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "your-s3-bucket-name")
USE_S3 = os.environ.get("USE_S3", "False").lower() == "true"

# PySpark Configuration
USE_SPARK = os.environ.get("USE_SPARK", "False").lower() == "true"
SPARK_OUTPUT_FOLDER = os.environ.get("SPARK_OUTPUT_FOLDER", "./spark_data")

# ------------------------------------------------------------------------------
# Report Generation Settings
# ------------------------------------------------------------------------------
# Model for generating financial reports.
MODEL_NAME_REPORT = os.environ.get("MODEL_NAME_REPORT", "gpt-4o")

# Temperature setting for controlling creativity & randomness in LLM outputs.
TEMPERATURE_REPORT = float(os.environ.get("TEMPERATURE_REPORT", 0))

# Chunk size for splitting text during embedding creation.
CHUNK_SIZE_REPORT = int(os.environ.get("CHUNK_SIZE_REPORT", 3000))

# Overlap size between consecutive chunks to maintain context.
CHUNK_OVERLAP_REPORT = int(os.environ.get("CHUNK_OVERLAP_REPORT", 500))

# Enable or disable RAG for financial report generation
USE_RAG = os.environ.get("USE_RAG", "True").lower() == "true"

# ------------------------------------------------------------------------------
# Evaluation Settings
# ------------------------------------------------------------------------------
# Model for evaluating extracted data quality.
MODEL_NAME_EVAL = os.environ.get("MODEL_NAME_EVAL", "gpt-4-turbo")

# Temperature setting for controlling LLM outputs.
TEMPERATURE_EVAL = float(os.environ.get("TEMPERATURE_EVAL", 0))