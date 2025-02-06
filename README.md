# 🤖 StatGPT

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com/blog/openai-api)
[![LangChain](https://img.shields.io/badge/🦜-LangChain-blue)](https://github.com/hwchase17/langchain)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An automated end-to-end solution for analyzing financial statements using Large Language Models. This project leverages OpenAI's advanced models and LangChain framework to extract, process, and generate insights from financial documents with high accuracy and reliability.

## 🌟 Key Features

- 📊 **Intelligent Data Extraction**: Advanced PDF parsing with OpenAI Vision for accurate financial data extraction
- 🔄 **Robust Data Processing**: Sophisticated validation and normalization of financial metrics  
- 📈 **Smart Financial Analysis**: Context-aware report generation with RAG (Retrieval Augmented Generation)
- 📋 **Comprehensive Evaluation**: Multi-metric quality assessment of generated summaries
- ☁️ **Cloud Integration**: Optional S3 storage and PySpark support for scalability

## 🏗️ System Architecture

StatGPT implements a fully automated pipeline for processing financial statements:

1. **Data Ingestion**: Automatically converts PDFs to structured data using OpenAI Vision
2. **Data Processing**: Smart validation and normalization of extracted data
3. **Report Generation**: AI-powered financial analyses and summaries
4. **Quality Evaluation**: Automatic assessment of extraction and summary quality
5. **Cloud Storage**: Seamless integration with S3 and Spark (optional)

## 📁 Code Structure
```
StatGPT/
├── financial_statements/   # Input: Place your PDF files here
├── extracted_data/        # Output: Extracted JSON data from PDFs
├── processed_data/        # Output: Cleaned financial data
├── generated_data/        # Output: Generated summaries & reports
├── src/
│   ├── data_extraction.py    # PDF to structured data conversion
│   ├── data_preprocessing.py # Data cleaning & normalization
│   ├── generation.py         # Report & summary generation
│   └── evaluation.py         # Quality assessment
├── requirements.txt         # Project dependencies
├── config.py               # Configuration settings
├── app.py                  # Main execution script
├── deliverables/           # I save the deliverable results here
└── README.md               # Documentation
```

## 🔧 Prerequisites

- Python 3.10
- OpenAI API key
- For cloud features (optional):
  - AWS credentials for S3 storage
  - Apache Spark environment for distributed processing
  
## 📂 Automated Pipeline

Simply run `app.py` and StatGPT will:

1. Read all PDFs from `financial_statements/`
2. Extract and store structured data in `extracted_data/`
3. Process and normalize data in `processed_data/`
4. Generate comprehensive reports in `generated_data/`
5. Evaluate and validate all outputs automatically

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/Wongcheukwai/StatGPT.git
cd StatGPT
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
Either set your OpenAI API key in `.env`:
```bash
echo "OPENAI_API_KEY=your_key_here" > .env
```
Or update directly in `config.py`:
```python
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your_key_here")
```

4. **Run the analysis**
Just run:
```bash
python app.py
```

## 🛠️ Configuration

Key settings in `config.py`:
```python
# Model Settings
MODEL_NAME_PDF = "gpt-4-turbo"
MODEL_NAME_REPORT = "gpt-4-turbo"

# Processing Settings
USE_RAG = True  # Enable Retrieval Augmented Generation
USE_SPARK = False  # Enable PySpark integration
USE_S3 = False  # Enable S3 storage
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions and feedback, please open an issue or reach out to [zhuowei.wang.cs.uts@gmail.com]