"""
Financial Report Generation Module
==================================
This module generates structured financial reports from processed financial data.
It extracts key financial metrics, summarizes statements, and optionally utilizes
a retrieval-augmented generation (RAG) approach for enhanced contextual insights.

Features include:
- Extracting key financial metrics from structured JSON data.
- Generating detailed summaries for individual financial statements.
- Producing an overall financial summary across all statements.
- Supporting retrieval-augmented generation (RAG) for context-aware summaries.
- Storing financial reports in structured text and CSV formats.

This module ensures that financial data is presented in a meaningful, structured, and
insightful manner for analysis and reporting.
"""

import os
import json
import csv
import warnings
import logging
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from typing import Any, List, Optional, Union
from config import (
    OPENAI_API_KEY,
    MODEL_NAME_REPORT,
    TEMPERATURE_REPORT,
    REPORT_FOLDER,
    PROCESSED_DATA_FOLDER,
    USE_RAG,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Reports:
    """
    Handles financial report generation for multiple statements.
    """

    def __init__(self):
        self.openai_api_key = OPENAI_API_KEY
        self.model_name = MODEL_NAME_REPORT
        self.model_temperature = TEMPERATURE_REPORT
        self.report_folder = REPORT_FOLDER
        self.processed_data_folder = PROCESSED_DATA_FOLDER
        os.makedirs(self.report_folder, exist_ok=True)
        self.use_rag = USE_RAG  # Enable or disable RAG
        logging.info(f"Reports module initialized. RAG enabled: {self.use_rag}")

        # Initialize RAG components if enabled
        if self.use_rag:
            self.vector_store = None

    def extract_data_from_json(self, json_file_path: str) -> dict:
        """Reads and parses a JSON financial statement."""
        logging.info(f"Reading JSON file: {json_file_path}")
        with open(json_file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def extract_financial_metrics(self, json_data: dict, csv_file: Optional[str] = None) -> dict:
        """
        Extracts key financial metrics and saves them to a CSV file.
        """
        logging.info("Extracting financial metrics from the JSON data.")

        metrics_schemas = [
            ResponseSchema(name="Revenue Last Year", description="Total revenue recognized in Last Year"),
            ResponseSchema(name="Revenue Previous Year", description="Total revenue recognized in Previous Year"),
            ResponseSchema(name="Net Income Last Year", description="Net income after taxes in Last Year"),
            ResponseSchema(name="Net Income Previous Year", description="Net income after taxes in Previous Year"),
            ResponseSchema(name="Operating Expenses Last Year", description="Total operating expenses in Last Year"),
            ResponseSchema(name="Operating Expenses Previous Year",
                           description="Total operating expenses in Previous Year"),
            ResponseSchema(name="Cash Flow Last Year",
                           description="Cash and cash equivalents at the end of the reporting period in Last Year"),
            ResponseSchema(name="Cash Flow Previous Year",
                           description="Cash and cash equivalents at the end of the reporting period in Previous Year"),
        ]

        output_parser = StructuredOutputParser.from_response_schemas(metrics_schemas)
        format_instructions = output_parser.get_format_instructions()

        system_message = SystemMessage(
            content="You are a financial data extraction assistant. Extract key financial metrics in structured JSON.")
        user_message = HumanMessage(
            content=f"Extract the following financial metrics from the statement:\n{json.dumps(json_data, indent=2)}\n\n{format_instructions}")

        logging.info("Sending request to LLM for financial metric extraction...")
        llm = ChatOpenAI(openai_api_key=self.openai_api_key, model_name=self.model_name,
                         temperature=self.model_temperature)
        response_content = llm.invoke([system_message, user_message]).content

        try:
            parsed_metrics = output_parser.parse(response_content)
            logging.info("Successfully extracted financial metrics.")
        except Exception as e:
            logging.error(f"Failed to parse structured LLM output: {e}")
            return {}

        if csv_file:
            logging.info(f"Saving extracted financial metrics to CSV: {csv_file}")
            with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["Financial Metrics", "Value"])
                writer.writeheader()
                for metric, value in parsed_metrics.items():
                    writer.writerow({"Financial Metrics": metric, "Value": value})
            logging.info("Financial metrics successfully saved to CSV.")

        return parsed_metrics

    def build_vector_store(self, input_data: Union[str, dict, list]):
        """
        Builds a vector store for input data.
        """
        if not self.use_rag:
            return None  # Skip if RAG is disabled

        logging.info("Building vector store for RAG...")
        raw_text = json.dumps(input_data, ensure_ascii=False) if isinstance(input_data, (dict, list)) else str(
            input_data)

        # Split text into chunks
        splitter = CharacterTextSplitter(separator="\n", chunk_size=3000, chunk_overlap=500)
        chunks = splitter.split_text(raw_text)
        logging.info(f"Created {len(chunks)} chunks from input data.")

        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.vector_store = Chroma.from_texts(chunks, embeddings)
        logging.info("Vector store built successfully.")

    def process_all_reports(self):
        """
        Processes all financial statements and generates reports.
        """
        logging.info("Starting to process all financial reports...")
        all_metrics = {}

        for file_name in os.listdir(self.processed_data_folder):
            if file_name.endswith(".json"):
                file_path = os.path.join(self.processed_data_folder, file_name)
                logging.info(f"Processing file: {file_path}")

                json_data = self.extract_data_from_json(file_path)

                # Build vector store if RAG is enabled
                if self.use_rag:
                    logging.info(f"Building vector store for {file_name}...")
                    self.build_vector_store(json_data)

                for statement_name, statement_data in json_data.items():
                    # Modify summary naming convention to ensure evaluation finds the correct JSON and statement
                    summary_file = os.path.join(
                        self.report_folder,
                        f"{file_name.replace('.json', '')}_{statement_name.replace(' ', '_')}_summary.txt"
                    )

                    logging.info(f"Generating summary for financial statement: {statement_name}")
                    self.generate_financial_summary(statement_data, statement_name, output_file=summary_file)

                # Process overall financial report
                if "overall_metrics" not in all_metrics:
                    csv_file = os.path.join(self.report_folder,
                                            f"{file_name.replace('.json', '')}_financial_metrics.csv")
                    all_metrics = self.extract_financial_metrics(json_data, csv_file=csv_file)

        # Generate overall summary
        overall_summary_file = os.path.join(self.report_folder, f"{file_name.replace('.json', '')}_overall_summary.txt")
        logging.info("Generating overall financial summary.")
        self.generate_financial_summary(all_metrics, "Overall Financial Report", output_file=overall_summary_file)
        logging.info("Finished processing all financial reports.")

    def generate_financial_summary(self, key_metrics: dict, statement_name: str,
                                   output_file: Optional[str] = None) -> str:
        """
        Generates a summary for a specific financial statement.
        """
        logging.info(f"Generating summary for: {statement_name}")

        financial_context = ""
        if self.use_rag:
            financial_context = self.retrieve_financial_data(f"Generate insights for {statement_name}")

        system_message = SystemMessage(
            content="You are a financial analysis assistant. Summarize the provided key metrics concisely.")
        user_message = HumanMessage(content=f"""
        Generate a financial summary for {statement_name}.
        Key financial metrics:
        {json.dumps(key_metrics, indent=2)}

        Additional context (if any):
        {financial_context}

        Must Include 3 sections:
        - Key financial metrics
        - Notable trends or observations
        - A short narrative summary
        """)

        logging.info("Sending request to LLM for financial summary generation...")
        llm = ChatOpenAI(openai_api_key=self.openai_api_key, model_name=self.model_name,
                         temperature=self.model_temperature)
        response = llm.invoke([system_message, user_message])
        summary_text = response.content.strip()

        if output_file:
            logging.info(f"Saving financial summary to file: {output_file}")
            with open(output_file, mode="w", encoding="utf-8") as f:
                f.write(summary_text)

        return summary_text


if __name__ == "__main__":
    report_generator = Reports()
    report_generator.process_all_reports()