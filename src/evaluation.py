"""
Financial Summary Evaluation Module
===================================
This module evaluates generated financial summaries against their corresponding
JSON financial statements. It leverages LLM-based evaluation methods to assess
the quality and accuracy of generated summaries.

Features include:
- Mapping generated summaries to their respective JSON financial statements.
- Extracting the correct reference text from structured JSON data.
- Using an LLM-based evaluation process to score summaries on key quality metrics.
- Logging evaluation results without storing them persistently.
"""

import os
import json
import logging
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from config import (
    OPENAI_API_KEY,
    MODEL_NAME_REPORT,
    TEMPERATURE_REPORT,
    REPORT_FOLDER,
    PROCESSED_DATA_FOLDER,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class SummaryEvaluator:
    """
    Evaluates generated financial summaries against their corresponding JSON financial statements.
    """

    def __init__(self):
        self.openai_api_key = OPENAI_API_KEY
        self.model_name = MODEL_NAME_REPORT
        self.model_temperature = TEMPERATURE_REPORT
        self.report_folder = REPORT_FOLDER
        self.processed_data_folder = PROCESSED_DATA_FOLDER
        os.makedirs(self.report_folder, exist_ok=True)
        logging.info("SummaryEvaluator initialized.")

    def find_json_file_and_section(self, summary_file: str) -> tuple:
        """
        Find the corresponding JSON file and statement section for a given summary file.
        """
        base_name = os.path.basename(summary_file).replace("_summary.txt", "")
        json_file = os.path.join(self.processed_data_folder, base_name.split("_")[0] + ".json")

        if "_overall" in summary_file:
            return json_file, None  # Overall summary corresponds to the entire JSON file
        else:
            statement_name = "_".join(base_name.split("_")[1:]).replace("_", " ")
            return json_file, statement_name  # Specific statement summary corresponds to a section in the JSON

    def read_json_data(self, json_file: str, statement_name: Optional[str] = None) -> str:
        """
        Read the JSON file and return the entire content or a specific statement section.
        """
        if not os.path.exists(json_file):
            logging.error(f"JSON file not found: {json_file}")
            return ""

        with open(json_file, "r", encoding="utf-8") as file:
            json_data = json.load(file)

        if statement_name:
            return json.dumps(json_data.get(statement_name, {}), indent=2)  # Get the specific statement section
        return json.dumps(json_data, indent=2)  # Get the entire JSON file

    def evaluate_summary(self, summary_text: str, reference_text: str) -> str:
        """
        Use ChatGPT to evaluate the summary quality based on predefined criteria.
        """
        logging.info("Starting LLM-based evaluation...")

        system_message_content = (
            "As a financial analysis and language expert, your task is to assess the quality of "
            "a generated financial summary by comparing it with the reference financial statement.\n\n"
            "Evaluate the summary based on the following criteria:\n\n"
            "1. **Fluency**: Is the language professional, clear, and grammatically accurate? (Score: 0-10)\n"
            "2. **Coherence**: Does the summary present information in a logical and structured manner? (Score: 0-10)\n"
            "3. **Relevance**: Does it accurately reflect key financial data and critical insights? (Score: 0-10)\n"
            "4. **Conciseness**: Is the summary succinct while still covering essential details without redundancy? (Score: 0-10)\n\n"
            "For each criterion, provide a score (out of 10) along with a brief justification. "
            "Conclude with an overall evaluation of how effectively the summary conveys the financial information."
        )

        user_message_content = f"""
        --- Generated Summary ---
        {summary_text}

        --- Reference Financial Statement ---
        {reference_text}

        Evaluate the summary based on the criteria provided above.
        """

        system_message = SystemMessage(content=system_message_content)
        user_message = HumanMessage(content=user_message_content)

        logging.info("Sending request to LLM for evaluation...")
        llm = ChatOpenAI(openai_api_key=self.openai_api_key, model_name=self.model_name,
                         temperature=self.model_temperature)
        response = llm.invoke([system_message, user_message])
        evaluation_result = response.content.strip()

        logging.info("Evaluation completed.")
        return evaluation_result

    def process_evaluation(self):
        """
        Process all generated summaries and evaluate them against their corresponding JSON statements.
        """
        logging.info("Starting evaluation process...")

        for file_name in os.listdir(self.report_folder):
            if file_name.endswith("_summary.txt"):
                summary_file = os.path.join(self.report_folder, file_name)
                logging.info(f"Evaluating: {summary_file}")

                # Locate the corresponding JSON file and statement section
                json_file, statement_name = self.find_json_file_and_section(summary_file)

                # Read summary text
                with open(summary_file, "r", encoding="utf-8") as file:
                    summary_text = file.read().strip()

                # Read corresponding reference JSON
                reference_text = self.read_json_data(json_file, statement_name)
                if not reference_text:
                    logging.warning(f"Skipping {summary_file} due to missing reference data.")
                    continue

                # Perform evaluation
                evaluation_result = self.evaluate_summary(summary_text, reference_text)

                # Log evaluation results instead of storing them
                logging.info(f"Evaluation for {file_name}:\n{evaluation_result}\n")

        logging.info("Evaluation process completed.")


if __name__ == "__main__":
    evaluator = SummaryEvaluator()
    evaluator.process_evaluation()