import logging
from src.data_extraction import FinancialDataIngestion
from src.data_preprocessing import FinancialDataProcessing
from src.generation import Reports
from src.evaluation import SummaryEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    # Step 1: Extract data from PDFs
    logging.info("Step 1: Extracting data from PDFs...")
    ingestion = FinancialDataIngestion()
    ingestion.process_all_pdfs()

    # Step 2: Preprocess extracted data
    logging.info("Step 2: Preprocessing extracted data...")
    processor = FinancialDataProcessing()
    processor.process_all_json_files()

    # Step 3: Generate financial reports
    logging.info("Step 3: Generating financial reports...")
    report_generator = Reports()
    report_generator.process_all_reports()

    # Step 4: Evaluate generated summaries
    logging.info("Step 4: Evaluating generated summaries...")
    evaluator = SummaryEvaluator()
    evaluator.process_evaluation()

if __name__ == "__main__":
    main()