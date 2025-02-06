# StatGPT: GenAI Financial Statement Analysis

## Approach
The **StatGPT** system automates financial statement analysis using **Large Language Models (LLMs)**, specifically OpenAI models, to extract, preprocess, and generate insights from financial documents. The system is designed to process **financial statements in PDF format**, extract key financial metrics, and generate structured summary reports. The core methodology includes:

### 1. Data Extraction
- PDFs are converted to images for processing.
- OpenAI Vision model is used to extract structured financial data from images.
- Extracted data is formatted into structured JSON.

### 2. Data Preprocessing
- Raw extracted data is cleaned, normalized, and validated.
- Missing values are handled, ensuring numerical consistency.
- The structured data is stored for downstream analysis.

### 3. Report Generation
- The preprocessed financial data is analyzed using **GPT-4 Turbo**.
- **Retrieval-Augmented Generation (RAG)** enhances contextual understanding of financial statements.
- The system generates a structured summary including **key metrics, trends, and narrative insights**.

### 4. Evaluation
- Generated summaries are automatically evaluated for fluency, coherence, relevance, and conciseness.
- The evaluation module ensures extracted data aligns with the actual financial reports.

The system supports **cloud scalability** through **S3 integration and PySpark support** (optional), enabling large-scale financial data processing.

---

## Challenges and Solutions

### 1. Financial Data Extraction from PDFs
**Challenge:** Extracting structured financial data from PDFs is difficult due to **varied layouts, tabular formats, and multi-page content**.

**Solution:**
- Used **OpenAI Vision API** to extract text from financial PDFs.
- Implemented **image-based parsing** to enhance accuracy for non-textual financial tables.
- Ensured consistency in extracted data by preserving the original table structures.

### 2. Model Selection for Financial Report Analysis
**Challenge:** Since there is no access to a large proprietary dataset of financial statements, **fine-tuning was not feasible**.

**Solution:**
- Used **pre-trained OpenAI LLMs (GPT-4 Turbo, GPT-4o)** to leverage existing generalization capabilities.
- Designed **custom system prompts** to guide LLMs for structured financial metric extraction.
- Optimized token limits and API calls for cost-efficient analysis.

### 3. Handling Long Financial Documents Efficiently
**Challenge:** Financial statements contain **multiple sections and large amounts of data**, making it difficult for models to maintain context.

**Solution:**
- Implemented **Retrieval-Augmented Generation (RAG)** to efficiently retrieve relevant financial context.
- Used **chunking techniques** with **overlapping segments** to ensure no data loss in financial reports.
- Optimized **embedding-based retrieval** to provide accurate context for LLM-based summarization.

---

## GenAI Implementation Details

### 1. Data Extraction with GPT-4 Vision
```python
def extract_text_from_images(self, images):
    """Uses OpenAI Vision to extract text and structure from financial PDFs."""
    img_data = [{"type": "image_url", 
                 "image_url": {"url": self.encode_image_to_base64(img)}} 
                for img in images]
    
    response = self.openai_client.chat.completions.create(
        model="gpt-4-turbo",  # Vision-capable model
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_PDF},
            {"role": "user", "content": img_data}
        ],
        max_tokens=4096,
        temperature=0,  # Deterministic output
        top_p=0.1
    )
    return response.choices[0].message.content.strip()
```

### 2. Financial Report Generation with RAG
```python
def generate_financial_summary(self, key_metrics: dict, statement_name: str):
    """Generates context-aware financial summaries using RAG."""
    if self.use_rag:
        raw_text = json.dumps(key_metrics, ensure_ascii=False)
        # Split text into chunks
        splitter = CharacterTextSplitter(separator="\n", 
                                       chunk_size=3000, 
                                       chunk_overlap=500)
        chunks = splitter.split_text(raw_text)
        
        # Create embeddings for RAG
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.vector_store = Chroma.from_texts(chunks, embeddings)
        
    system_message = SystemMessage(
        content="You are a financial analysis assistant. Generate a comprehensive yet concise summary."
    )
    user_message = HumanMessage(content=f"""
    Generate financial summary for {statement_name}.
    Must Include 3 sections:
    - Key financial metrics
    - Notable trends or observations
    - A short narrative summary

    Metrics: {json.dumps(key_metrics, indent=2)}
    Context: {self.retrieve_financial_data(f"Generate insights for {statement_name}")}
    """)

    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)
    return llm.invoke([system_message, user_message]).content
```

### 3. Quality Evaluation Pipeline
```python
def evaluate_summary(self, summary_text: str, reference_text: str) -> str:
    """Evaluates generated summaries using LLM-based assessment."""
    system_message = SystemMessage(content="""
    You are an expert in financial analysis and language assessment.
    Evaluate the following generated summary based on:
    1. Fluency (0-10): Clarity, grammar, and professional tone
    2. Coherence (0-10): Logical flow and connection of ideas
    3. Relevance (0-10): Accuracy of financial information
    4. Conciseness (0-10): Comprehensive yet brief
    Provide scores and brief explanations.
    """)
    
    user_message = HumanMessage(content=f"""
    --- Generated Summary ---
    {summary_text}

    --- Reference Financial Statement ---
    {reference_text}

    Evaluate the summary based on the criteria provided above.
    """)

    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)
    return llm.invoke([system_message, user_message]).content
```

## Scalability Considerations

**Challenge:** The system must efficiently handle multiple financial statements while ensuring stability and performance in production environments.
    
**Solution:**
- **Batch Processing for PDFs:** Implemented a pipeline that processes multiple PDFs sequentially, ensuring efficient execution without overwhelming API limits.
- **Cloud Storage Integration:** The extracted and processed data can be stored in **AWS S3**, enabling easy access for distributed processing.
- **PySpark for Large-Scale Data Handling:** The system includes an optional **PySpark integration** for handling large volumes of financial documents in distributed environments.
- **Efficient API Utilization:** OpenAI API calls are optimized to minimize latency and cost while ensuring high accuracy in text extraction and summarization.
- **Logging and Monitoring:** The system implements structured logging to track processing steps, errors, and API usage, which is critical for debugging and scalability improvements.

---

## Conclusion
StatGPT successfully automates financial statement analysis using **LLM-powered extraction, RAG-based contextualization, and structured report generation**. The system is designed to scale efficiently while maintaining high accuracy in financial metric extraction and summary generation. Future improvements include **fine-tuning models on domain-specific financial data**, **enhancing evaluation metrics for financial accuracy validation**, and **expanding cloud-based deployment strategies for large-scale adoption**.