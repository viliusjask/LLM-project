# **Financial Intelligence System: Technical Project Report**

## **1. Technical Overview**

This project implements a sophisticated financial event analysis pipeline with a hybrid architecture that leverages both specialized models and large foundation models for different tasks. The core architecture follows:

```
[New Financial Event] → [Fine-tuned Event Classifier] → [RAG Retrieval] →
[Large Model Analysis] → [Fine-tuned Formatter] → [Structured Output]
```

This approach strategically balances efficiency, cost, and performance by using specialized components for specific tasks and leveraging powerful API-based models where their reasoning capabilities add the most value.

## **2. Detailed System Architecture**

### **2.1 New Financial Event Collection**

This stage ingests financial events from multiple tiers of sources:

- **Tier 1 (Core Financial Data)**
  - SEC EDGAR filings (10-K, 10-Q, 8-K, etc.)
  - Earnings call transcripts
  - Federal Reserve announcements
  - Economic indicators releases

- **Tier 2 (Financial News)**
  - Major financial publications (Bloomberg, Reuters, WSJ)
  - Industry-specific news (TechCrunch for tech, FierceBiotech for healthcare)
  - Regional financial news (for localized market insights)

- **Tier 3 (Alternative Data)**
  - Social media (Reddit finance communities, Twitter financial analysts)
  - Web forums (Seeking Alpha, Yahoo Finance discussions)
  - Conference transcripts

**Implementation Approaches:**

1. **Deterministic Scraping (Recommended for Tier 1)**
   - Built with BeautifulSoup, Scrapy, or Selenium
   - Structured API connections where available (SEC EDGAR API)
   - Scheduled data collection with Airflow/Luigi

2. **LLM-Enhanced Scraping (For Tier 2/3)**
   - Using models like Anthropic's Claude to extract relevant information from unstructured text
   - Event extraction models to identify market-moving events in news articles
   - Zero-shot classifiers to categorize content relevance

### **2.2 Event Classification**

A fine-tuned classifier that categorizes financial events into specific types:

**Model Options:**
- **RoBERTa-large-financial**: Fine-tuned on financial texts (recommended)
- **BERT-base**: Smaller but faster option
- **FinBERT**: Domain-specific baseline

**Event Categories:**
- Macro events: `RATE_CHANGE`, `INFLATION_DATA`, `GDP_REPORT`
- Corporate events: `EARNINGS_REPORT`, `MANAGEMENT_CHANGE`, `DIVIDEND_ANNOUNCEMENT`
- Market events: `SECTOR_ROTATION`, `VOLATILITY_SPIKE`, `IPO_LAUNCH`

**Example Event:**
```
INPUT: "Federal Reserve raises interest rates by 50 basis points, citing persistent inflation concerns."
OUTPUT:
{
  "event_type": "RATE_HIKE",
  "confidence": 0.94,
  "entities": ["Federal Reserve", "US Economy"],
  "key_metrics": {"rate_change_bps": 50, "previous_change_bps": 25}
}
```

### **2.3 RAG Retrieval**

The retrieval system pulls relevant historical context based on the classified event:

**Key Components:**
- **Vector Database**: Stores embeddings of historical events, market reactions, and analyses
- **Hybrid Search**: Combines vector similarity with keyword/BM25 for optimal retrieval
- **Reranking**: Uses a cross-encoder model to rerank retrieved documents by relevance

**Example Retrieved Context:**
```
HISTORICAL EVENT (2018-12-19): "Federal Reserve raises rates by 25 bps to 2.25-2.50%"
MARKET REACTION: "S&P 500 declined 7.7% in the following week"
ECONOMIC IMPACT: "Mortgage rates increased to 4.75%, housing starts declined 8.6% in Q1 2019"
ANALYST NOTE: "Powell's hawkish stance surprised markets expecting a more dovish forward guidance"
```

### **2.4 Large Model Analysis**

This stage leverages a powerful foundation model to generate comprehensive analysis:

**Model Options:**
- **GPT-4o/GPT-5** (OpenAI): Most powerful reasoning but most expensive
- **Claude 3 Opus/Sonnet** (Anthropic): Excellent at nuanced analysis, less hallucination
- **Gemini Ultra** (Google): Good performance with structured prompting

**Prompt Template Example:**
```
You are an expert financial analyst. Analyze this financial event with provided historical context:

EVENT: {event_description}
TYPE: {event_type}
DATE: {event_date}
ENTITIES: {event_entities}

HISTORICAL CONTEXT:
{retrieved_context}

Generate a comprehensive analysis including:
1. Economic mechanism and transmission channels
2. Relevant historical precedents with specific outcomes
3. Confidence assessment with justification
4. Potential counterarguments or alternative scenarios

Your analysis should be thorough but concise.
```

### **2.5 Fine-tuned Formatter**

A specialized model that structures the raw analysis into a consistent format:

**Model Options:**
- **Phi-3-mini** (Recommended): Lightweight, efficient for formatting tasks
- **Llama-3-8B**: More powerful but higher resource requirements
- **Mistral-7B**: Good balance of performance and efficiency

**Implementation:**
- Fine-tuned on 500-1000 examples of raw → formatted analysis pairs
- PEFT/LoRA fine-tuning for efficient adaptation
- ONNX optimization for faster inference

**Example Input/Output:**
```
INPUT (From large model): "When the Fed raises rates, it increases borrowing costs throughout the economy. Historically, the 2018 cycle saw markets decline sharply as Powell maintained a hawkish stance. However, we've seen gradual hiking cycles like 2004-2006 that markets absorbed well. Given current economic indicators, I'd assess medium confidence in market disruption, as labor markets remain strong which could offset negative effects."

OUTPUT (Formatted):
{
  "mechanism": "Rate hikes increase borrowing costs across the economy, reducing consumer spending and business investment.",
  "historical_precedents": [
    {"period": "2018", "event": "Powell hiking cycle", "outcome": "S&P declined 20%, Powell pivoted"},
    {"period": "2004-2006", "event": "Gradual hikes", "outcome": "Markets gained 15% as economy absorbed changes"}
  ],
  "confidence": {
    "level": "Medium",
    "rationale": "Markets partially priced in hike, but duration of high rates remains uncertain"
  },
  "counterarguments": "Strong labor market provides consumer spending buffer; companies have reduced leverage since 2020."
}
```

## **3. Comprehensive Tech Stack**

### **3.1 Data Processing & ETL**

**Options:**
- **Pandas** ✓
  - Pros: Easy to use, excellent for data manipulation
  - Cons: Limited scalability for very large datasets
- **PySpark**
  - Pros: Highly scalable, good for distributed processing
  - Cons: Steeper learning curve, overhead for smaller datasets
- **Dask**
  - Pros: Pandas-like API with better scaling
  - Cons: Less mature ecosystem than PySpark

**Recommendation:** Start with Pandas for simplicity, with architecture allowing migration to Dask if scale requires it.

### **3.2 Web Scraping & Data Collection**

**Options:**
- **Beautiful Soup + Requests** ✓
  - Pros: Simple, flexible, lightweight
  - Cons: No built-in parallelism, requires more manual handling
- **Scrapy**
  - Pros: Comprehensive framework, built-in concurrency
  - Cons: Steeper learning curve
- **Selenium**
  - Pros: Handles JavaScript-rendered content
  - Cons: Resource-intensive, slower

**Recommendation:** Beautiful Soup for static content, Selenium for dynamic sites that require JavaScript rendering.

### **3.3 Workflow Orchestration**

**Options:**
- **Apache Airflow** ✓
  - Pros: Robust scheduling, extensive monitoring, large ecosystem
  - Cons: Complex setup, potentially heavy for small projects
- **Prefect**
  - Pros: Modern API, easier to get started than Airflow
  - Cons: Smaller community
- **Luigi**
  - Pros: Simpler than Airflow, focus on task dependencies
  - Cons: Less feature-rich monitoring

**Recommendation:** Airflow for its maturity and robust scheduling capabilities.

### **3.4 Vector Databases**

**Options:**
- **Chroma DB** ✓
  - Pros: Easy to use, Python-native, good for development
  - Cons: Less scalable than cloud solutions
- **Pinecone**
  - Pros: Fully managed, highly scalable
  - Cons: Cost increases with scale
- **Weaviate**
  - Pros: Rich query features, multi-modal
  - Cons: More complex setup

**Recommendation:** Start with ChromaDB for development, with design allowing easy migration to Pinecone for production.

### **3.5 LLM Orchestration**

**Options:**
- **LangChain** ✓
  - Pros: Comprehensive framework, active development
  - Cons: Frequent API changes, can be complex
- **LlamaIndex**
  - Pros: Specialized for RAG, simpler for document retrieval
  - Cons: Less flexible than LangChain for general LLM applications
- **Custom integration**
  - Pros: Maximum flexibility, minimal dependencies
  - Cons: Requires more development time

**Recommendation:** LangChain for its comprehensive toolkit and active community.

### **3.6 Model Deployment & Serving**

**Options:**
- **Hugging Face Inference Endpoints** ✓
  - Pros: Easy deployment, supports many models
  - Cons: Can be expensive at scale
- **ONNX Runtime + FastAPI**
  - Pros: Optimized inference, more control
  - Cons: Requires more engineering
- **TorchServe/TensorFlow Serving**
  - Pros: Production-grade serving for specific frameworks
  - Cons: Framework-specific, more complex setup

**Recommendation:** Start with Hugging Face for simplicity, transition to ONNX + FastAPI for production optimization.

### **3.7 Cloud Providers**

**Options:**
- **AWS** ✓
  - Pros: Most comprehensive services, SageMaker for ML
  - Cons: Complex pricing, steeper learning curve
- **GCP**
  - Pros: Strong ML/AI offerings, Vertex AI
  - Cons: Fewer region options than AWS
- **Azure**
  - Pros: Good integration with Microsoft ecosystem
  - Cons: Some services less mature than AWS alternatives

**Recommendation:** AWS for its comprehensive services and established ML infrastructure.

### **3.8 Data Storage & Configuration**

**Options:**
- **Storage: S3 + MongoDB** ✓
  - Pros: S3 for raw data, MongoDB for processed/structured data
  - Cons: Managing two systems
- **Configuration: YAML** ✓
  - Pros: Human-readable, supports comments, hierarchical
  - Cons: Less standardized schema validation than JSON
- **Messaging: Apache Kafka**
  - Pros: Robust for event-driven architecture
  - Cons: Operational overhead

**Recommendation:** S3 + MongoDB for storage, YAML for configuration.

## **4. RAG Knowledge Base**

### **4.1 Data to Include**

1. **Historical Event Database**
   - Major market events (crashes, bubbles, corrections)
   - Central bank policy changes (2000-present)
   - Geopolitical events with market impact

2. **Company-Specific History**
   - Historical earnings reports (surprise/miss patterns)
   - Management changes and outcomes
   - M&A activity and market reactions

3. **Economic Data Time Series**
   - Interest rate cycles and market responses
   - Inflation/GDP/unemployment data
   - Sector performance during different economic regimes

4. **Analyst Reports**
   - Public summaries from major banks
   - Consensus estimates and historical accuracy
   - Sector outlook reports

### **4.2 Data Sources**

- **Financial Data Providers**
  - Bloomberg API (paid)
  - FRED (Federal Reserve Economic Data - free)
  - Alpha Vantage (free tier available)

- **Academic/Research**
  - NBER Working Papers
  - SSRN Financial Papers
  - Federal Reserve Published Research

- **Historical Archives**
  - Financial Times Archive
  - Wall Street Journal Historical
  - Financial Crisis Documents (2008, 2020)

### **4.3 Embedding Models**

**Options:**
- **text-embedding-3-small** (OpenAI) ✓
  - Pros: State-of-the-art performance, good cost/performance ratio
  - Cons: API costs, reliance on external service
- **Cohere Embed**
  - Pros: Competitive performance, specialized for financial texts
  - Cons: Similar cost concerns as OpenAI
- **all-MiniLM-L6-v2** (Sentence Transformers)
  - Pros: Free, self-hostable, decent performance
  - Cons: Less powerful than commercial alternatives

**Recommendation:** text-embedding-3-small for production quality, with all-MiniLM-L6-v2 as a fallback for cost-sensitive development.

### **4.4 RAG Architecture**

The optimal RAG approach for this system is a hybrid architecture:

1. **Chunking Strategy:**
   - Document-level for overall context
   - Paragraph-level for granular retrieval
   - Hierarchical chunks with parent-child relationships

2. **Retrieval Approach:**
   - **BM25 + Semantic Search:** Combine keyword matching with embedding similarity
   - **Metadata Filtering:** Filter by event type, date range, entities before vector search
   - **Reranking:** Use a cross-encoder to rerank retrieved documents

3. **Context Processing:**
   - Apply contextual compression to focus on most relevant parts
   - Synthesize retrieved content into concise summary when multiple documents are relevant

## **5. Testing & Evaluation**

### **5.1 Testing Pipelines**

**CI/CD Implementation:**
- **GitHub Actions** ✓
  - Automated testing on push/PR
  - Model evaluation on scheduled intervals
  - Docker image building and deployment

**Recommended Testing Layers:**
1. **Unit Tests:** Test individual components (parsers, classifiers, formatters)
2. **Integration Tests:** Test interactions between components
3. **System Tests:** End-to-end pipeline testing
4. **Evaluation Tests:** Measure output quality metrics

### **5.2 Evaluation Metrics**

**Event Classification:**
- Precision/Recall/F1 Score
- Confusion matrix across event types
- ROC-AUC for confidence calibration

**RAG System:**
- RAGAS metrics (faithfulness, relevance, context recall)
- Retrieval precision@k
- Latency measurements

**Large Model Analysis:**
- Human evaluation of sample outputs
- Factual consistency with retrieved context
- Comprehensiveness assessment

**Formatter Model:**
- Format adherence rate
- Content preservation score
- Error rate on structured fields

### **5.3 Performance Tuning**

1. **Model Quantization**
   - ONNX runtime with int8 quantization for formatter model
   - KV cache optimization for inference efficiency

2. **Batched Processing**
   - Group similar events for batch embedding
   - Pipeline parallel processing where possible

3. **Caching Strategies**
   - LRU cache for common retrievals
   - Persistent caching of embeddings
   - Pre-computed embeddings for static knowledge base

4. **Monitoring**
   - Prometheus + Grafana dashboards
   - Model performance drift detection
   - Cost tracking per component

## **6. Example Outputs**

### **6.1 Final Analysis Examples**

**Example 1: Central Bank Policy Change**

```
Event: Federal Reserve Raises Interest Rates by 50 basis points (2025-02-15)

Analysis:

Mechanism: Rate hikes increase borrowing costs across the economy, reducing consumer spending and business investment. Higher yields on fixed-income make equities relatively less attractive, particularly for growth stocks with distant earnings.

Historical Precedents:
- 2018 Powell hiking cycle → S&P declined 20%, Powell pivoted
- 2004-2006 Gradual hikes → Markets gained 15% as economy absorbed changes
- 1994-1995 Rate shock → Brief correction then strong rally

Confidence: Medium (markets partially priced in hike, but duration of high rates remains uncertain)

Counterarguments: Inflation may decline faster than expected allowing quicker pivot to cuts; strong labor market provides consumer spending buffer; companies have reduced leverage since 2020.
```

**Example 2: Trade Policy Impact**

```
Event: United States Announces 25% Tariffs on Chinese Technology Imports (2025-04-10)

Analysis:

Mechanism: Tariffs raise input costs, squeezing margins for import-reliant sectors (technology hardware, consumer electronics). Supply chains face disruption as companies consider relocating manufacturing, increasing short-term uncertainty and capital expenditure.

Historical Precedents:
- 2018-2019 Trump tariffs → Tech hardware index dropped 15% over 3 months
- 2002 Bush steel tariffs → S&P industrials declined 12%, policy reversed after WTO ruling
- 1930 Smoot-Hawley → Prolonged economic contraction, global trade collapsed 66%

Confidence: High (direct impact on technology supply chains unavoidable, though magnitude uncertain)

Counterarguments: Companies may absorb costs rather than raise prices; alternative suppliers in Vietnam/India could offset disruption; China might negotiate rather than escalate with counter-tariffs.
```

**Example 3: Corporate Earnings Surprise**

```
Event: NVIDIA Reports 85% Revenue Growth, Beating Estimates by 27% (2025-05-22)

Analysis:

Mechanism: Significant earnings beats in major tech companies signal stronger-than-expected AI adoption and enterprise spending. NVIDIA's results specifically indicate accelerating data center buildouts and sustained AI chip demand.

Historical Precedents:
- 2023 NVIDIA AI boom → Stock rose 45% in following month, lifted entire semiconductor sector
- 2020 COVID tech acceleration → Nasdaq gained 30% in 6 months as digitalization trends accelerated
- 2017 Crypto mining boom → Short-lived rally followed by 50% sector correction when demand normalized

Confidence: Medium-High (clear evidence of strong demand, but valuations already elevated)

Counterarguments: AI spending could face rationalization as ROI scrutiny increases; competition from AMD/Intel/custom chips may pressure margins; regulatory concerns around AI could create headwinds.
```

### **6.2 Workflow Examples**

**Workflow 1: New SEC Filing Analysis**

1. **Event Detection:**
   - System detects new 8-K filing from Tesla reporting unexpected production delays
   - Event classifier identifies as: `PRODUCTION_DISRUPTION`

2. **RAG Retrieval:**
   - Retrieves historical: Tesla's previous production issues (2018, 2021)
   - Retrieves: Competitor production disruption impacts
   - Retrieves: Supply chain analysis reports

3. **Large Model Analysis:**
   - GPT-4o processes filing content + historical context
   - Generates assessment of impact magnitude, timeline, competitive implications

4. **Formatting:**
   - Phi-3 structures analysis into standardized format
   - Adds confidence metrics based on historical precedent similarity

5. **Final Output:**
   - Structured analysis with clear mechanism, precedents, and counterarguments
   - Delivered via API and dashboard alert

**Workflow 2: Breaking Financial News Analysis**

1. **Event Detection:**
   - System monitors financial news APIs, detects major story about unexpected merger
   - Event classified as: `MERGER_ANNOUNCEMENT`

2. **Entity Recognition:**
   - Identifies companies involved, sector, deal size
   - Extracts key terms from announcement

3. **RAG Retrieval:**
   - Retrieves: Similar mergers in same industry
   - Retrieves: Regulatory precedents for similar deals
   - Retrieves: Market reactions to comparable announcements

4. **Large Model Analysis:**
   - Claude Opus analyzes announcement details + historical context
   - Assesses regulatory risks, timeline, market implications

5. **Final Output:**
   - Comprehensive analysis with clear sections on mechanism, historical precedents
   - Probability assessment of deal completion
   - Structured for consumption by financial professionals

**Workflow 3: Macroeconomic Data Release Analysis**

1. **Event Detection:**
   - System identifies new CPI data release showing unexpected inflation increase
   - Event classified as: `INFLATION_SURPRISE`

2. **Contextual Retrieval:**
   - Retrieves: Previous inflation surprises and market reactions
   - Retrieves: Fed policy response patterns to inflation
   - Retrieves: Sector performance during inflation spikes

3. **Analysis Generation:**
   - GPT-5 processes inflation data, context, and recent Fed communications
   - Generates nuanced analysis of potential policy responses

4. **Formatter Processing:**
   - Structures analysis into standardized format
   - Ensures all required elements (mechanism, precedents, etc.) are included

5. **Delivery & Archiving:**
   - Analysis pushed to dashboard and notification system
   - Event and analysis archived in knowledge base for future retrieval

### **6.3 Example Prompts**

**Event Classification Prompt:**
```
Classify the following financial event into the most appropriate category:

Text: "{event_text}"

Available categories:
- RATE_CHANGE: Central bank interest rate decisions
- EARNINGS_REPORT: Corporate earnings announcements
- MERGER_ACQUISITION: Company mergers or acquisitions
- REGULATORY_ACTION: Government regulatory decisions
- MARKET_VOLATILITY: Significant market movements
- ECONOMIC_DATA: Major economic indicator releases
- [Additional categories...]

Return the category and confidence score.
```

**Large Model Analysis Prompt:**
```
You are an expert financial analyst with deep market knowledge. Analyze the following financial event using the retrieved historical context.

EVENT: {event_description}
DATE: {event_date}
TYPE: {event_type}
ENTITIES: {entities}

RETRIEVED CONTEXT:
{context_1}
{context_2}
{context_3}

Your task is to provide a comprehensive analysis including:
1. Economic mechanism: Explain how this event affects markets and the economy
2. Historical precedents: Identify 2-3 similar historical events and their outcomes
3. Confidence assessment: Provide your confidence level (High/Medium/Low) with justification
4. Counterarguments: Present alternative viewpoints or mitigating factors

Your analysis should be evidence-based, nuanced, and focused on the financial implications. Avoid political bias or speculation without historical basis.
```

**Formatter Model Prompt:**
```
Transform the following analytical text into a structured financial analysis with the following components:

1. Mechanism
2. Historical Precedents (bulleted list)
3. Confidence assessment (with rationale)
4. Counterarguments

INPUT TEXT:
{raw_analysis_text}

Format the output into these clear sections while preserving all key insights and data points.
```

## **7. Fundamental Considerations**

### **7.1 RAG Knowledge Base vs. New Events**

The RAG knowledge base should be **distinct** from the new financial events sources:

- **RAG Knowledge Base**: Contains historical data, pre-processed and embedded for retrieval
  - Curated, verified, and structured information
  - Regularly updated (weekly/monthly) but not real-time
  - Focused on quality over recency

- **New Events Sources**: Real-time detection of market-moving information
  - Constant monitoring of feeds
  - Minimal processing before classification
  - Optimized for speed and coverage

This separation ensures:
1. Clean historical data for context retrieval
2. No contamination of the knowledge base with unverified breaking news
3. Clear separation of concerns in the architecture

### **7.2 Smart Implementation Tips**

1. **Tiered Processing**
   - Apply increasingly expensive models only to increasingly relevant content
   - Use lightweight filters before heavy processing

2. **Parallel Retrieval Strategies**
   - Run multiple retrieval approaches simultaneously
   - Compare/merge results for better coverage

3. **Continuous Evaluation**
   - Implement human feedback loop for edge cases
   - Track model performance over time to detect drift

4. **Cost Optimization**
   - Cache common retrievals and embeddings
   - Batch process where possible
   - Use quantized models for formatting/classification

5. **Incremental Development**
   - Start with core financial events (earnings, Fed decisions)
   - Expand to more nuanced events as system matures
   - Add sources incrementally to manage complexity

### **7.3 Additional Considerations**

1. **Responsible AI**
   - Implement bias detection for financial analysis
   - Add confidence metrics with clear limitations
   - Include disclaimers about investment advice regulations

2. **Explainability**
   - Track provenance of all insights to source documents
   - Enable "citations" in final analysis
   - Log retrieval decisions and confidence scores

3. **User Feedback Loop**
   - Allow users to rate analysis quality
   - Use feedback to improve retrieval and formatting
   - Track which historical precedents were most helpful

4. **Multilingual Support**
   - Consider international financial markets
   - Implement translation layer for non-English sources
   - Ensure consistent terminology across languages

## **8. Timeline & Cost Estimate**

### **8.1 Realistic Timeline (20-30 hours/week)**

**Phase 1: Infrastructure & Data Collection (3 weeks)**
- Week 1: Set up cloud infrastructure, design data schemas
- Week 2: Implement SEC filing & financial news scrapers
- Week 3: Build event classification system

**Phase 2: Knowledge Base & RAG (4 weeks)**
- Week 4: Collect and process historical data
- Week 5: Implement embedding pipeline and vector database
- Week 6: Build retrieval system with evaluation
- Week 7: Optimize RAG performance

**Phase 3: Analysis Generation (3 weeks)**
- Week 8: Implement large model integration with prompt engineering
- Week 9: Develop formatter model training data
- Week 10: Fine-tune and optimize formatter model

**Phase 4: Integration & Testing (2 weeks)**
- Week 11: End-to-end system integration
- Week 12: Testing, evaluation, and optimization

**Total Timeline: 12 weeks (3 months)**

### **8.2 Cost Estimate**

**Development Costs (One-time)**
- Cloud infrastructure setup: $100-200
- Historical data collection: $200-500 (if using paid APIs)
- Model fine-tuning: $100-300

**Recurring Costs (Monthly)**
- Cloud infrastructure: $50-100/month
  - EC2/Lambda compute: $30-50
  - S3/MongoDB storage: $10-20
  - Miscellaneous services: $10-30

- API Usage:
  - OpenAI (GPT-4o, embeddings): $200-500/month
  - Alternative APIs (news, financial data): $50-200/month

- Hugging Face Endpoints (optional): $100-300/month

**Total Estimated Cost:**
- Initial Development: $400-1,000
- Monthly Operation: $300-800

**Cost Optimization Options:**
- Use open-source models when possible
- Implement caching aggressively
- Batch process events rather than real-time for everything
- Use tiered processing approach

---

This project combines cutting-edge ML/NLP techniques with practical financial domain knowledge to create a sophisticated analysis system. While ambitious, the modular design allows for incremental development and clear learning outcomes at each stage. The hybrid architecture balances performance and cost considerations, making it both an impressive portfolio project and a potentially valuable tool.