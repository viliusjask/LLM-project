**Request:**
Your task is to make a *VERY* detailed project report based on the below requirements. You have to be *VERY* honest and *DO NOT SUGARGOAT* while considering it and answering questions.

Focus on a *TECHNICAL* overview and summary of the project. Provide the general structure of the model such as:
```
[New Financial Event] → [Possibly fine-tuned Event Classifier] → [RAG Retrieval] →
[LARGE model Analysis] → [Fine-tuned Formatter] → [Structured Output]
```
Then describe each stage and go into details regarding the tech stack (provide a few *MOST POPULAR* options and explain pros and cons of each), give examples for each stage and review the below requests and questions.

Do not include pure code in this response.

**Tech stack:**
- What libraries to use for each stage?
  - For evaluation, general architecture, ML pipelines, end-to-end system.
- What models to use in general as of 14/03/2025?
- What cloud providers to consider and use?
- Do research and touch on best Machine Learning and Generative AI Engineering Best Practices as of 14/03/2025.
- JSON/YAML?
- Expand on the tech stack as much as possible. Give suggestions and alternatives to suggestions. Provide pros and cons.

NEW data scraping:
- what sources to scrape for the NEW financial events (tier1: SEC fillings, earnings reports, tier2: news, tier3: alternative data sources, i.e reddit, twitter). Feel free to give additional suggestions.
- How should this scraping be done?
  - Option 1: non-LLM scraping, just deterministic.
  - Option 2: deep research with LLM or other AI models. Suggest if this is possible. If so, what are the best tools or models for that as of 14/03/2025.

**RAG knowledge base:**
- what data to include in RAG database
- where to get this data from
- what kind of embedding algorithm / model to use
- should RAG model be the same as the one reasoning (the big model) or would it be as performant and more efficient to use a smaller model for retrieval and then pass it to the large model? Never mind, I guess the retrieval is just a deterministic vector search algorithm. The only thin is that we will need to embed the new data sources to do that.

**Testing:**
- What kind of testing pipelines to implement?
  - Is CI/CDI useful here, or good for learning outcomes?
- What kind of unit testing / general testing should be done? Should this all be on github?
- What kind of evaluation metrics can we use for each stage of the project?
  - For the models themselves? For fine tuning?
- What are good suggestions and ideas for performance tuning?

**Examples:**
- Provide final example outputs. Think of 10 options but present only 3 best ones. Two examples would be as such:
"""
Event: Federal Reserve Raises Interest Rates by 50 basis points (2023-11-12)

Analysis:

Mechanism: Rate hikes increase borrowing costs across the economy, reducing consumer spending and business investment. Higher yields on fixed-income make equities relatively less attractive, particularly for growth stocks with distant earnings.

Historical Precedents:
- 2018 Powell hiking cycle → S&P declined 20%, Powell pivoted
- 2004-2006 Gradual hikes → Markets gained 15% as economy absorbed changes
- 1994-1995 Rate shock → Brief correction then strong rally

Confidence: Medium (markets partially priced in hike, but duration of high rates remains uncertain)

Counterarguments: Inflation may decline faster than expected allowing quicker pivot to cuts; strong labor market provides consumer spending buffer; companies have reduced leverage since 2020.
"""

"""
"Mechanism: Tariffs raise input costs, squeezing margins for import-reliant sectors (autos, retail).

Historical:

2018 Trump tariffs → S&P dropped 6% in 2 weeks.

2002 Bush steel tariffs → industrials fell 15%.
Confidence: Medium (tariffs often overblown, but macro is fragile).
Counterarguments: Fed might cut rates to offset, or China delays retaliation.”*
"""
- Provide detailed final workflow examples. Think of 10 options but present only 3 best ones.
- Provide multiple example prompts for each stage / model used.

**Fundamental questions:**
- should RAG knowledge base be different from the NEW financial events sources? (tier1: SEC fillings, earnings reports, tier2: news, tier3: alternative data sources, i.e reddit, twitter)
  - my understanding is that we get the new events but its not a part of RAG. Instead we scrape the RAG knowledge base for similarities and based on the historical RAG events, make analysis on the current day events
- Provide general tips and tricks that might be smart to implement
- Consider anything that was not mentioned here and suggest any additional ideas.

**Timeline:**
- Make a realistic and honest timeline for this project if I can work 3-4 hours per day on this (20-30 hours per week).
- Rough estimated cost?