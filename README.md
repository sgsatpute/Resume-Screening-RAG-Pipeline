# Resume Screening RAG Pipeline

A Retrieval-Augmented Generation (RAG) resume screening assistant for helping hiring teams search, compare, and analyze candidate resumes against job descriptions.

The current runnable demo is a Streamlit application that uses:

- FAISS for vector search over resume chunks
- Hugging Face sentence-transformer embeddings
- LangChain for document loading, splitting, retrieval, and LLM integration
- Ollama with Llama3 for local chat and RAG Fusion query generation
- Pandas for resume CSV loading and uploaded dataset handling

The repository also contains research notebooks, generated test sets, evaluation outputs, PDF resume conversion utilities, and prebuilt vector indexes.

## Demo Preview

The Streamlit app opens directly into a chat-style resume screening assistant.

![Resume Screening GPT starting screen](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline/assets/46376260/3a7122d5-1c8e-4d98-bb06-cbc28813a2c3)

When the user provides a job description, the system retrieves matching candidates and produces a ranked hiring summary.

![Job description response example](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline/assets/46376260/d3e47a4e-257c-47d6-a12e-73e48dacc137)

When the user asks about a specific applicant ID, the system performs an exact lookup and analyzes that candidate.

![Applicant ID response example](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline/assets/46376260/94081148-b99f-40d9-b665-b5cbb7e15123)

## What The App Does

The assistant supports three main query paths:

1. Job description search
   - Detects hiring-style prompts such as "Find Python developers with ML experience".
   - Retrieves relevant resume chunks from FAISS.
   - Maps chunks back to full resumes.
   - Computes a keyword, bigram, and skill-density match score.
   - Returns ranked candidates and asks the LLM to summarize the top choices.

2. Applicant ID lookup
   - Detects numeric applicant IDs with at least three digits.
   - Fetches exact matching candidate resumes from the active dataset.
   - Produces a structured candidate analysis.

3. General recruitment questions
   - Skips retrieval when no job description, skill query, or applicant ID is detected.
   - Answers using the recent chat history and the local LLM.

## Current Demo Features

- Streamlit chat interface
- Local Llama3 model through Ollama
- Generic RAG and RAG Fusion modes
- RAG Fusion sub-query generation
- Exact applicant ID retrieval
- Resume upload through CSV files
- Automatic FAISS vectorstore rebuild if `vectorstore` is missing
- Conversation clearing
- Retrieval verbosity panel showing:
  - query type
  - selected RAG mode
  - retrieved resumes
  - generated sub-queries
  - reciprocal-rank-fusion scores
  - elapsed time

## Architecture

The application is built around a retrieval layer and a generation layer. The retrieval layer decides whether the user query needs resume context, finds relevant candidates, and prepares candidate data. The generation layer uses the retrieved resume text plus the user's request to produce a recruiter-friendly answer.

![Chatbot structure](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline/assets/46376260/dc97c06c-ca5d-4882-8e78-9101d528ee75)

At a lower level, resumes are embedded into a FAISS vectorstore. For job descriptions, the query is embedded, candidate chunks are retrieved, and RAG Fusion can expand the original query into multiple focused sub-queries before re-ranking.

![RAG pipeline](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline/assets/46376260/4259837e-9e2c-40f8-8276-e9469667b98b)

## How It Works End To End

1. The user submits a message in the Streamlit chat UI.
2. `SelfQueryRetriever.retrieve_docs()` classifies the message.
3. If the message contains applicant IDs, the retriever directly searches the active dataframe by `ID`.
4. If the message looks like a job description or skill search, the retriever uses FAISS similarity search.
5. If RAG Fusion is selected, `ChatBot.generate_subquestions()` asks Llama3 to split the job description into 3-4 focused search queries.
6. The vectorstore retrieves candidate chunks for each query.
7. Candidate chunk IDs are combined and re-ranked with reciprocal rank fusion.
8. The selected IDs are mapped back to full resume text from the dataframe.
9. Each full resume receives a readable match score based on keyword overlap, bigram overlap, and skill density.
10. The ranked resumes are sent to Llama3 as context.
11. Llama3 streams a structured answer back to Streamlit.
12. The verbosity panel shows what was retrieved, what sub-queries were used, and how long the run took.

This flow gives the LLM concrete resume context instead of asking it to answer from memory.

## Retrieval Logic In Detail

### Query classification

`detect_query_type()` checks the user message in this order:

- A number with at least three digits means applicant ID lookup.
- Hiring keywords such as `find`, `hire`, `candidate`, `developer`, `engineer`, `manager`, or `years of experience` mean job description retrieval.
- Known skill terms such as `python`, `react`, `sql`, `tensorflow`, `aws`, `docker`, or `scrum` also trigger job description retrieval.
- Anything else is treated as a general recruitment question.

### Generic RAG

Generic RAG uses the original user query only:

```text
User job description -> FAISS search -> candidate IDs -> full resumes -> LLM response
```

This is faster and simpler.

### RAG Fusion

RAG Fusion expands the original query:

```text
User job description
-> Llama3 generated sub-queries
-> FAISS search for each query
-> reciprocal rank fusion
-> candidate IDs
-> full resumes
-> LLM response
```

This can improve retrieval when the job description is long, broad, or contains several different requirements.

### Match scoring

The displayed match score is not the raw FAISS score. It is a presentation score computed from:

- unigram keyword overlap
- bigram phrase overlap, weighted higher than single words
- a small skill-density bonus

This makes the ranked output easier for users to read while FAISS still handles semantic retrieval.

## Generation Logic In Detail

The app uses different prompts depending on the query type:

- Job description retrieval: return the top candidates, preserve ranking order, choose the best candidate, and explain why.
- Applicant ID lookup: summarize strengths, weaknesses, and an overall recommendation.
- General question: answer as a recruitment assistant using recent chat history.

Responses are streamed through Streamlit with `st.write_stream()`, so the answer appears progressively.

## Project Structure

```text
Resume-Screening-RAG-Pipeline-main/
|-- README.md                         # Root README, current project guide
`-- Resume-Screening-RAG-Pipeline-main/
    |-- README.md                     # Original project README
    |-- requirements.txt
    |-- demo/
    |   |-- interface.py              # Streamlit app entry point
    |   |-- llm_agent.py              # Ollama/Llama3 prompting and streaming
    |   |-- retriever.py              # Query detection, scoring, RAG retrieval
    |   |-- ingest_data.py            # FAISS vectorstore builder
    |   |-- chatbot_verbosity.py      # Retrieval/debug UI panel
    |   `-- interactive/
    |       |-- convert_pdf.py        # Convert PDF resumes to CSV
    |       `-- ingest_data.py        # Environment-driven FAISS ingestion script
    |-- preprocessing/
    |   |-- data_cleaning.ipynb       # Job description and resume preprocessing
    |   `-- data_ingestion.ipynb      # Notebook-based vectorstore creation
    |-- evaluation/
    |   |-- testset_generation.ipynb  # Synthetic test set generation
    |   |-- results_generation.ipynb  # RAG result generation
    |   |-- ragas_evaluation.ipynb    # RAGAS metric computation
    |   |-- metrics_computation.ipynb # Accuracy and metric plots
    |   `-- images/                   # Saved evaluation plots
    |-- data/
    |   |-- main-data/                # Synthetic resumes, test sets, results
    |   `-- supplementary-data/       # Source job descriptions and PDF resumes
    |-- vectorstore/                  # Default FAISS index for the demo
    |-- vectorstore-pdf/              # FAISS index for PDF-derived resumes
    `-- vectorstore-synthetic/        # FAISS index for synthetic resumes
```

## Core Code Flow

### Streamlit app

`demo/interface.py` is the main entry point.

It loads:

- `data/main-data/synthetic-resumes.csv`
- `vectorstore`
- `sentence-transformers/all-MiniLM-L6-v2`
- `SelfQueryRetriever`
- `ChatBot`

If the FAISS index cannot be loaded, the app rebuilds it from the active resume dataframe and saves it to `vectorstore`.

### Retriever

`demo/retriever.py` handles:

- skill and hiring keyword detection
- applicant ID detection
- FAISS similarity search
- reciprocal rank fusion
- full resume reconstruction from retrieved IDs
- match score calculation
- skill extraction

RAG Fusion uses the original query plus generated sub-queries, retrieves candidates for each, and re-ranks them with reciprocal rank fusion.

### LLM Agent

`demo/llm_agent.py` wraps Ollama:

```python
Ollama(model="llama3")
```

It provides:

- sub-query generation for RAG Fusion
- structured candidate recommendation prompts
- structured applicant ID analysis prompts
- general recruitment assistant prompts using recent chat history

### Vectorstore Ingestion

`demo/ingest_data.py` builds FAISS indexes from a dataframe column.

Current chunking:

- chunk size: `1000`
- chunk overlap: `150`
- distance strategy: cosine

## Data Files

Important CSV schemas:

- `data/main-data/synthetic-resumes.csv`
  - columns: `ID`, `Resume`
- uploaded resume CSVs
  - required columns: `ID`, `Resume`
- `data/main-data/test-sets/testset-*.csv`
  - columns: `Job Description`, `Ground Truth`
- generated/evaluation result CSVs
  - usually include `question`, `ground_truth`, `answer`, `contexts`
- supplementary job descriptions
  - columns: `Job Title`, `Job Description`

The repository includes many large data and result files. They are used as datasets and experiment outputs, not as application source code.

## Requirements

- Python 3.10 or 3.11 recommended
- Ollama installed locally
- Llama3 pulled in Ollama
- Enough disk space for embeddings, FAISS indexes, and the included datasets

Install Python packages from inside the nested project directory:

```powershell
cd D:\Saurav\project\Resume-Screening-RAG-Pipeline-main\Resume-Screening-RAG-Pipeline-main
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The current app imports `langchain_huggingface`. If your environment does not already have it, install it too:

```powershell
pip install langchain-huggingface
```

## Ollama Setup

Install Ollama, then pull Llama3:

```powershell
ollama pull llama3
```

Make sure the Ollama service is running before starting the Streamlit app:

```powershell
ollama serve
```

If Ollama is already running as a desktop/background service, you do not need to run `ollama serve` again.

## Run The Demo

From the nested project directory:

```powershell
cd D:\Saurav\project\Resume-Screening-RAG-Pipeline-main\Resume-Screening-RAG-Pipeline-main
streamlit run demo/interface.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

## Example Queries

```text
Find Python developers with machine learning experience
Hire a senior backend engineer with API and SQL skills
Who has TensorFlow experience?
Show applicant 101
Compare the top 3 candidates
```

## Uploading Your Own Resumes

Use the sidebar upload control with a CSV containing:

```csv
ID,Resume
101,"Resume text here..."
102,"Another resume text here..."
```

After upload, the app:

1. Reads the CSV into a dataframe.
2. Validates that `ID` and `Resume` columns exist.
3. Builds a fresh FAISS vectorstore in memory.
4. Replaces the active retriever and dataframe for the current session.

Uploaded indexes are not saved to disk by the current UI path.

## Rebuilding The Default Vectorstore

The app automatically rebuilds `vectorstore` if it cannot load the existing index.

To force a rebuild manually:

1. Stop Streamlit.
2. Rename or remove the nested `vectorstore` directory.
3. Start Streamlit again.

The rebuilt vectorstore will be based on:

```text
data/main-data/synthetic-resumes.csv
```

## PDF Resume Conversion

`demo/interactive/convert_pdf.py` converts PDF files from:

```text
data/supplementary-data/pdf-resumes/
```

into:

```text
data/supplementary-data/pdf-resumes.csv
```

It extracts text with `pypdf.PdfReader` and writes rows with `ID` and `Resume`.

## Research And Evaluation Notebooks

The notebooks under `preprocessing/` and `evaluation/` document the original research workflow:

- cleaning job description data
- preparing synthetic resumes
- building FAISS indexes
- generating test sets
- producing RAG/RAG Fusion answers
- evaluating with RAGAS metrics:
  - context precision
  - context recall
  - faithfulness
  - answer similarity
- computing selection accuracy and semantic similarity

Some notebook cells still reference OpenAI or Azure OpenAI style endpoints. The current Streamlit demo does not require an OpenAI API key because it uses local Ollama/Llama3.

## Evaluation Images

The repository includes saved plots from the research evaluation workflow.

| Metric | Plot |
| --- | --- |
| Context Precision | ![Context precision plot](Resume-Screening-RAG-Pipeline-main/evaluation/images/CP.png) |
| Context Recall | ![Context recall plot](Resume-Screening-RAG-Pipeline-main/evaluation/images/CR.png) |
| Faithfulness | ![Faithfulness plot](Resume-Screening-RAG-Pipeline-main/evaluation/images/FA.png) |
| Answer Similarity | ![Answer similarity plot](Resume-Screening-RAG-Pipeline-main/evaluation/images/AS.png) |

## Troubleshooting

### `ModuleNotFoundError: langchain_huggingface`

Install the missing package:

```powershell
pip install langchain-huggingface
```

### Ollama connection error

Confirm Llama3 is installed:

```powershell
ollama list
```

If `llama3` is missing:

```powershell
ollama pull llama3
```

Then make sure Ollama is running.

### FAISS load error

The app loads FAISS with `allow_dangerous_deserialization=True` because LangChain FAISS indexes include pickle metadata. Only load indexes you trust.

If loading fails, let the app rebuild the index from the resume CSV.

### CSV upload error

Uploaded files must include:

```text
ID, Resume
```

The column names are case-sensitive in the current code.

### Slow first run

The first run may download the embedding model and build or load FAISS indexes. Later runs should be faster because the model and vectorstore are cached locally.

## Notes For Development

- The app stores `ChatBot`, embeddings, dataframe, retriever, and chat history in `st.session_state`.
- `ChatBot` is cached in session state so it is not recreated on every Streamlit rerun.
- The retriever uses small-to-big retrieval: chunk-level similarity search returns IDs, then full resumes are sent to the LLM.
- The scoring helper is keyword-based and separate from FAISS similarity; it is used to present readable match percentages.
- The `.env` file should not contain committed real secrets. It is not needed for the current local Ollama demo.

## License

This project includes an Apache License 2.0 file in the nested project directory.

## Acknowledgements

The original project was created as a resume screening RAG proof of concept and was inspired by RAG Fusion.
