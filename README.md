# DocuMind AI --- Evidence-Grounded Document Research Assistant

> Project type: Retrieval-Augmented Generation (RAG) with a separate
> evidence-verification step\
> Interface: Streamlit\
> Embeddings: Sentence Transformers (`all-MiniLM-L6-v2`)\
> LLM: Google Gemini API\
> Status: Internship prototype; not production-ready

DocuMind AI lets users upload supported PDF or TXT documents, ask
questions about their contents, retrieve relevant passages, generate an
answer grounded in those passages, and run a separate verification step
to assess whether the answer is supported by the retrieved evidence.

The aim is to make answers easier to inspect: users can review the
retrieved passages instead of treating a generated answer as
automatically correct.

## Features

-   Upload supported PDF/TXT files.
-   Extract text and split it into chunks.
-   Create local embeddings using Sentence Transformers.
-   Retrieve relevant chunks using vector similarity.
-   Ask Gemini to draft an answer using retrieved evidence.
-   Run a separate Gemini verification call.
-   Display the answer and evidence in Streamlit.
-   Show a warning if verification is unavailable.

Scanned/image-only PDFs may require OCR before their contents can be
searched.

## Workflow and architecture

``` text
Upload PDF/TXT
    -> Extract text
    -> Split text into chunks and track sources
    -> Create local embeddings
    -> Retrieve relevant passages for the question
    -> Gemini drafts an answer from retrieved evidence
    -> Gemini checks the draft against the evidence
    -> Display answer, evidence, and verification status
```

### Implementation details

-   Ingestion: extracts text from uploaded files and divides it into
    chunks.
-   Embeddings: `all-MiniLM-L6-v2` generates local vectors; document
    embeddings are normalized and represented as NumPy arrays.
-   Retrieval: ranks chunks by similarity to the question and passes
    selected passages to the model.
-   Generation: the prompt instructs Gemini to use only retrieved
    evidence and cite passages using markers such as `[1]` and `[2]`.
-   Verification: a second Gemini call compares the draft with the
    retrieved evidence and returns a verdict such as `SUPPORTED`,
    `PARTIALLY SUPPORTED`, or `UNSUPPORTED`.
-   UI: Streamlit provides file upload, a question field, workflow
    status, and answer/evidence display.

This is a lightweight RAG workflow with a separate verification stage,
not a fully autonomous production agent.

## Tech stack

  Component             Technology
  --------------------- --------------------------------------------
  Interface             Streamlit
  Language model        Google Gemini API via `google-genai`
  Embeddings            Sentence Transformers
  Vector operations     NumPy
  Document processing   PDF/TXT extraction implemented in `app.py`
  Language              Python

## Run locally

1.  Install a compatible Python version and obtain a Gemini API key with
    access to a supported model.

2.  Open the extracted project folder in VS Code.

3.  Open **Terminal → New Terminal** and ensure the terminal is in the
    folder containing `app.py` and `requirements.txt`.

4.  Install dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

5.  Start the app:

    ``` bash
    streamlit run app.py
    ```

6.  Streamlit should open the app in your browser, usually at
    `http://localhost:8501`.

7.  Enter the Gemini API key in the app's sidebar if prompted.


## How to test

Use a document whose contents you can verify. The questions below are
examples; adapt them to the uploaded document.

1.  Summary: "What is the main goal of this document?"
2.  Evidence: "List the key points and cite the passages that
    support them."
3.  Specific detail: "What methods, tools, or components are
    mentioned in the document?"
4.  Insufficient evidence: "What does this document say about a
    topic that is not mentioned in it?"


### Test log

  --------------------------------------------------------------------------------
  Test              Question                   Expected behavior Actual result
  ----------------- -------------------------- ----------------- -----------------
  1                 What is the main goal of   Concise answer    Pending --- fill
                    this document?             grounded in the   after testing
                                               document          

  2                 List key points with       Relevant evidence Pending --- fill
                    supporting passages        is shown and      after testing
                                               cited             

  3                 Which                      Details match the Pending --- fill
                    methods/tools/components   document          after testing
                    are mentioned?                               

  4                 Ask about information      App acknowledges  Pending --- fill
                    absent from the document   insufficient      after testing
                                               evidence          
  --------------------------------------------------------------------------------

## Screenshots


``` text
screenshots/
├── question_1.png
├── question_2.png
├── question_3.png
├── question_4.png
└── home.png
```


## Known issues and error log

This section documents errors observed during development and the
measures attempted. It does not claim that every issue is permanently
resolved.

### 1. `404 NOT_FOUND` --- initial Gemini model unavailable

Observed: The API reported that `models/gemini-2.5-flash` was no
longer available to new users in this setup and recommended another
model ID.

Likely cause: The configured model ID was unavailable to the API
key/project being used.

Measure taken: Changed `ask_gemini()` to use the model ID
recommended in the API error, `gemini-3.8-flash`.

Status: A later request reached a `503` error, but model
availability can vary. Confirm the exact model ID is currently supported
for the API project before demoing the app.

### 2. `503 UNAVAILABLE` --- verification call failed

Observed: The error occurred at the verification call after the
draft answer had been generated.

Cause reported by API: Temporary high demand.

Measure taken: Added a `try/except` around the verification call to
show a warning and explain that the answer was not independently
verified if the verification call returns a temporary `503`.

Limitation: This only handles failure in verification. It does not
catch errors from the earlier draft-generation call.

### 3. `NameError: name 'client' is not defined`

Observed: The verification call could not access `client`.

Cause: The `try/except` block had been placed outside the `if run:`
/ `with st.status(...)` scope where `client` and `verify_prompt` were
defined. Python indentation determines block scope.

Measure taken: Moved the verification exception handler back inside
the same workflow block and aligned it with the draft-generation call.

Status: Check the current indentation in `app.py`. The verification
block must remain inside the `if run:` and `with st.status(...)` blocks.

### 4. `503 UNAVAILABLE` --- draft-generation call failed

Observed: The error occurred at
`draft = ask_gemini(client, draft_prompt)`, before verification.

Cause reported by API: Temporary high demand.

Measure taken: Updated `ask_gemini()` to try the configured primary
model and then a fallback model if the first returns a
`503`/`UNAVAILABLE` error. A user-facing exception handler was also
proposed around draft generation so the app can display a message
instead of an unhandled traceback.

Important: The fallback model ID must be supported by the Gemini API
project and account. Fallback is an attempted recovery, not a guarantee.
Verify current model IDs and test with the API key being used.

### Error-handling summary

-   A model `404` is a model-availability/configuration issue; use a
    currently supported model ID.
-   A model `503` is a temporary service issue; retrying or trying an
    available fallback may help.
-   A `NameError` is a Python scope/indentation issue; fix the code
    structure rather than changing the API key.
-   Never claim independent verification succeeded if the verification
    call failed.

## What worked and next improvements

Complete this section with observations from your final test run.

### Implemented

-   Text extraction, chunking, and local document embeddings.
-   Similarity-based retrieval of relevant passages.
-   Gemini answer generation grounded in retrieved evidence.
-   A separate verification prompt comparing the draft answer against
    the evidence.
-   A Streamlit interface for uploading documents and asking questions.


### Improvements to make next

1.  Add bounded retries with exponential backoff for temporary API
    failures.
2.  Validate model availability at startup and configure the model name
    in one place.
3.  Improve source tracking so each retrieved chunk shows the original
    filename and true chunk number.
4.  Add automated tests for chunking, retrieval, missing evidence, and
    API failure handling.
5.  Evaluate retrieval quality and answer faithfulness on a labelled
    test set before making performance claims.
6.  Add OCR support for scanned PDFs.
7.  Improve secret management and API quota/rate-limit messages.
8.  Improve display for long answers and multiple evidence passages.
9.  Add debugging logs without recording API keys or sensitive document
    contents.

## Limitations

-   Generated answers may still be incorrect. A second model call is a
    helpful check, not a guarantee.
-   Retrieval quality depends on extraction, chunking, embeddings, and
    question wording.
-   Scanned PDFs may require OCR.
-   API access, model IDs, quotas, rate limits, and availability depend
    on the Gemini account/project and may change.
-   No benchmark results or production-readiness claims are made here.
    Add measured results only after a reproducible evaluation.


## Project status

DocuMind AI is an internship-oriented prototype demonstrating document
ingestion, retrieval, answer generation, and evidence checking. Before
submission, run the four test questions, capture screenshots of the real
outputs, update the test log, and confirm that the configured Gemini
model IDs work with your API key.
