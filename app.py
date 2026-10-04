import os
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai

st.set_page_config(page_title="DocuMind AI", page_icon="🔎", layout="wide")

st.title("🔎 DocuMind AI")
st.caption("A source-grounded document research assistant with a retrieve → answer → verify workflow.")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_text(uploaded_file):
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return uploaded_file.getvalue().decode("utf-8", errors="ignore")

def chunk_text(text, chunk_size=900, overlap=150):
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks

def retrieve(question, chunks, embeddings, embedder, k=4):
    q_embedding = embedder.encode([question], normalize_embeddings=True)[0]
    scores = embeddings @ q_embedding
    indices = np.argsort(scores)[::-1][:k]
    return [{"text": chunks[i], "score": float(scores[i]), "source": f"Chunk {i+1}"} for i in indices]

# def ask_gemini(client, prompt):
#     response = client.models.generate_content(
#         model="gemini-3.8-flash",
#         contents=prompt
#     )
#     return response.text or ""

def ask_gemini(client, prompt):
    last_error = None

    for model in ("gemini-3.8-flash", "gemini-3.5-flash-lite"):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            return response.text or ""

        except Exception as e:
            message = str(e).upper()

            if "503" in message or "UNAVAILABLE" in message:
                last_error = e
                continue

            raise

    raise last_error

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Gemini API key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    st.caption("Get a key from Google AI Studio. The key is used locally and is not saved by this app.")
    uploaded_files = st.file_uploader("Upload PDF or TXT documents", type=["pdf", "txt"], accept_multiple_files=True)
    st.markdown("---")
    st.markdown("**Workflow**")
    st.write("1. Extract & chunk")
    st.write("2. Embed & retrieve")
    st.write("3. Generate grounded answer")
    st.write("4. Verify answer against evidence")

if not uploaded_files:
    st.info("Upload one or more PDF/TXT files to begin. Or try the sample document included in the GitHub repository.")
    st.markdown("**Demo questions**")
    st.write("- What are the main risks?")
    st.write("- Summarize the recommendations.")
    st.write("- What evidence supports the conclusion?")
    st.stop()

all_chunks, chunk_sources = [], []
for file in uploaded_files:
    raw_text = extract_text(file)
    pieces = chunk_text(raw_text)
    all_chunks.extend(pieces)
    chunk_sources.extend([file.name] * len(pieces))

if not all_chunks:
    st.error("No extractable text found. For scanned PDFs, OCR is needed before using this app.")
    st.stop()

embedder = load_embedder()
with st.spinner("Creating document embeddings…"):
    doc_embeddings = embedder.encode(all_chunks, normalize_embeddings=True, show_progress_bar=False)
    doc_embeddings = np.asarray(doc_embeddings, dtype=np.float32)

st.success(f"Indexed {len(uploaded_files)} document(s) into {len(all_chunks)} chunks.")

question = st.text_input("Ask a question about your documents", placeholder="What are the main findings and supporting evidence?")
run = st.button("Research answer", type="primary", disabled=not question.strip())

if run:
    if not api_key:
        st.error("Add your Gemini API key in the sidebar first.")
        st.stop()

    client = genai.Client(api_key=api_key)

    with st.status("Running the agent workflow…", expanded=True) as status:
        st.write("Step 1/3 · Retrieving relevant passages")
        retrieved = retrieve(question, all_chunks, doc_embeddings, embedder)
        evidence = "\n\n".join(
            f"[{i+1}] Source: {chunk_sources[int(item['source'].split()[-1])-1] if False else item['source']} | Similarity: {item['score']:.3f}\n{item['text']}"
            for i, item in enumerate(retrieved)
        )
        st.write("Step 2/3 · Drafting an answer from retrieved evidence")
        draft_prompt = f"""You are DocuMind, a careful document research assistant.
Answer the user's question using ONLY the evidence below.
If evidence is insufficient, say so. Do not invent facts.
Cite supporting passages using [1], [2], etc. Keep the answer clear and concise.

QUESTION:
{question}

RETRIEVED EVIDENCE:
{evidence}

Draft answer:"""
        draft = ask_gemini(client, draft_prompt)

        st.write("Step 3/3 · Checking whether the answer is supported")
        verify_prompt = f"""You are an evidence verifier. Compare the proposed answer against the supplied evidence.
Return exactly:
VERDICT: SUPPORTED, PARTIALLY SUPPORTED, or UNSUPPORTED
REASON: one or two concise sentences.
If unsupported, identify the claim that lacks evidence.

QUESTION: {question}
EVIDENCE:
{evidence}
PROPOSED ANSWER:
{draft}
"""
        try:
            verification = ask_gemini(client, verify_prompt)
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                verification = (
                    "Verification temporarily unavailable due to Gemini server load. "
                    "The answer was generated, but its evidence could not be "
                    "independently verified. Please check the cited source excerpts."
                )
                st.warning(
                    "Gemini is experiencing high demand. Showing the answer "
                    "without independent verification."
                )
            else:
                raise

        status.update(label="Workflow complete", state="complete", expanded=False)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Answer")
        st.markdown(draft)
    with right:
        st.subheader("Evidence check")
        st.markdown(verification)
        st.metric("Retrieved passages", len(retrieved))

    with st.expander("Inspect retrieved passages"):
        for i, item in enumerate(retrieved, start=1):
            st.markdown(f"**[{i}] {item['source']} · similarity {item['score']:.3f}**")
            st.write(item["text"])
            st.divider()

st.markdown("---")
st.caption("Prototype note: semantic similarity is a retrieval signal, not a guarantee of factual correctness. The verifier is an LLM-based check and can also make mistakes.")
