import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import json
import tempfile
from openai import OpenAI
from streamlit_agraph import agraph, Node, Edge, Config

# --- OPENAI SETUP ---
client = OpenAI()

st.set_page_config(page_title="PDF Glossary Network Map", layout="wide")
st.title("🧠 PDF Glossary Bubble Network (with Clickable Nodes)")

# --- PDF UPLOAD ---
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    st.write("---")

# --- PDF TEXT EXTRACTION ---
def extract_text_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        tmpfile.write(pdf_file.read())
        tmpfile.flush()
        doc = fitz.open(tmpfile.name)
        full_text = "\n\n".join(page.get_text() for page in doc)
    return full_text

# --- LLM TITLE FROM CONTENT ---
def get_pdf_title_from_content(full_text, max_words=8, chunk_size=1000):
    chunk = ' '.join(full_text.split()[:chunk_size])
    prompt = (
        f"Based on the following text, summarize the main topic or theme in a short, clear phrase suitable as the root node of a mindmap. "
        f"Use no more than {max_words} words.\n\n"
        f"Text:\n{chunk}"
    )
    try:
        response = client.responses.create(
            model="gpt-4.1",
            input=prompt,
        )
        short_title = response.output_text.strip().split("\n")[0]
        short_title = ' '.join(short_title.split()[:max_words])
        if not short_title or "please provide" in short_title.lower():
            return "Untitled Document"
        return short_title
    except Exception:
        return "Untitled Document"

# --- GLOSSARY EXTRACTION (GPT-4.1 Responses API) ---
def get_glossary_via_gpt41(full_text, max_terms=16):
    input_prompt = (
        f"Extract up to {max_terms} key glossary terms from the following document.\n"
        "For each, provide a one-sentence definition or explanation as a tooltip.\n"
        "Return as a JSON array: [{\"term\": \"...\", \"tooltip\": \"...\"}, ...]\n"
        "Document:\n"
        "---\n"
        f"{full_text}"
    )
    response = client.responses.create(
        model="gpt-4.1",
        input=input_prompt,
    )
    glossary_json = response.output_text
    start = glossary_json.find('[')
    end = glossary_json.rfind(']')
    if start != -1 and end != -1:
        glossary_json = glossary_json[start:end+1]
    return json.loads(glossary_json)

# --- SUMMARY WITH WEB SEARCH (GPT-4.1 Responses API) ---
def get_summary_with_web_search(term, context_text=None):
    base_prompt = (
        f"Write a concise, one-paragraph explanation of the term '{term}'. "
        "Include up-to-date, reputable internet citations (with direct URLs as hyperlinks) in your answer. "
    )
    if context_text:
        base_prompt += f"Use this as reference context (if helpful):\n{context_text}\n"
    response = client.responses.create(
        model="gpt-4.1",
        tools=[{"type": "web_search_preview"}],
        input=base_prompt,
    )
    return response.output_text

# --- CSV EXPORT ---
def glossary_to_csv(glossary):
    df = pd.DataFrame(glossary)
    return df.to_csv(index=False)

# --- SESSION STATE ---
if "glossary" not in st.session_state:
    st.session_state.glossary = None
if "full_text" not in st.session_state:
    st.session_state.full_text = ""
if "pdf_title" not in st.session_state:
    st.session_state.pdf_title = None
if "selected_term" not in st.session_state:
    st.session_state.selected_term = None

# --- MAIN WORKFLOW ---
if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        full_text = extract_text_from_pdf(uploaded_file)
        st.session_state.full_text = full_text
        st.session_state.pdf_title = get_pdf_title_from_content(full_text)
    with st.spinner("Extracting glossary using GPT-4.1..."):
        glossary = get_glossary_via_gpt41(full_text, max_terms=16)
        st.session_state.glossary = glossary

glossary = st.session_state.get("glossary")
full_text = st.session_state.get("full_text", "")
pdf_title = st.session_state.get("pdf_title", "Document")

# --- SIDEBAR: CSV DOWNLOAD ---
if glossary:
    csv_data = glossary_to_csv(glossary)
    st.sidebar.download_button(
        label="Download Glossary as CSV",
        data=csv_data,
        file_name="glossary.csv",
        mime="text/csv"
    )

# --- NETWORK GRAPH / BUBBLE MAP ---
if glossary:
    st.subheader(f"Clickable Glossary Bubble Network (Root: {pdf_title})")

    # Build nodes and edges for agraph
    nodes = [
        Node(id=pdf_title, label=pdf_title, size=50, shape="CIRCLE", color="#eaf0fe", font={"size": 32, "strokeWidth": 2, "strokeColor": "#528fff"})
    ]
    edges = []
    for g in glossary:
        nodes.append(
            Node(
                id=g["term"],
                label=g["term"],
                size=35,
                shape="CIRCLE",
                title=g["tooltip"],  # Tooltip
                color="#ffffff",
                font={"size": 22}
            )
        )
        edges.append(Edge(source=pdf_title, target=g["term"]))

    config = Config(
        width=1400,
        height=850,
        directed=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False,
        node={"labelProperty": "label"},
        link={"highlightColor": "#A5AAFF"},
        staticGraph=False,
        panAndZoom=True,
    )

    selected = agraph(nodes=nodes, edges=edges, config=config)
    # selected is the node label (term) if a node is clicked, else None

    if selected and selected != pdf_title:
        st.session_state.selected_term = selected

# --- SUMMARY IN MAIN PANEL ---
if st.session_state.get("selected_term"):
    term = st.session_state.selected_term
    st.markdown(f"## {term}")
    with st.spinner(f"Generating summary for '{term}' (with web citations)..."):
        summary = get_summary_with_web_search(term, full_text)
    st.markdown(summary, unsafe_allow_html=True)
    # Optionally, clear selected term after showing summary (so clicking again triggers a refresh)
    # st.session_state.selected_term = None

st.caption("Powered by OpenAI GPT-4.1 with web search. Click any bubble for a summary with citations. © 2025")
