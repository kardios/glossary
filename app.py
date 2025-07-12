import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import json
import tempfile
import openai
from streamlit_js_eval import streamlit_js_eval

# --- CONFIG ---
st.set_page_config(page_title="PDF Glossary Mindmap", layout="wide")
st.title("🧠 PDF Glossary Mindmap Explorer")

# --- SIDEBAR: PDF UPLOAD AND DOWNLOAD ---
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    st.write("---")

# --- HELPER FUNCTIONS ---
def extract_text_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        tmpfile.write(pdf_file.read())
        tmpfile.flush()
        doc = fitz.open(tmpfile.name)
        full_text = "\n\n".join(page.get_text() for page in doc)
    return full_text

def get_glossary_via_gpt41(full_text, max_terms=20):
    prompt = f"""
Extract up to {max_terms} key glossary terms from the following document.
For each, provide a one-sentence definition or explanation as a tooltip.
Return as a JSON array: [{{"term": "...", "tooltip": "..."}}, ...]
Document:
---
{full_text}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",  # Or your GPT-4.1 deployment
        messages=[
            {"role": "system", "content": "You are a helpful glossary extractor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1200
    )
    glossary_json = response.choices[0].message.content
    # Extract just the JSON part (sometimes LLM adds prose)
    start = glossary_json.find('[')
    end = glossary_json.rfind(']')
    if start != -1 and end != -1:
        glossary_json = glossary_json[start:end+1]
    glossary = json.loads(glossary_json)
    return glossary

def get_summary_via_gpt4o(term, full_text):
    prompt = f"""
You are an expert assistant.

Write a concise, one-paragraph explanation of the term "{term}" as used in the following document (do not mention page numbers).
Include 2–3 reputable internet citations (with direct URLs as hyperlinks at the end).

Document context:
---
{full_text}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful, concise academic assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=400
    )
    return response.choices[0].message.content

def glossary_to_csv(glossary):
    df = pd.DataFrame(glossary)
    return df.to_csv(index=False)

def create_mindmap_html(glossary):
    nodes = [
        {"id": "Glossary", "group": 0}
    ] + [
        {"id": item["term"], "group": 1, "tooltip": item["tooltip"]}
        for item in glossary
    ]
    links = [
        {"source": "Glossary", "target": item["term"]}
        for item in glossary
    ]
    mindmap_html = f"""
    <div id="mindmap"></div>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
    #mindmap {{ width:100%; height:640px; min-height:480px; background:#f7faff; border-radius:18px; }}
    .tooltip-glossary {{
        position: absolute; pointer-events: none; background: #fff; border: 1.5px solid #4f7cda; border-radius: 8px;
        padding: 10px 13px; font-size: 1em; color: #2c4274; box-shadow: 0 2px 12px rgba(60,100,180,0.15); z-index: 10;
        opacity: 0; transition: opacity 0.18s; max-width: 260px;
    }}
    </style>
    <script>
    const nodes = {json.dumps(nodes)};
    const links = {json.dumps(links)};
    const width = 900, height = 620;
    const svg = d3.select("#mindmap").append("svg")
        .attr("width", width).attr("height", height);
    const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id(d => d.id).distance(220))
        .force("charge", d3.forceManyBody().strength(-900))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide().radius(70));

    // Links
    const link = svg.append("g")
        .selectAll("line").data(links).enter().append("line")
        .attr("stroke", "#b8cfff").attr("stroke-width", 2);

    // Nodes
    const node = svg.append("g")
        .selectAll("g")
        .data(nodes).enter().append("g")
        .attr("class", "node");

    node.append("circle")
        .attr("r", d => d.group === 0 ? 74 : 56)
        .attr("fill", d => d.group === 0 ? "#eaf0fe" : "#fff")
        .attr("stroke", "#528fff").attr("stroke-width", 3)
        .on("mouseover", function(e, d) {{
            if(d.tooltip) {{
                tooltip.style("opacity", 1).html("<b>" + d.id + "</b><br>" + d.tooltip)
                  .style("left", (e.pageX+12)+"px").style("top", (e.pageY-18)+"px");
            }}
        }})
        .on("mousemove", function(e) {{
            tooltip.style("left", (e.pageX+12)+"px").style("top", (e.pageY-18)+"px");
        }})
        .on("mouseout", function(e, d) {{
            tooltip.style("opacity", 0);
        }})
        .on("click", function(e, d) {{
            if(d.group !== 0) {{
                window.parent.postMessage({{type:'mindmap_click', term:d.id}}, '*');
            }}
        }});

    node.append("text")
        .attr("dy", ".35em")
        .text(d => d.id)
        .style("font-size", d => d.group === 0 ? "1.25em" : "1em")
        .attr("y", d => d.group === 0 ? -6 : 0)
        .style("pointer-events", "none")
        .attr("text-anchor", "middle");

    simulation.on("tick", () => {{
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node
            .attr("transform", d => `translate(${{d.x}},${{d.y}})`);
    }});

    // Tooltip
    const tooltip = d3.select("body").append("div")
        .attr("class", "tooltip-glossary");

    </script>
    """
    return mindmap_html

# --- SESSION STATE ---
if "glossary" not in st.session_state:
    st.session_state.glossary = None
if "full_text" not in st.session_state:
    st.session_state.full_text = ""
if "clicked_term" not in st.session_state:
    st.session_state.clicked_term = None

# --- MAIN WORKFLOW ---
if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        full_text = extract_text_from_pdf(uploaded_file)
        st.session_state.full_text = full_text
    with st.spinner("Extracting glossary using GPT-4.1..."):
        glossary = get_glossary_via_gpt41(full_text, max_terms=20)
        st.session_state.glossary = glossary

glossary = st.session_state.get("glossary")
full_text = st.session_state.get("full_text", "")

# --- SIDEBAR GLOSSARY + DOWNLOAD ---
if glossary:
    st.sidebar.header("Glossary Terms")
    for idx, item in enumerate(glossary):
        if st.sidebar.button(item['term'], key=f"sidebar_{idx}"):
            st.session_state.clicked_term = item['term']
    st.sidebar.write("---")
    csv_data = glossary_to_csv(glossary)
    st.sidebar.download_button(
        label="Download Glossary as CSV",
        data=csv_data,
        file_name="glossary.csv",
        mime="text/csv"
    )

# --- MINIMAP ---
if glossary:
    st.subheader("Glossary Mindmap")
    mindmap_html = create_mindmap_html(glossary)
    st.components.v1.html(mindmap_html, height=670, width=930, scrolling=False)

    # Listen for JS bubble click events
    clicked_term = streamlit_js_eval(
        js_expressions="""
        new Promise(resolve => {
            window.addEventListener('message', function handler(e) {
                if (e.data && e.data.type === 'mindmap_click') {
                    resolve(e.data.term);
                    window.removeEventListener('message', handler);
                }
            });
        })
        """,
        key="mindmap_click",
    )
    # Sidebar or bubble click both set st.session_state.clicked_term
    if clicked_term:
        st.session_state.clicked_term = clicked_term

# --- MODAL SUMMARY ON TERM CLICK ---
if st.session_state.get("clicked_term"):
    term = st.session_state.clicked_term
    with st.spinner(f"Summarizing '{term}'..."):
        summary = get_summary_via_gpt4o(term, full_text)
    st.markdown(f"### {term}")
    st.markdown(summary, unsafe_allow_html=True)
    # Option to open PDF in browser (download if local)
    if uploaded_file:
        st.markdown(
            f'<a href="data:application/pdf;base64,{uploaded_file.getvalue().decode("ISO-8859-1")}" download="{uploaded_file.name}" target="_blank" style="font-weight:bold;">Open PDF</a>',
            unsafe_allow_html=True
        )
    # Reset to avoid double-pop
    st.session_state.clicked_term = None

st.caption("Powered by OpenAI GPT-4.1 and GPT-4o. © 2025")

