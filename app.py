import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import json
import tempfile
from openai import OpenAI
from streamlit_js_eval import streamlit_js_eval
from streamlit_modal import Modal

# --- OPENAI SETUP ---
client = OpenAI()  # Uses env var OPENAI_API_KEY

st.set_page_config(page_title="PDF Glossary Mindmap", layout="wide")
st.title("🧠 PDF Glossary Mindmap Explorer")

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

# --- FORCE SHORT GPT-SUMMARIZED TITLE ---
def get_pdf_title(pdf_file, full_text):
    # Step 1: Extract raw title/gist from PDF
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(pdf_file.read())
            tmpfile.flush()
            doc = fitz.open(tmpfile.name)
            title = doc.metadata.get("title")
            if title and title.strip() and title.lower() != "untitled":
                raw_title = title.strip()
            else:
                first_page_text = doc[0].get_text().strip()
                raw_title = next((line.strip() for line in first_page_text.splitlines() if line.strip()), "Document")
    except Exception:
        raw_title = "Document"
    # Step 2: Force short GPT summary for root node (8 words max)
    try:
        prompt = (
            "Summarize this document title or main topic in a concise phrase for a mindmap root node. "
            "Use no more than 8 words, and be specific and clear.\n"
            f"Original title or gist: {raw_title}"
        )
        response = client.responses.create(
            model="gpt-4.1",
            input=prompt,
        )
        short_title = response.output_text.strip().split("\n")[0]
        # Optionally enforce the word limit
        short_title = ' '.join(short_title.split()[:8])
        return short_title
    except Exception:
        return raw_title if len(raw_title.split()) <= 8 else "Document"

# --- GLOSSARY EXTRACTION (GPT-4.1 Responses API) ---
def get_glossary_via_gpt41(full_text, max_terms=20):
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

# --- D3 MINDMAP HTML WITH DRAGGING AND TEXT WRAPPING ---
def create_mindmap_html(glossary, root_title="Glossary"):
    nodes = [
        {"id": root_title, "group": 0}
    ] + [
        {"id": item["term"], "group": 1, "tooltip": item["tooltip"]}
        for item in glossary
    ]
    links = [
        {"source": root_title, "target": item["term"]}
        for item in glossary
    ]
    mindmap_html = f"""
    <div id="mindmap"></div>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
    #mindmap {{ width:100%; height:880px; min-height:700px; background:#f7faff; border-radius:18px; }}
    .tooltip-glossary {{
        position: absolute; pointer-events: none; background: #fff; border: 1.5px solid #4f7cda; border-radius: 8px;
        padding: 10px 13px; font-size: 1em; color: #2c4274; box-shadow: 0 2px 12px rgba(60,100,180,0.15); z-index: 10;
        opacity: 0; transition: opacity 0.18s; max-width: 300px;
    }}
    </style>
    <script>
    const nodes = {json.dumps(nodes)};
    const links = {json.dumps(links)};
    const width = 1400, height = 880;
    const svg = d3.select("#mindmap").append("svg")
        .attr("width", width).attr("height", height);
    const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id(d => d.id).distance(270))
        .force("charge", d3.forceManyBody().strength(-1500))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide().radius(80));

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
        .attr("r", d => d.group === 0 ? 110 : 75)
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

    // Text wrapping for node labels with vertical centering for root
    node.append("text")
        .attr("text-anchor", "middle")
        .style("font-size", d => d.group === 0 ? "1.4em" : "1.08em")
        .each(function(d) {{
            const text = d3.select(this);
            const maxChars = d.group === 0 ? 14 : 16;
            const words = d.id.split(' ');
            let lines = [];
            let current = '';
            words.forEach(word => {{
                if ((current + ' ' + word).trim().length > maxChars) {{
                    lines.push(current.trim());
                    current = word;
                }} else {{
                    current += ' ' + word;
                }}
            }});
            if (current.trim()) lines.push(current.trim());
            const startDy = d.group === 0 ? -((lines.length - 1) / 2) * 1.1 : 0;
            lines.forEach((line, i) => {{
                text.append("tspan")
                    .attr("x", 0)
                    .attr("dy", i === 0 ? `${{startDy}}em` : "1.1em")
                    .text(line);
            }});
        }});

    // DRAG BEHAVIOR
    node.call(
      d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended)
    );

    function dragstarted(event, d) {{
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }}
    function dragged(event, d) {{
        d.fx = event.x;
        d.fy = event.y;
    }}
    function dragended(event, d) {{
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }}

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
if "pdf_title" not in st.session_state:
    st.session_state.pdf_title = None
if "clicked_term" not in st.session_state:
    st.session_state.clicked_term = None

# --- MAIN WORKFLOW ---
if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        full_text = extract_text_from_pdf(uploaded_file)
        st.session_state.full_text = full_text
        st.session_state.pdf_title = get_pdf_title(uploaded_file, full_text)
    with st.spinner("Extracting glossary using GPT-4.1..."):
        glossary = get_glossary_via_gpt41(full_text, max_terms=20)
        st.session_state.glossary = glossary

glossary = st.session_state.get("glossary")
full_text = st.session_state.get("full_text", "")
pdf_title = st.session_state.get("pdf_title", "Document")

# --- SIDEBAR: CSV DOWNLOAD ONLY ---
if glossary:
    csv_data = glossary_to_csv(glossary)
    st.sidebar.download_button(
        label="Download Glossary as CSV",
        data=csv_data,
        file_name="glossary.csv",
        mime="text/csv"
    )

# --- MINDMAP ---
if glossary:
    st.subheader(f"Glossary Mindmap (Root: {pdf_title})")
    mindmap_html = create_mindmap_html(glossary, root_title=pdf_title)
    st.components.v1.html(mindmap_html, height=900, width=1450, scrolling=False)

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
    if clicked_term:
        st.session_state.clicked_term = clicked_term

# --- MODAL SUMMARY ON TERM CLICK ---
if st.session_state.get("clicked_term"):
    term = st.session_state.clicked_term
    modal = Modal(term, key=f"modal_{term}", padding=20)
    modal.open()
    if modal.is_open():
        with modal.container():
            with st.spinner(f"Summarizing '{term}' (with web citations)..."):
                summary = get_summary_with_web_search(term, full_text)
            st.markdown(f"### {term}")
            st.markdown(summary, unsafe_allow_html=True)
            if uploaded_file:
                st.markdown(
                    f'<a href="data:application/pdf;base64,{uploaded_file.getvalue().decode("ISO-8859-1")}" download="{uploaded_file.name}" target="_blank" style="font-weight:bold;">Open PDF</a>',
                    unsafe_allow_html=True
                )
    st.session_state.clicked_term = None

st.caption("Powered by OpenAI GPT-4.1 with live web search for term summaries. © 2025")
