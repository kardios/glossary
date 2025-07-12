import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import json
import tempfile
from openai import OpenAI
import hashlib

client = OpenAI()
st.set_page_config(page_title="PDF Mindmap Explorer", layout="wide")
st.title("🧠 PDF Multi-level Bubble Mindmap Explorer")

MAX_TERMS = 16

def extract_text_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        tmpfile.write(pdf_file.read())
        tmpfile.flush()
        doc = fitz.open(tmpfile.name)
        full_text = "\n\n".join(page.get_text() for page in doc)
    return full_text

def prompt_glossary(full_text, max_terms=MAX_TERMS):
    return (
        f"Extract up to {max_terms} important glossary terms or concepts from the following document.\n"
        "For each, provide a one-sentence definition as a tooltip.\n"
        "Return as a JSON array: [{\"term\": \"...\", \"tooltip\": \"...\"}, ...]\n"
        "Document:\n"
        "---\n"
        f"{full_text}"
    )

def prompt_hierarchical_mindmap(full_text):
    return (
        "Summarize the main ideas in this document as a hierarchical mindmap, aiming for 2 to 3 levels of topics and subtopics if the structure allows. "
        "Identify 3–6 major topics (first level). For each topic, list 2–4 key sub-ideas (second level). If helpful, you may add a third level for important details, but do not force an extra level if it does not fit the document’s structure. "
        "For each node, provide a short tooltip. "
        "Return as JSON like this:\n"
        "{"
        "\"name\": \"Document Title\","
        "\"tooltip\": \"...\","
        "\"children\": ["
        "  {"
        "    \"name\": \"Main Topic 1\", \"tooltip\": \"...\", \"children\": ["
        "      {\"name\": \"Sub-idea 1\", \"tooltip\": \"...\"},"
        "      {\"name\": \"Sub-idea 2\", \"tooltip\": \"...\", \"children\": ["
        "         {\"name\": \"Detail A\", \"tooltip\": \"...\"}"
        "      ]}"
        "    ]"
        "  }"
        "]"
        "}\n"
        "Document:\n"
        "---\n"
        f"{full_text}"
    )

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

def get_glossary(full_text, max_terms=MAX_TERMS):
    input_prompt = prompt_glossary(full_text, max_terms)
    response = client.responses.create(model="gpt-4.1", input=input_prompt)
    glossary_json = response.output_text
    start = glossary_json.find('[')
    end = glossary_json.rfind(']')
    if start != -1 and end != -1:
        glossary_json = glossary_json[start:end+1]
    glossary = json.loads(glossary_json)
    return glossary[:max_terms]

def get_hierarchical_mindmap(full_text):
    prompt = prompt_hierarchical_mindmap(full_text)
    response = client.responses.create(model="gpt-4.1", input=prompt)
    raw = response.output_text
    first = raw.find('{')
    last = raw.rfind('}')
    if first != -1 and last != -1:
        try:
            return json.loads(raw[first:last+1])
        except Exception:
            pass
    return {}

def glossary_to_tree(glossary, root_title="Glossary"):
    return {
        "name": root_title,
        "tooltip": "Glossary of key terms.",
        "children": [
            {"name": item["term"], "tooltip": item["tooltip"]} for item in glossary
        ]
    }

def flatten_tree_to_nodes_links(tree, parent_name=None, nodes=None, links=None):
    if nodes is None: nodes = []
    if links is None: links = []
    this_id = tree.get("name")
    tooltip = tree.get("tooltip", "")
    nodes.append({"id": this_id, "tooltip": tooltip})
    if parent_name:
        links.append({"source": parent_name, "target": this_id})
    for child in tree.get("children", []):
        flatten_tree_to_nodes_links(child, this_id, nodes, links)
    return nodes, links

def create_multilevel_mindmap_html(tree, center_title="Root"):
    nodes, links = flatten_tree_to_nodes_links(tree)
    for n in nodes:
        n["group"] = 0 if n["id"] == center_title else 1

    nodes_json = json.dumps(nodes)
    links_json = json.dumps(links)
    mindmap_html = f"""
    <div id="mindmap"></div>
    <style>
    #mindmap {{ width:100%; height:880px; min-height:700px; background:#f7faff; border-radius:18px; }}
    .tooltip-glossary {{
        position: absolute; pointer-events: none; background: #fff; border: 1.5px solid #4f7cda; border-radius: 8px;
        padding: 10px 13px; font-size: 1em; color: #2c4274; box-shadow: 0 2px 12px rgba(60,100,180,0.15); z-index: 10;
        opacity: 0; transition: opacity 0.18s; max-width: 320px;
    }}
    </style>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
    const nodes = {nodes_json};
    const links = {links_json};
    const width = 1400, height = 900;
    const rootID = "{center_title.replace('"', '\\"')}";

    const svg = d3.select("#mindmap").append("svg")
        .attr("width", width)
        .attr("height", height)
        .style("background", "#f7faff");

    // --- GROUP for PAN & ZOOM ---
    const container = svg.append("g");

    // --- ZOOM BEHAVIOR ---
    svg.call(
        d3.zoom()
          .scaleExtent([0.3, 2.5])
          .on("zoom", (event) => container.attr("transform", event.transform))
    );

    const link = container.append("g")
        .selectAll("line").data(links).enter().append("line")
        .attr("stroke", "#b8cfff").attr("stroke-width", 2);

    const node = container.append("g")
        .selectAll("g")
        .data(nodes).enter().append("g")
        .attr("class", "node");

    node.append("circle")
        .attr("r", d => d.id === rootID ? 110 : 75)
        .attr("fill", d => d.id === rootID ? "#eaf0fe" : "#fff")
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
        }});

    node.append("text")
        .attr("text-anchor", "middle")
        .style("font-size", d => d.id === rootID ? "1.4em" : "1.08em")
        .each(function(d) {{
            const text = d3.select(this);
            const maxChars = d.id === rootID ? 14 : 16;
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
            const startDy = d.id === rootID ? -((lines.length - 1) / 2) * 1.1 : 0;
            lines.forEach((line, i) => {{
                text.append("tspan")
                    .attr("x", 0)
                    .attr("dy", i === 0 ? `${{startDy}}em` : "1.1em")
                    .text(line);
            }});
        }});

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

    const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id(d => d.id).distance(d => d.source === rootID ? 270 : 180))
        .force("charge", d3.forceManyBody().strength(-1400))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide().radius(82));

    simulation.on("tick", () => {{
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node
            .attr("transform", d => `translate(${{d.x}},${{d.y}})`);
    }});

    const tooltip = d3.select("body").append("div")
        .attr("class", "tooltip-glossary");
    </script>
    """
    return mindmap_html

# --- SESSION STATE INIT ---
for key in ["file_hash", "full_text", "pdf_title", "glossary", "hierarchical", "view_mode"]:
    if key not in st.session_state:
        st.session_state[key] = None

with st.sidebar:
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    view_mode = st.radio(
        "Show as:",
        ["Glossary Bubble Map", "Multi-level Mindmap"],
        index=0,
        key="view_mode"
    )
    st.write("---")

def compute_file_hash(file_obj):
    file_obj.seek(0)
    data = file_obj.read()
    file_obj.seek(0)
    return hashlib.md5(data).hexdigest()

if uploaded_file:
    file_hash = compute_file_hash(uploaded_file)
    if st.session_state.file_hash != file_hash:
        st.session_state.file_hash = file_hash
        with st.spinner("Extracting text from PDF..."):
            full_text = extract_text_from_pdf(uploaded_file)
            st.session_state.full_text = full_text
            st.session_state.pdf_title = get_pdf_title_from_content(full_text)
        with st.spinner("Extracting glossary terms..."):
            st.session_state.glossary = get_glossary(st.session_state.full_text, MAX_TERMS)
        with st.spinner("Extracting document hierarchy..."):
            st.session_state.hierarchical = get_hierarchical_mindmap(st.session_state.full_text)

glossary = st.session_state.get("glossary")
pdf_title = st.session_state.get("pdf_title", "Document")
hierarchical = st.session_state.get("hierarchical")
view_mode = st.session_state.get("view_mode", "Glossary Bubble Map")

if uploaded_file and glossary:
    st.subheader(f"{view_mode} (Root: {pdf_title})")
    if view_mode == "Glossary Bubble Map":
        csv_data = pd.DataFrame(glossary).to_csv(index=False)
        with st.sidebar:
            st.download_button(
                label="Download Glossary as CSV",
                data=csv_data,
                file_name="glossary.csv",
                mime="text/csv"
            )
        glossary_tree = glossary_to_tree(glossary, root_title=pdf_title)
        mindmap_html = create_multilevel_mindmap_html(glossary_tree, center_title=pdf_title)
        st.components.v1.html(mindmap_html, height=900, width=1450, scrolling=False)
    elif view_mode == "Multi-level Mindmap":
        if hierarchical and hierarchical.get("children"):
            mindmap_html = create_multilevel_mindmap_html(hierarchical, center_title=hierarchical.get("name", "Root"))
            st.components.v1.html(mindmap_html, height=900, width=1450, scrolling=False)
        else:
            st.info("No hierarchical structure was extracted.")

st.caption("Powered by OpenAI GPT-4.1. Now with zoom/pan navigation. © 2025")
