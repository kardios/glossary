import streamlit as st
import fitz  # PyMuPDF
import json
import tempfile
from openai import OpenAI
import hashlib
import re
import time

client = OpenAI()

st.set_page_config(page_title="BubbleMap", layout="wide")
st.title("🧠 Bubble Mindmap Explorer")

MAX_TERMS = 16

# --- PDF EXTRACT ---
def extract_text_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        tmpfile.write(pdf_file.read())
        tmpfile.flush()
        doc = fitz.open(tmpfile.name)
        full_text = "\n\n".join(page.get_text() for page in doc)
    return full_text

# --- Robust JSON Extraction ---
def robust_json_extract(raw, want_list=False):
    m = re.search(r'(\{[\s\S]+\})', raw)
    if not m and want_list:
        m = re.search(r'(\[[\s\S]+\])', raw)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    try:
        return json.loads(raw)
    except Exception:
        return None

# --- LLM PROMPTS ---
def prompt_concept_map(full_text, max_terms=MAX_TERMS):
    return (
        f"Extract up to {max_terms} of the most important concepts, technical terms, or keywords from the following document, prioritizing those that are central to its arguments, themes, or subject matter. "
        "For each term, provide a clear and concise one-sentence explanation suitable as a tooltip for a mindmap node.\n\n"
        "Return as a JSON array:\n"
        "[\n"
        '  {"term": "Concept 1", "tooltip": "Short definition or explanation."},\n'
        "  ...\n"
        "]\n"
        "Only return valid JSON; do not include commentary, explanation, or text before or after the JSON.\n\n"
        "Document:\n"
        "---\n"
        f"{full_text}"
    )

def prompt_structure_map(full_text):
    return (
        "Summarize the structure of this document as a hierarchical mindmap. Your mindmap should have:\n"
        "- 3 to 6 major topics at the first level (root children).\n"
        "- 2 to 4 subtopics or key points for each topic (second level).\n"
        "- (Optional) A third level for important supporting details, but only if clearly warranted by the document.\n"
        "Each node must have a \"name\" (the topic/idea) and a \"tooltip\" (a brief description or summary of its meaning or role in the document).\n\n"
        "Return as valid JSON in the following format:\n"
        "{\n"
        '  "name": "Short title of the document",\n'
        '  "tooltip": "Concise summary of the overall subject",\n'
        '  "children": [\n'
        '    {\n'
        '      "name": "Main Topic 1",\n'
        '      "tooltip": "...",\n'
        '      "children": [\n'
        '        {"name": "Subtopic A", "tooltip": "..."},\n'
        '        {"name": "Subtopic B", "tooltip": "..."}\n'
        '      ]\n'
        '    },\n'
        '    ...\n'
        '  ]\n'
        '}\n\n'
        "Only output the JSON. Do not include any commentary or additional explanation.\n\n"
        "Document:\n"
        "---\n"
        f"{full_text}"
    )

def prompt_argument_map(full_text):
    return (
        "Extract the main argument structure from the following document as a hierarchical mindmap. For each node, include:\n"
        '- "name": A very short label (max 4–5 words).\n'
        '- "type": One of: "Thesis", "Supporting Argument", "Evidence", "Counterargument".\n'
        '- "tooltip": A brief summary or example (1–2 sentences).\n\n'
        'Use "Thesis" for the root claim, "Supporting Argument" for reasons/sub-reasons, "Evidence" for supporting facts/examples, and "Counterargument" for objections or opposing points.\n\n'
        "Return valid JSON only, preserving the hierarchy.\n\n"
        "Document:\n"
        "---\n"
        f"{full_text}"
    )

# --- LLM CALLS ---
def get_concept_map(full_text, max_terms=MAX_TERMS):
    input_prompt = prompt_concept_map(full_text, max_terms)
    response = client.responses.create(model="gpt-4.1", input=input_prompt)
    glossary_json = response.output_text
    result = robust_json_extract(glossary_json, want_list=True)
    if not result:
        return []
    return result[:max_terms]

def get_structure_map(full_text):
    prompt = prompt_structure_map(full_text)
    response = client.responses.create(model="gpt-4.1", input=prompt)
    raw = response.output_text
    result = robust_json_extract(raw)
    if not result:
        return {}
    return result

def get_argument_map(full_text):
    prompt = prompt_argument_map(full_text)
    response = client.responses.create(model="gpt-4.1", input=prompt)
    raw = response.output_text
    result = robust_json_extract(raw)
    if not result:
        return {}
    return result

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

def concept_map_to_tree(glossary, root_title="Concept Map"):
    return {
        "name": root_title,
        "tooltip": "Key concepts from the document.",
        "children": [
            {"name": item["term"], "tooltip": item["tooltip"]} for item in glossary
        ]
    }

def flatten_tree_to_nodes_links(tree, parent_name=None, nodes=None, links=None):
    if nodes is None: nodes = []
    if links is None: links = []
    this_id = tree.get("name")
    tooltip = tree.get("tooltip", "")
    node_type = tree.get("type", "")  # May not exist in Concept/Structure map
    nodes.append({"id": this_id, "tooltip": tooltip, "type": node_type})
    if parent_name:
        links.append({"source": parent_name, "target": this_id})
    for child in tree.get("children", []) or []:
        flatten_tree_to_nodes_links(child, this_id, nodes, links)
    return nodes, links

def create_multilevel_mindmap_html(tree, center_title="Root", mode="concept"):
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
    .legend-container {{
        margin-bottom: 10px;
        margin-top: 5px;
    }}
    .legend-item {{
        display: inline-block;
        margin-right: 18px;
        font-size: 1em;
        vertical-align: middle;
    }}
    .legend-circle {{
        display: inline-block;
        width: 17px; height: 17px;
        border-radius: 50%;
        margin-right: 7px;
        vertical-align: middle;
    }}
    </style>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <div class="legend-container">
      {"<span class='legend-item'><span class='legend-circle' style='background:#3B82F6;'></span>Thesis</span>" if mode=='argument' else ""}
      {"<span class='legend-item'><span class='legend-circle' style='background:#22C55E;'></span>Supporting Argument</span>" if mode=='argument' else ""}
      {"<span class='legend-item'><span class='legend-circle' style='background:#D1D5DB;'></span>Evidence</span>" if mode=='argument' else ""}
      {"<span class='legend-item'><span class='legend-circle' style='background:#F87171;'></span>Counterargument</span>" if mode=='argument' else ""}
    </div>
    <script>
    const nodes = {nodes_json};
    const links = {links_json};
    const width = 1400, height = 900;
    const rootID = "{center_title.replace('"', '\\"')}";

    function getNodeColor(type, id) {{
        if ("{mode}" !== "argument") {{
            return id === rootID ? "#eaf0fe" : "#fff";
        }}
        // Argument map colors
        const t = (type || "").toLowerCase();
        if (t === "thesis") return "#3B82F6";
        if (t === "supporting argument") return "#22C55E";
        if (t === "evidence") return "#D1D5DB";
        if (t === "counterargument") return "#F87171";
        return "#D1D5DB";
    }}

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
        .attr("fill", d => getNodeColor(d.type, d.id))
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

# --- HTML WRAPPER FOR DOWNLOAD ---
def full_html_wrap(mindmap_html, title="Bubble Mindmap Explorer"):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {{ margin:0; background:#f7faff; font-family:sans-serif; }}
  </style>
</head>
<body>
{mindmap_html}
</body>
</html>
"""

# --- EXPORT FORMATTING ---
def concept_map_txt(glossary):
    txt = ""
    for item in glossary:
        txt += f"Term: {item['term']}\nDefinition: {item['tooltip']}\n\n"
    return txt

def tree_map_txt(tree, level=0):
    txt = ""
    indent = "  " * level
    txt += f"{indent}- {tree.get('name', '')}: {tree.get('tooltip', '')}\n"
    for child in tree.get("children", []) or []:
        txt += tree_map_txt(child, level+1)
    return txt

def argument_map_txt(tree, level=0):
    txt = ""
    indent = "  " * level
    node_type = tree.get("type", "")
    txt += f"{indent}- [{node_type}] {tree.get('name', '')}: {tree.get('tooltip', '')}\n"
    for child in tree.get("children", []) or []:
        txt += argument_map_txt(child, level+1)
    return txt

# --- SESSION STATE INIT ---
for key in ["file_hash", "full_text", "pdf_title", "concept_map", "structure_map", "argument_map", "view_mode"]:
    if key not in st.session_state:
        st.session_state[key] = None

# --- SIDEBAR: File upload and view mode selection ---
with st.sidebar:
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    view_mode = st.radio(
        "Show as:",
        ["Concept Map", "Structure Map", "Argument Map"],
        index=0,
        key="view_mode"
    )
    st.write("---")

# --- Compute file hash ---
def compute_file_hash(file_obj):
    file_obj.seek(0)
    data = file_obj.read()
    file_obj.seek(0)
    return hashlib.md5(data).hexdigest()

# --- SIDEBAR: Live log and progress bar ---
log_box = st.sidebar.empty()
progress_bar = st.sidebar.progress(0)
log_msgs = []

def log(msg):
    log_msgs.append(msg)
    log_box.markdown("### Log\n" + "\n".join(f"- {m}" for m in log_msgs))

# --- FILE UPLOAD & PROCESSING ---
if uploaded_file:
    file_hash = compute_file_hash(uploaded_file)
    if st.session_state.file_hash != file_hash:
        st.session_state.file_hash = file_hash

        # Clear the log for each new file
        log_msgs.clear()
        log_box.markdown("### Log")

        # Step 1: Extract text
        log("📄 Extracting text from PDF...")
        progress_bar.progress(0)
        t0 = time.perf_counter()
        full_text = extract_text_from_pdf(uploaded_file)
        t1 = time.perf_counter()
        st.session_state.full_text = full_text
        st.session_state.pdf_title = get_pdf_title_from_content(full_text)
        st.session_state["extract_text_time"] = t1 - t0
        log(f"🟢 Text extracted in {t1 - t0:.2f}s")
        progress_bar.progress(0.25)

        # Step 2: Concept map
        log("🧠 Generating concept map...")
        t0 = time.perf_counter()
        st.session_state.concept_map = get_concept_map(st.session_state.full_text, MAX_TERMS)
        t1 = time.perf_counter()
        st.session_state["concept_map_time"] = t1 - t0
        log(f"🟢 Concept map generated in {t1 - t0:.2f}s")
        progress_bar.progress(0.5)

        # Step 3: Structure map
        log("🗂️ Generating structure map...")
        t0 = time.perf_counter()
        st.session_state.structure_map = get_structure_map(st.session_state.full_text)
        t1 = time.perf_counter()
        st.session_state["structure_map_time"] = t1 - t0
        log(f"🟢 Structure map generated in {t1 - t0:.2f}s")
        progress_bar.progress(0.75)

        # Step 4: Argument map
        log("⚖️ Generating argument map...")
        t0 = time.perf_counter()
        st.session_state.argument_map = get_argument_map(st.session_state.full_text)
        t1 = time.perf_counter()
        st.session_state["argument_map_time"] = t1 - t0
        log(f"🟢 Argument map generated in {t1 - t0:.2f}s")
        progress_bar.progress(1.0)

        # Show step timings
        log("---")
        log(f"**Summary**")
        log(f"Text extraction: {st.session_state['extract_text_time']:.2f} s")
        log(f"Concept map: {st.session_state['concept_map_time']:.2f} s")
        log(f"Structure map: {st.session_state['structure_map_time']:.2f} s")
        log(f"Argument map: {st.session_state['argument_map_time']:.2f} s")

concept_map = st.session_state.get("concept_map")
structure_map = st.session_state.get("structure_map")
argument_map = st.session_state.get("argument_map")
pdf_title = st.session_state.get("pdf_title", "Document")
view_mode = st.session_state.get("view_mode", "Concept Map")

# --- MAIN DISPLAY ---
if uploaded_file and concept_map and structure_map and argument_map:
    if view_mode == "Concept Map":
        concept_tree = concept_map_to_tree(concept_map, root_title=pdf_title)
        mindmap_html = create_multilevel_mindmap_html(concept_tree, center_title=pdf_title, mode="concept")
        st.components.v1.html(mindmap_html, height=900, width=1450, scrolling=False)
        txt_data = concept_map_txt(concept_map)
        st.sidebar.download_button(
            label="Download Concept Map as TXT",
            data=txt_data,
            file_name="concept_map.txt",
            mime="text/plain"
        )
        # --- HTML download button ---
        html_file = full_html_wrap(mindmap_html, title=f"BubbleMap - {pdf_title} (Concept Map)")
        st.sidebar.download_button(
            label="Download Concept Map as HTML",
            data=html_file.encode("utf-8"),
            file_name="concept_map.html",
            mime="text/html"
        )

    elif view_mode == "Structure Map":
        if structure_map and structure_map.get("children"):
            mindmap_html = create_multilevel_mindmap_html(structure_map, center_title=structure_map.get("name", "Root"), mode="structure")
            st.components.v1.html(mindmap_html, height=900, width=1450, scrolling=False)
            txt_data = tree_map_txt(structure_map)
            st.sidebar.download_button(
                label="Download Structure Map as TXT",
                data=txt_data,
                file_name="structure_map.txt",
                mime="text/plain"
            )
            # --- HTML download button ---
            html_file = full_html_wrap(mindmap_html, title=f"BubbleMap - {pdf_title} (Structure Map)")
            st.sidebar.download_button(
                label="Download Structure Map as HTML",
                data=html_file.encode("utf-8"),
                file_name="structure_map.html",
                mime="text/html"
            )
        else:
            st.info("No structure was extracted.")

    elif view_mode == "Argument Map":
        if argument_map and argument_map.get("children"):
            mindmap_html = create_multilevel_mindmap_html(argument_map, center_title=argument_map.get("name", "Root"), mode="argument")
            st.components.v1.html(mindmap_html, height=900, width=1450, scrolling=False)
            txt_data = argument_map_txt(argument_map)
            st.sidebar.download_button(
                label="Download Argument Map as TXT",
                data=txt_data,
                file_name="argument_map.txt",
                mime="text/plain"
            )
            # --- HTML download button ---
            html_file = full_html_wrap(mindmap_html, title=f"BubbleMap - {pdf_title} (Argument Map)")
            st.sidebar.download_button(
                label="Download Argument Map as HTML",
                data=html_file.encode("utf-8"),
                file_name="argument_map.html",
                mime="text/html"
            )
        else:
            st.info("No argument structure was extracted.")

st.caption("Powered by OpenAI GPT-4.1. Explore any PDF as a mindmap. © 2025")
