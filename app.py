import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import json
import tempfile
from openai import OpenAI
import hashlib
import re

client = OpenAI()

# --- APP NAME & PAGE CONFIG ---
st.set_page_config(page_title="Bubble Mindmap Explorer", layout="wide")
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

# --- PROMPTS ---
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

def prompt_hierarchical_map(full_text):
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

def prompt_stakeholder_map(full_text):
    return (
        "Extract all stakeholder groups mentioned or implied in the document—such as organizations, agencies, interest groups, professions, communities, or segments of the public. For each stakeholder:\n"
        '- "id": Group or organization name (as it appears)\n'
        '- "role": One sentence on their role, interest, or position in the context of the document\n\n'
        "Then, for each direct relationship or interaction (e.g., collaboration, influence, conflict, regulation, support, opposition, dependency, or affected-by), list an edge:\n"
        '- "source": Stakeholder 1\n'
        '- "target": Stakeholder 2\n'
        '- "relationship": Short label (e.g., \"regulates\", \"supports\", \"opposes\", \"influences\", \"benefits from\")\n\n'
        "Return valid JSON:\n"
        "{\n"
        '  "nodes": [\n'
        '    {"id": "Ministry of Health", "role": "Sets policy and regulates healthcare providers."},\n'
        '    {"id": "Hospitals", "role": "Provide health services to patients."},\n'
        '    {"id": "Patients", "role": "Receive care and are directly affected by policy."}\n'
        '  ],\n'
        '  "edges": [\n'
        '    {"source": "Ministry of Health", "target": "Hospitals", "relationship": "regulates"},\n'
        '    {"source": "Hospitals", "target": "Patients", "relationship": "serves"}\n'
        '  ]\n'
        '}\n\n'
        "Do not add explanations or commentary. Only output the JSON.\n\n"
        "Document:\n"
        "---\n"
        f"{full_text}"
    )

# --- LLM CALLS ---
def get_concept_map(full_text, max_terms=MAX_TERMS):
    input_prompt = prompt_concept_map(full_text, max_terms)
    response = client.responses.create(model="gpt-4.1", input=input_prompt)
    glossary_json = response.output_text
    start = glossary_json.find('[')
    end = glossary_json.rfind(']')
    if start != -1 and end != -1:
        glossary_json = glossary_json[start:end+1]
    glossary = json.loads(glossary_json)
    return glossary[:max_terms]

def get_hierarchical_mindmap(full_text):
    prompt = prompt_hierarchical_map(full_text)
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

def get_stakeholder_map(full_text):
    prompt = prompt_stakeholder_map(full_text)
    response = client.responses.create(model="gpt-4.1", input=prompt)
    raw = response.output_text
    match = re.search(r'({.*})', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception as e:
            st.warning(f"Failed to parse stakeholder map JSON: {e}")
    else:
        st.info("No valid stakeholder map JSON found.")
    return {"nodes": [], "edges": []}

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

def concept_map_to_tree(glossary, root_title="Glossary"):
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

def create_stakeholder_map_html(stakeholder_data):
    nodes = stakeholder_data.get("nodes", [])
    edges = stakeholder_data.get("edges", [])
    color = "#A7C7E7"
    nodes_json = json.dumps([
        {**node, "color": color}
        for node in nodes
    ])
    edges_json = json.dumps(edges)
    html = f"""
    <div id="stakeholdermap"></div>
    <style>
    #stakeholdermap {{ width:100%; height:880px; background:#f7faff; border-radius:18px; }}
    .tooltip-entity {{
        position: absolute; pointer-events: none; background: #fff; border: 1.5px solid #666; border-radius: 8px;
        padding: 10px 13px; font-size: 1em; color: #2c4274; box-shadow: 0 2px 12px rgba(60,100,180,0.15); z-index: 10;
        opacity: 0; transition: opacity 0.18s; max-width: 350px;
    }}
    </style>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
    const nodes = {nodes_json};
    const links = {edges_json};
    const width = 1400, height = 900;
    const svg = d3.select("#stakeholdermap").append("svg")
        .attr("width", width).attr("height", height)
        .style("background", "#f7faff");
    const container = svg.append("g");
    svg.call(
        d3.zoom().scaleExtent([0.3, 2.5]).on("zoom", (event) => container.attr("transform", event.transform))
    );
    const link = container.append("g")
        .selectAll("line").data(links).enter().append("line")
        .attr("stroke", "#b8cfff").attr("stroke-width", 2);

    const node = container.append("g")
        .selectAll("g")
        .data(nodes).enter().append("g")
        .attr("class", "node");
    node.append("circle")
        .attr("r", 65)
        .attr("fill", d => d.color)
        .attr("stroke", "#528fff").attr("stroke-width", 3)
        .on("mouseover", function(e, d) {{
            tooltip.style("opacity", 1).html("<b>" + d.id + "</b><br>(" + d.role + ")")
              .style("left", (e.pageX+12)+"px").style("top", (e.pageY-18)+"px");
        }})
        .on("mousemove", function(e) {{
            tooltip.style("left", (e.pageX+12)+"px").style("top", (e.pageY-18)+"px");
        }})
        .on("mouseout", function(e, d) {{
            tooltip.style("opacity", 0);
        }});

    node.append("text")
        .attr("text-anchor", "middle")
        .style("font-size", "1.05em")
        .each(function(d) {{
            const text = d3.select(this);
            const maxChars = 15;
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
            const startDy = -((lines.length - 1) / 2) * 1.1;
            lines.forEach((line, i) => {{
               
