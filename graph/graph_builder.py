"""
Knowledge Graph generator using NetworkX and standalone interactive Vis.js HTML export.
Connects research papers and extracted concepts into a browsable scientific map.
"""

import os
import json
from typing import Dict, Any, List
import networkx as nx
from storage.database import Database

class KnowledgeGraphBuilder:
    def __init__(self, db: Database, output_html: str = "data/graph.html"):
        self.db = db
        self.output_html = output_html
        os.makedirs(os.path.dirname(os.path.abspath(self.output_html)), exist_ok=True)

    def build_graph(self) -> nx.Graph:
        """
        Constructs a NetworkX graph linking papers and their extracted concepts/keywords.
        Also connects papers to each other based on shared concept overlap.
        """
        G = nx.Graph()
        papers = self.db.get_all_papers_for_graph()
        entities = self.db.get_graph_entities()

        # Map paper_id to its entities
        paper_entities: Dict[int, List[str]] = {}
        for ent in entities:
            pid = ent["paper_id"]
            ename = ent["entity_name"]
            paper_entities.setdefault(pid, []).append(ename)

            # Add concept node
            if not G.has_node(f"concept:{ename}"):
                G.add_node(
                    f"concept:{ename}",
                    label=ename,
                    type="concept",
                    color="#4ecdc4",
                    size=12,
                    title=f"Concept: {ename}"
                )
            
            # Add edge from paper to concept
            G.add_edge(f"paper:{pid}", f"concept:{ename}", weight=ent.get("weight", 1.0))

        # Add paper nodes
        for p in papers:
            pid = p["id"]
            score = p.get("score") or 70
            tokens = p.get("total_tokens") or 0
            title = p.get("title", "Untitled")

            node_color = "#ff6b6b" if score >= 90 else ("#feca57" if score >= 80 else "#54a0ff")
            node_size = max(18, min(35, int(score / 3)))

            G.add_node(
                f"paper:{pid}",
                label=title[:30] + "..." if len(title) > 30 else title,
                full_title=title,
                type="paper",
                score=score,
                tokens=tokens,
                color=node_color,
                size=node_size,
                title=f"<b>{title}</b><br>Score: {score}/100<br>Tokens: {tokens:,}<br>Source: {p.get('source')}"
            )

        # Add paper-to-paper similarity edges based on shared concepts (Jaccard similarity)
        pids = list(paper_entities.keys())
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                p1, p2 = pids[i], pids[j]
                s1, s2 = set(paper_entities[p1]), set(paper_entities[p2])
                if not s1 or not s2:
                    continue
                intersection = s1.intersection(s2)
                union = s1.union(s2)
                jaccard = len(intersection) / len(union) if union else 0

                if jaccard >= 0.15:  # meaningful overlap
                    G.add_edge(
                        f"paper:{p1}",
                        f"paper:{p2}",
                        weight=round(jaccard * 5, 2),
                        color="#ffffff33",
                        title=f"Shared concepts: {', '.join(list(intersection)[:3])}"
                    )

        return G

    def export_interactive_html(self) -> str:
        """
        Builds the graph and exports a standalone interactive Vis.js HTML visualization.
        """
        G = self.build_graph()

        nodes_data = []
        for node_id, data in G.nodes(data=True):
            nodes_data.append({
                "id": node_id,
                "label": data.get("label", node_id),
                "title": data.get("title", ""),
                "color": data.get("color", "#54a0ff"),
                "size": data.get("size", 15),
                "shape": "box" if data.get("type") == "paper" else "dot",
                "font": {"color": "#ffffff", "face": "system-ui, sans-serif", "size": 13}
            })

        edges_data = []
        for u, v, data in G.edges(data=True):
            edges_data.append({
                "from": u,
                "to": v,
                "value": data.get("weight", 1.0),
                "color": data.get("color", "rgba(255, 255, 255, 0.2)"),
                "title": data.get("title", "")
            })

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantitative Finance - Research Knowledge & Alpha Graph</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #0f111a;
            color: #ffffff;
            height: 100vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}
        header {{
            background: #171926;
            padding: 14px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #23273a;
        }}
        .brand {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .brand h1 {{
            font-size: 1.15rem;
            font-weight: 600;
            background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .legend {{
            display: flex;
            gap: 16px;
            font-size: 0.85rem;
            color: #a0a5ba;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }}
        #network {{
            flex: 1;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at center, #151828 0%, #0c0e17 100%);
        }}
        #info-panel {{
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: rgba(23, 25, 38, 0.85);
            backdrop-filter: blur(10px);
            border: 1px solid #2d334d;
            border-radius: 10px;
            padding: 16px;
            max-width: 320px;
            font-size: 0.85rem;
            color: #d1d5db;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }}
        #info-panel h3 {{
            color: #fff;
            margin-bottom: 6px;
            font-size: 0.95rem;
        }}
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <h1>Quantitative Finance Knowledge Graph</h1>
            <span style="font-size: 0.85rem; color: #717894;">{len(nodes_data)} Nodes | {len(edges_data)} Connections</span>
        </div>
        <div class="legend">
            <div class="legend-item"><span class="dot" style="background:#ff6b6b;"></span> Score &ge; 90</div>
            <div class="legend-item"><span class="dot" style="background:#feca57;"></span> Score 80-89</div>
            <div class="legend-item"><span class="dot" style="background:#54a0ff;"></span> Score &lt; 80</div>
            <div class="legend-item"><span class="dot" style="background:#4ecdc4;"></span> Quant Concept / Keyword</div>
        </div>
    </header>
    <div id="network"></div>
    <div id="info-panel">
        <h3>Interactive Controls</h3>
        <p>Scroll to zoom in/out.</p>
        <p>Click and drag nodes to explore clusters.</p>
        <p>Hover over nodes for evaluation details and token statistics.</p>
    </div>

    <script type="text/javascript">
        const nodes = new vis.DataSet({json.dumps(nodes_data)});
        const edges = new vis.DataSet({json.dumps(edges_data)});

        const container = document.getElementById('network');
        const data = {{ nodes: nodes, edges: edges }};
        const options = {{
            nodes: {{
                shape: 'dot',
                shadow: true
            }},
            edges: {{
                smooth: {{ type: 'continuous' }},
                color: {{ opacity: 0.3 }}
            }},
            physics: {{
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {{
                    gravitationalConstant: -38,
                    centralGravity: 0.01,
                    springLength: 85,
                    springConstant: 0.08
                }},
                maxVelocity: 40,
                minVelocity: 0.1,
                stabilization: {{ iterations: 120 }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100
            }}
        }};

        const network = new vis.Network(container, data, options);
    </script>
</body>
</html>
"""
        with open(self.output_html, "w", encoding="utf-8") as f:
            f.write(html_content)

        return self.output_html
