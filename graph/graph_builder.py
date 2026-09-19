"""
High-Performance Quantitative Finance Knowledge Graph Generator.
Constructs a semantically optimized research graph using NetworkX,
TF-IDF concept weighting, sparse k-NN cosine similarity, modularity-based
community detection, and an interactive, hardware-accelerated Vis.js visualization
with an embedded Token Search & Footprint Engine.
"""

import os
import json
import math
from collections import Counter
from typing import Dict, Any, List, Set, Tuple, Optional
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from storage.database import Database

# Cohesive cluster color palette for dark-mode visualization
CLUSTER_PALETTE = [
    "#4ecdc4",  # Turquoise / Microstructure
    "#ff6b6b",  # Coral / High Score / Alpha
    "#45aaf2",  # Ocean Blue / Econometrics
    "#a55eea",  # Purple / Machine Learning & Deep RL
    "#fd9644",  # Orange / Volatility & Derivatives
    "#26de81",  # Mint Green / Portfolio Optimization
    "#fed330",  # Amber / Factors & Stat Arb
    "#eb3b5a",  # Crimson / High-Frequency Trading
    "#2bcbba",  # Cyan / Computational Finance
    "#778beb",  # Lavender / Mathematical Finance
    "#f78fb3",  # Pink / Crypto & DeFi
    "#cf6a87",  # Rose / Risk Management
]


class KnowledgeGraphBuilder:
    def __init__(self, db: Database, output_html: str = "data/graph.html"):
        self.db = db
        self.output_html = output_html
        os.makedirs(os.path.dirname(os.path.abspath(self.output_html)), exist_ok=True)

    def build_graph(
        self,
        min_concept_degree: int = 1,
        top_k_similar: int = 3,
        sim_threshold: float = 0.20,
        max_prevalence: float = 0.65
    ) -> nx.Graph:
        """
        Constructs a mathematically principled research graph:
        1. Embeds paper nodes with rich metadata (score, tokens, abstract, takeaway, extracted keywords).
        2. Links papers to extracted domain concepts.
        3. Applies TF-IDF weighting and discounts ubiquitous corpus-wide concepts.
        4. Adds sparse top-K nearest-neighbor similarity edges between papers based
           on cosine similarity of concept TF-IDF vectors, eliminating O(N^2) cliques.
        5. Computes Louvain/modularity community detection to partition papers into
           coherent quantitative finance research clusters.
        """
        G = nx.Graph()
        papers = self.db.get_all_papers_for_graph()
        entities = self.db.get_graph_entities()

        if not papers:
            return G

        total_papers = len(papers)

        # 1. Map paper to its entities and count entity document frequency
        paper_entities: Dict[int, List[str]] = {}
        concept_df: Counter = Counter()

        for ent in entities:
            pid = ent["paper_id"]
            ename = ent["entity_name"].strip().lower()
            if not ename:
                continue
            if pid not in paper_entities:
                paper_entities[pid] = []
            if ename not in paper_entities[pid]:
                paper_entities[pid].append(ename)
                concept_df[ename] += 1

        # Also incorporate concepts stored directly in paper evaluation if not in entities
        for p in papers:
            pid = p["id"]
            eval_concepts = p.get("concepts") or []
            for c in eval_concepts:
                c_clean = c.strip().lower()
                if not c_clean:
                    continue
                if pid not in paper_entities:
                    paper_entities[pid] = []
                if c_clean not in paper_entities[pid]:
                    paper_entities[pid].append(c_clean)
                    concept_df[c_clean] += 1

        # 2. Add paper nodes with rich metadata
        paper_lookup: Dict[int, Dict[str, Any]] = {}
        for p in papers:
            pid = p["id"]
            score = int(p.get("score") or 75)
            tokens = int(p.get("total_tokens") or 0)
            title = p.get("title", f"Paper #{pid}")
            category = p.get("category") or "Quantitative Finance"
            source = p.get("source") or "arXiv"

            paper_lookup[pid] = p

            G.add_node(
                f"paper:{pid}",
                paper_id=pid,
                label=title[:28] + "..." if len(title) > 28 else title,
                full_title=title,
                type="paper",
                score=score,
                tokens=tokens,
                category=category,
                source=source,
                authors=p.get("authors", []),
                published_date=p.get("published_date", ""),
                external_id=p.get("external_id", ""),
                pdf_url=p.get("pdf_url", ""),
                hook=p.get("hook", ""),
                breakthrough_summary=p.get("breakthrough_summary", ""),
                takeaway=p.get("takeaway", ""),
                abstract=p.get("abstract", ""),
                extracted_keywords=p.get("extracted_keywords", []),
                section_breakdown=p.get("section_breakdown", {}),
                concepts=paper_entities.get(pid, []),
                color="#ff6b6b" if score >= 90 else ("#feca57" if score >= 80 else "#54a0ff"),
                size=max(18, min(36, int(score / 2.7))),
                title=f"<b>{title}</b><br>Score: {score}/100<br>Category: {category}"
            )

        # 3. Add concept nodes and paper-concept edges
        for pid, c_list in paper_entities.items():
            if f"paper:{pid}" not in G:
                continue
            for cname in c_list:
                freq = concept_df[cname]
                if freq < min_concept_degree:
                    continue

                node_id = f"concept:{cname}"

                if not G.has_node(node_id):
                    # Size based on corpus frequency
                    c_size = max(10, min(24, 10 + int(freq * 1.5)))
                    G.add_node(
                        node_id,
                        concept_name=cname,
                        label=cname,
                        type="concept",
                        frequency=freq,
                        is_leaf=(freq == 1),
                        color="#2bcbba",
                        size=c_size,
                        title=f"Concept: <b>{cname}</b><br>Referenced by {freq} paper(s)"
                    )

                G.add_edge(
                    f"paper:{pid}",
                    node_id,
                    type="has_concept",
                    weight=1.0,
                    color="rgba(43, 203, 186, 0.25)"
                )

        # 4. Sparse k-NN Paper-to-Paper Similarity via TF-IDF & Cosine Similarity
        # Compute IDF: log((N + 1) / (DF + 1)) + 1
        idf: Dict[str, float] = {}
        for cname, df_val in concept_df.items():
            prevalence = df_val / total_papers
            # Discount ubiquitous concepts appearing in > max_prevalence of papers
            if prevalence > max_prevalence:
                idf[cname] = 0.05
            else:
                idf[cname] = math.log((total_papers + 1.0) / (df_val + 1.0)) + 1.0

        # Construct unit-norm TF-IDF vector for each paper
        paper_vectors: Dict[int, Dict[str, float]] = {}
        for pid, c_list in paper_entities.items():
            vec = {}
            norm_sq = 0.0
            for c in c_list:
                w = idf.get(c, 1.0)
                if w > 0:
                    vec[c] = w
                    norm_sq += w * w
            if norm_sq > 0:
                norm = math.sqrt(norm_sq)
                paper_vectors[pid] = {c: val / norm for c, val in vec.items()}

        # Compute pairwise cosine similarity and extract Top-K neighbors per paper
        pids = list(paper_vectors.keys())
        p_neighbors: Dict[int, List[Tuple[int, float, List[str]]]] = {pid: [] for pid in pids}

        for i in range(len(pids)):
            p1 = pids[i]
            v1 = paper_vectors[p1]
            for j in range(i + 1, len(pids)):
                p2 = pids[j]
                v2 = paper_vectors[p2]

                shared = set(v1.keys()).intersection(v2.keys())
                if not shared:
                    continue

                cosine_sim = sum(v1[c] * v2[c] for c in shared)
                if cosine_sim >= sim_threshold:
                    shared_sorted = sorted(list(shared), key=lambda c: idf.get(c, 0), reverse=True)
                    p_neighbors[p1].append((p2, cosine_sim, shared_sorted))
                    p_neighbors[p2].append((p1, cosine_sim, shared_sorted))

        # Add top-K similarity edges to the graph
        added_edges: Set[Tuple[str, str]] = set()
        for pid, candidates in p_neighbors.items():
            # Sort by cosine similarity descending
            candidates.sort(key=lambda x: x[1], reverse=True)
            for other_pid, sim_score, shared_terms in candidates[:top_k_similar]:
                edge_key = tuple(sorted([f"paper:{pid}", f"paper:{other_pid}"]))
                if edge_key not in added_edges and G.has_node(edge_key[0]) and G.has_node(edge_key[1]):
                    added_edges.add(edge_key)
                    shared_display = ", ".join(shared_terms[:3])
                    G.add_edge(
                        edge_key[0],
                        edge_key[1],
                        type="paper_similarity",
                        weight=round(sim_score * 3.5, 2),
                        similarity=round(sim_score, 3),
                        shared_concepts=shared_terms,
                        color="rgba(119, 139, 235, 0.4)",
                        title=f"Similarity: {int(sim_score * 100)}%<br>Shared: {shared_display}"
                    )

        # 5. Modularity Community / Cluster Detection
        try:
            communities = list(greedy_modularity_communities(G))
            for cluster_idx, comm in enumerate(communities):
                color = CLUSTER_PALETTE[cluster_idx % len(CLUSTER_PALETTE)]
                cluster_concepts = [
                    G.nodes[n].get("concept_name", "")
                    for n in comm if G.nodes[n].get("type") == "concept"
                ]
                top_concept = cluster_concepts[0] if cluster_concepts else f"Theme {cluster_idx + 1}"
                cluster_name = f"Cluster {cluster_idx + 1}: {top_concept.title()}"

                for node_id in comm:
                    G.nodes[node_id]["cluster_id"] = cluster_idx + 1
                    G.nodes[node_id]["cluster_name"] = cluster_name
                    G.nodes[node_id]["cluster_color"] = color
        except Exception:
            for node_id in G.nodes():
                G.nodes[node_id]["cluster_id"] = 1
                G.nodes[node_id]["cluster_name"] = "Quantitative Finance"
                G.nodes[node_id]["cluster_color"] = CLUSTER_PALETTE[0]

        return G

    def export_interactive_html(self, min_concept_degree: int = 1) -> str:
        """
        Builds the optimized knowledge graph and generates a standalone,
        feature-complete Vis.js web application with:
        - Live search & neighbor highlighting
        - Research community filtering
        - Physics auto-freeze
        - Inspector sidebar
        - Full-featured Token Search & Footprint Engine
        """
        G = self.build_graph(min_concept_degree=min_concept_degree)

        nodes_data: List[Dict[str, Any]] = []
        for node_id, data in G.nodes(data=True):
            node_type = data.get("type", "paper")
            cluster_color = data.get("cluster_color", "#54a0ff")
            score = data.get("score", 75)

            if node_type == "paper":
                node_color = {
                    "background": "#ff6b6b" if score >= 90 else ("#feca57" if score >= 80 else "#54a0ff"),
                    "border": cluster_color,
                    "highlight": {"background": "#ffffff", "border": "#ff6b6b"}
                }
                shape = "dot"
            else:
                node_color = {
                    "background": cluster_color,
                    "border": "#ffffff44",
                    "highlight": {"background": "#ffffff", "border": cluster_color}
                }
                shape = "dot"

            nodes_data.append({
                "id": node_id,
                "label": data.get("label", node_id),
                "full_title": data.get("full_title", data.get("concept_name", node_id)),
                "title": data.get("title", ""),
                "type": node_type,
                "score": score,
                "tokens": data.get("tokens", 0),
                "category": data.get("category", ""),
                "source": data.get("source", ""),
                "authors": data.get("authors", []),
                "published_date": data.get("published_date", ""),
                "external_id": data.get("external_id", ""),
                "pdf_url": data.get("pdf_url", ""),
                "hook": data.get("hook", ""),
                "breakthrough_summary": data.get("breakthrough_summary", ""),
                "takeaway": data.get("takeaway", ""),
                "abstract": data.get("abstract", ""),
                "extracted_keywords": data.get("extracted_keywords", []),
                "section_breakdown": data.get("section_breakdown", {}),
                "concepts": data.get("concepts", []),
                "frequency": data.get("frequency", 1),
                "is_leaf": data.get("is_leaf", False),
                "cluster_id": data.get("cluster_id", 1),
                "cluster_name": data.get("cluster_name", "Quantitative Finance"),
                "color": node_color,
                "size": data.get("size", 14),
                "shape": shape,
                "borderWidth": 2,
                "font": {
                    "color": "#e2e8f0",
                    "face": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
                    "size": 11 if node_type == "paper" else 10,
                    "strokeWidth": 2,
                    "strokeColor": "#0b0d17"
                }
            })

        edges_data: List[Dict[str, Any]] = []
        for u, v, data in G.edges(data=True):
            edge_type = data.get("type", "similarity")
            is_similarity = (edge_type == "paper_similarity")
            weight = data.get("weight", 1.0)

            edges_data.append({
                "id": f"{u}__{v}",
                "from": u,
                "to": v,
                "value": weight,
                "type": edge_type,
                "color": {
                    "color": "rgba(254, 202, 87, 0.45)" if is_similarity else "rgba(43, 203, 186, 0.20)",
                    "highlight": "#ffffff",
                    "hover": "#feca57"
                },
                "width": max(1, min(4, int(weight * 1.2))) if is_similarity else 1,
                "dashes": False,
                "title": data.get("title", ""),
                "smooth": {"type": "continuous", "roundness": 0.25}
            })

        # Summary statistics
        paper_count = sum(1 for n in nodes_data if n["type"] == "paper")
        concept_count = sum(1 for n in nodes_data if n["type"] == "concept")
        edge_count = len(edges_data)
        clusters_found = len(set(n["cluster_name"] for n in nodes_data))
        scores = [n["score"] for n in nodes_data if n["type"] == "paper"]
        avg_score = round(sum(scores) / len(scores), 1) if scores else 0
        html_content = self._render_html_template(
            nodes_data=nodes_data,
            edges_data=edges_data,
            paper_count=paper_count,
            concept_count=concept_count,
            edge_count=edge_count,
            clusters_found=clusters_found,
            avg_score=avg_score
        )

        with open(self.output_html, "w", encoding="utf-8") as f:
            f.write(html_content)

        return self.output_html

    def _render_html_template(
        self,
        nodes_data: List[Dict[str, Any]],
        edges_data: List[Dict[str, Any]],
        paper_count: int,
        concept_count: int,
        edge_count: int,
        clusters_found: int,
        avg_score: float
    ) -> str:
        nodes_json = json.dumps(nodes_data)
        edges_json = json.dumps(edges_data)

        # Unique clusters for filter dropdown
        unique_clusters = sorted(list(set(n["cluster_name"] for n in nodes_data)))
        cluster_options = "\n".join(
            f'<option value="{c}">{c}</option>' for c in unique_clusters
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantitative Finance Research Knowledge & Alpha Graph</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0a0c14;
            --bg-surface: rgba(18, 21, 35, 0.85);
            --bg-surface-elevated: #161a2b;
            --border-color: #232840;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-cyan: #2bcbba;
            --accent-gold: #feca57;
            --accent-red: #ff6b6b;
            --accent-blue: #54a0ff;
            --accent-purple: #a55eea;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            height: 100vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}

        /* Header & Control Bar */
        header {{
            background: rgba(15, 18, 29, 0.95);
            backdrop-filter: blur(12px);
            padding: 10px 20px;
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            z-index: 20;
            gap: 12px;
        }}
        .brand-group {{
            display: flex;
            align-items: center;
            gap: 14px;
        }}
        .brand-title {{
            font-size: 1.05rem;
            font-weight: 700;
            background: linear-gradient(135deg, #ff6b6b 0%, #feca57 50%, #4ecdc4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.01em;
        }}
        .stats-badges {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        .badge {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-secondary);
        }}
        .badge strong {{
            color: #fff;
        }}
        .badge-gold strong {{
            color: var(--accent-gold);
        }}

        /* Toolbar Controls */
        .controls-group {{
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .search-box {{
            position: relative;
        }}
        .search-box input {{
            background: var(--bg-surface-elevated);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 6px 12px 6px 28px;
            font-size: 0.8rem;
            color: #fff;
            width: 190px;
            outline: none;
            transition: all 0.2s ease;
        }}
        .search-box input:focus {{
            border-color: var(--accent-cyan);
            width: 240px;
            box-shadow: 0 0 0 2px rgba(43, 203, 186, 0.2);
        }}
        .search-icon {{
            position: absolute;
            left: 9px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 0.75rem;
            color: var(--text-muted);
        }}
        .filter-select {{
            background: var(--bg-surface-elevated);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            font-size: 0.8rem;
            padding: 6px 10px;
            border-radius: 6px;
            outline: none;
            cursor: pointer;
        }}
        .filter-select:focus {{
            border-color: var(--accent-cyan);
        }}
        .btn {{
            background: var(--bg-surface-elevated);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            font-size: 0.8rem;
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-weight: 500;
            transition: all 0.15s ease;
            white-space: nowrap;
        }}
        .btn:hover {{
            background: #232840;
            border-color: #3b4263;
        }}
        .btn-active {{
            background: rgba(43, 203, 186, 0.15);
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
        }}
        .btn-token {{
            background: linear-gradient(135deg, rgba(254, 202, 87, 0.18) 0%, rgba(255, 107, 107, 0.18) 100%);
            border: 1px solid rgba(254, 202, 87, 0.45);
            color: #feca57;
            font-weight: 600;
        }}
        .btn-token:hover {{
            background: linear-gradient(135deg, rgba(254, 202, 87, 0.35) 0%, rgba(255, 107, 107, 0.35) 100%);
            border-color: #feca57;
            color: #fff;
            box-shadow: 0 0 14px rgba(254, 202, 87, 0.35);
        }}

        /* Canvas Area */
        #workspace {{
            position: relative;
            flex: 1;
            width: 100%;
            height: 100%;
            overflow: hidden;
        }}
        #network {{
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at center, #121528 0%, #080911 100%);
        }}

        /* Floating HUD & Legend */
        .hud-legend {{
            position: absolute;
            top: 16px;
            left: 16px;
            background: var(--bg-surface);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 12px 16px;
            font-size: 0.78rem;
            color: var(--text-secondary);
            z-index: 10;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            display: flex;
            flex-direction: column;
            gap: 8px;
            pointer-events: auto;
        }}
        .legend-row {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-indicator {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            flex-shrink: 0;
        }}
        .legend-line {{
            width: 16px;
            height: 2px;
            background: var(--accent-gold);
            border-radius: 2px;
        }}

        /* Slide-out Inspector Sidebar */
        #inspector {{
            position: absolute;
            top: 16px;
            right: 16px;
            bottom: 16px;
            width: 380px;
            background: var(--bg-surface);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.6);
            display: flex;
            flex-direction: column;
            z-index: 15;
            transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.2s ease;
            transform: translateX(410px);
            opacity: 0;
            pointer-events: none;
            overflow: hidden;
        }}
        #inspector.open {{
            transform: translateX(0);
            opacity: 1;
            pointer-events: auto;
        }}
        .inspector-header {{
            padding: 16px 18px 12px 18px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(22, 26, 43, 0.6);
        }}
        .node-type-badge {{
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            padding: 3px 8px;
            border-radius: 4px;
            background: rgba(43, 203, 186, 0.15);
            color: var(--accent-cyan);
            border: 1px solid rgba(43, 203, 186, 0.3);
        }}
        .close-btn {{
            background: transparent;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            font-size: 1.3rem;
            line-height: 1;
            padding: 4px;
            border-radius: 4px;
        }}
        .close-btn:hover {{
            color: #fff;
            background: rgba(255,255,255,0.08);
        }}
        .inspector-body {{
            padding: 18px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 16px;
            font-size: 0.85rem;
            color: #cbd5e1;
        }}
        .inspector-body h2 {{
            font-size: 1.05rem;
            font-weight: 600;
            color: #ffffff;
            line-height: 1.4;
        }}
        .inspector-meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            font-size: 0.75rem;
            color: var(--text-muted);
        }}
        .card-section {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .card-section h4 {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
        }}
        .hook-box {{
            border-left: 3px solid var(--accent-gold);
            padding-left: 10px;
            font-style: italic;
            color: #e2e8f0;
            line-height: 1.45;
        }}
        .pill-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }}
        .concept-pill {{
            background: rgba(43, 203, 186, 0.1);
            border: 1px solid rgba(43, 203, 186, 0.3);
            color: #5eead4;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.72rem;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        .concept-pill:hover {{
            background: rgba(43, 203, 186, 0.25);
            border-color: #5eead4;
            transform: translateY(-1px);
        }}
        .keyword-pill {{
            background: rgba(84, 160, 255, 0.1);
            border: 1px solid rgba(84, 160, 255, 0.25);
            color: #93c5fd;
            padding: 2px 7px;
            border-radius: 10px;
            font-size: 0.7rem;
            cursor: pointer;
        }}
        .keyword-pill:hover {{
            background: rgba(84, 160, 255, 0.25);
            color: #fff;
        }}
        .btn-link {{
            background: rgba(84, 160, 255, 0.15);
            border: 1px solid rgba(84, 160, 255, 0.3);
            color: #93c5fd;
            padding: 6px 12px;
            border-radius: 6px;
            text-decoration: none;
            font-size: 0.78rem;
            font-weight: 500;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            width: fit-content;
        }}
        .btn-link:hover {{
            background: rgba(84, 160, 255, 0.3);
            color: #fff;
        }}

        /* Token Search Engine Modal */
        .modal-overlay {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(8, 10, 18, 0.75);
            backdrop-filter: blur(8px);
            z-index: 50;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.2s ease;
        }}
        .modal-overlay.open {{
            opacity: 1;
            pointer-events: auto;
        }}
        .token-modal {{
            background: #111424;
            border: 1px solid #2d334d;
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(254, 202, 87, 0.15);
            border-radius: 14px;
            width: 92%;
            max-width: 820px;
            max-height: 85vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            transform: scale(0.96);
            transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        }}
        .modal-overlay.open .token-modal {{
            transform: scale(1);
        }}
        .token-modal-header {{
            padding: 16px 20px;
            background: rgba(22, 26, 43, 0.8);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .token-modal-controls {{
            padding: 16px 20px;
            background: rgba(15, 18, 30, 0.6);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        .token-search-input {{
            background: #171b2d;
            border: 1px solid #2e3552;
            border-radius: 8px;
            padding: 10px 16px 10px 38px;
            font-size: 0.95rem;
            color: #fff;
            width: 100%;
            outline: none;
            transition: all 0.2s ease;
        }}
        .token-search-input:focus {{
            border-color: var(--accent-gold);
            box-shadow: 0 0 0 2px rgba(254, 202, 87, 0.2);
        }}
        .token-filters-row {{
            display: flex;
            align-items: center;
            gap: 14px;
            flex-wrap: wrap;
        }}
        .filter-group {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.8rem;
            color: var(--text-secondary);
        .token-results-header {{
            padding: 10px 20px;
            background: rgba(13, 16, 26, 0.9);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            font-size: 0.78rem;
            color: var(--text-secondary);
            font-family: 'JetBrains Mono', monospace;
        }}
        .token-results-container {{
            padding: 12px 20px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 10px;
            flex: 1;
        }}
        .token-paper-card {{
            background: rgba(23, 27, 45, 0.6);
            border: 1px solid #282f48;
            border-radius: 8px;
            padding: 12px 14px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            transition: all 0.15s ease;
        }}
        .token-paper-card:hover {{
            background: rgba(28, 33, 56, 0.9);
            border-color: #3f4a73;
            transform: translateY(-1px);
        }}
        .token-card-top {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 12px;
        }}
        .token-card-title {{
            font-size: 0.88rem;
            font-weight: 600;
            color: #fff;
            line-height: 1.35;
        }}

        /* Bottom Status bar */
        footer {{
            background: #0d101d;
            border-top: 1px solid var(--border-color);
            padding: 6px 20px;
            display: flex;
            justify-content: space-between;
            font-size: 0.72rem;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
            z-index: 10;
        }}
    </style>
</head>
<body>
    <header>
        <div class="brand-group">
            <h1 class="brand-title">Quantitative Finance Knowledge Graph</h1>
            <div class="stats-badges">
                <span class="badge">Papers: <strong>{paper_count}</strong></span>
                <span class="badge">Concepts: <strong>{concept_count}</strong></span>
                <span class="badge">Connections: <strong>{edge_count}</strong></span>
                <span class="badge">Clusters: <strong>{clusters_found}</strong></span>
            </div>
        </div>

        <div class="controls-group">
            <button id="token-engine-btn" class="btn btn-token" onclick="toggleTokenEngine()">⚡ Search Engine</button>

            <div class="search-box">
                <span class="search-icon">🔍</span>
                <input type="text" id="search-input" placeholder="Search papers or concepts...">
            </div>

            <select id="cluster-filter" class="filter-select" title="Filter by Research Cluster">
                <option value="all">All Research Clusters</option>
                {cluster_options}
            </select>

            <button id="leaf-filter-btn" class="btn" onclick="toggleLeafNodes()">Core Hubs Only</button>
            <button id="sim-edges-btn" class="btn btn-active" onclick="toggleSimEdges()">Similarity Links</button>
            <button id="physics-btn" class="btn" onclick="togglePhysics()">Physics: AUTO</button>
            <button class="btn" onclick="resetView()">Reset View</button>
        </div>
    </header>

    <div id="workspace">
        <div class="hud-legend">
            <div class="legend-row">
                <span class="legend-indicator" style="background:#ff6b6b;"></span>
                <span>Score &ge; 90 (Alpha Elite)</span>
            </div>
            <div class="legend-row">
                <span class="legend-indicator" style="background:#feca57;"></span>
                <span>Score 80-89 (Strong)</span>
            </div>
            <div class="legend-row">
                <span class="legend-indicator" style="background:#54a0ff;"></span>
                <span>Score &lt; 80 (Standard)</span>
            </div>
            <div class="legend-row">
                <span class="legend-indicator" style="background:#2bcbba;"></span>
                <span>Quant Concept Node</span>
            </div>
            <div class="legend-row">
                <span class="legend-line"></span>
                <span>TF-IDF Semantic Similarity</span>
            </div>
        </div>

        <div id="network"></div>

        <div id="inspector">
            <div class="inspector-header">
                <span id="insp-badge" class="node-type-badge">RESEARCH PAPER</span>
                <button class="close-btn" onclick="closeInspector()">&times;</button>
            </div>
            <div class="inspector-body" id="insp-content">
                <!-- Dynamically populated -->
            </div>
        </div>
    </div>

    <!-- Research & Keyword Search Engine Modal -->
    <div id="token-engine-overlay" class="modal-overlay" onclick="closeTokenEngineOnBackdrop(event)">
        <div class="token-modal">
            <div class="token-modal-header">
                <div style="display:flex; align-items:center; gap:10px;">
                    <span style="font-size: 1.4rem;">⚡</span>
                    <div>
                        <h2 style="font-size: 1.05rem; font-weight:700; color:#fff;">Research & Keyword Search Engine</h2>
                        <div style="font-size: 0.75rem; color: var(--text-muted);">
                            Search across <strong>{paper_count} papers</strong> &bull; Titles, abstracts, concepts & keywords
                        </div>
                    </div>
                </div>
                <button class="close-btn" onclick="toggleTokenEngine()">&times;</button>
            </div>

            <div class="token-modal-controls">
                <div style="position:relative; width:100%;">
                    <span class="search-icon" style="left:12px; font-size:0.9rem;">🔎</span>
                    <input type="text" id="token-query-input" class="token-search-input" placeholder="Search keywords, methods, models, or topics (e.g. 'limit order', 'stochastic volatility', 'reinforcement learning')...">
                </div>

                <div class="token-filters-row">
                    <div class="filter-group">
                        <label>Sort By:</label>
                        <select id="token-sort-select" class="filter-select" onchange="applyTokenSearch()">
                            <option value="score_desc">Highest Alpha Score</option>
                            <option value="relevance">Search Relevance</option>
                            <option value="title_asc">Title (A-Z)</option>
                        </select>
                    </div>

                    <div class="filter-group" style="margin-left:auto; display:flex; gap:8px;">
                        <button class="btn btn-active" onclick="filterNetworkToTokenMatches()">Filter Canvas to Matches</button>
                        <button class="btn" onclick="clearTokenSearch()">Reset</button>
                    </div>
                </div>
            </div>

            <div class="token-results-header">
                <span id="token-results-count">Showing 0 matching papers</span>
            </div>

            <div class="token-results-container" id="token-results-list">
                <!-- Dynamically populated cards -->
            </div>
        </div>
    </div>

    <footer>
        <span>Press <kbd style="background:#1e2438; padding:1px 4px; border-radius:3px; color:#fff;">Ctrl+K</kbd> or <kbd style="background:#1e2438; padding:1px 4px; border-radius:3px; color:#fff;">/</kbd> for Search Engine | Click node to inspect</span>
        <span id="status-text">Physics stabilized. Graph optimized.</span>
    </footer>

    <script type="text/javascript">
        const rawNodes = {nodes_json};
        const rawEdges = {edges_json};

        const nodes = new vis.DataSet(rawNodes);
        const edges = new vis.DataSet(rawEdges);

        const container = document.getElementById('network');
        const data = {{ nodes: nodes, edges: edges }};

        let physicsEnabled = true;

        const options = {{
            nodes: {{
                shape: 'dot',
                shadow: {{ enabled: true, color: 'rgba(0,0,0,0.5)', size: 8 }}
            }},
            edges: {{
                smooth: {{ type: 'continuous', roundness: 0.25 }},
                selectionWidth: 2
            }},
            physics: {{
                solver: 'barnesHut',
                barnesHut: {{
                    gravitationalConstant: -2600,
                    centralGravity: 0.22,
                    springLength: 90,
                    springConstant: 0.04,
                    damping: 0.09,
                    avoidOverlap: 0.25
                }},
                stabilization: {{
                    enabled: true,
                    iterations: 140,
                    updateInterval: 25
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                hideEdgesOnDrag: true,
                hideEdgesOnZoom: true
            }}
        }};

        const network = new vis.Network(container, data, options);

        network.on("stabilizationIterationsDone", function () {{
            network.setOptions({{ physics: false }});
            physicsEnabled = false;
            const btn = document.getElementById('physics-btn');
            if (btn) {{
                btn.innerText = "Physics: OFF";
                btn.classList.remove("btn-active");
            }}
            document.getElementById('status-text').innerText = "Physics frozen (0% CPU). Ready for exploration.";
        }});

        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            network.setOptions({{ physics: physicsEnabled }});
            const btn = document.getElementById('physics-btn');
            if (physicsEnabled) {{
                btn.innerText = "Physics: ON";
                btn.classList.add("btn-active");
                document.getElementById('status-text').innerText = "Physics simulation running...";
            }} else {{
                btn.innerText = "Physics: OFF";
                btn.classList.remove("btn-active");
                document.getElementById('status-text').innerText = "Physics simulation frozen.";
            }}
        }}

        // 1-Hop Neighbor Highlighting and Dimming
        let highlightedNodeId = null;

        network.on("click", function (params) {{
            if (params.nodes.length > 0) {{
                const selectedId = params.nodes[0];
                highlightNeighborhood(selectedId);
                showInspector(selectedId);
            }} else {{
                resetHighlight();
                closeInspector();
            }}
        }});

        function highlightNeighborhood(selectedId) {{
            highlightedNodeId = selectedId;
            const connectedNodes = new Set(network.getConnectedNodes(selectedId));
            connectedNodes.add(selectedId);
            const connectedEdges = new Set(network.getConnectedEdges(selectedId));

            const updatedNodes = [];
            rawNodes.forEach(node => {{
                if (connectedNodes.has(node.id)) {{
                    updatedNodes.push({{
                        id: node.id,
                        opacity: 1.0,
                        font: {{ color: '#ffffff' }}
                    }});
                }} else {{
                    updatedNodes.push({{
                        id: node.id,
                        opacity: 0.12,
                        font: {{ color: 'rgba(255,255,255,0.1)' }}
                    }});
                }}
            }});

            const updatedEdges = [];
            rawEdges.forEach(edge => {{
                if (connectedEdges.has(edge.id)) {{
                    updatedEdges.push({{
                        id: edge.id,
                        opacity: 1.0,
                        color: {{ opacity: 0.9 }}
                    }});
                }} else {{
                    updatedEdges.push({{
                        id: edge.id,
                        opacity: 0.04,
                        color: {{ opacity: 0.04 }}
                    }});
                }}
            }});

            nodes.update(updatedNodes);
            edges.update(updatedEdges);
        }}

        function resetHighlight() {{
            highlightedNodeId = null;
            const updatedNodes = rawNodes.map(node => ({{
                id: node.id,
                opacity: 1.0,
                font: node.font
            }}));
            const updatedEdges = rawEdges.map(edge => ({{
                id: edge.id,
                opacity: 1.0,
                color: edge.color
            }}));
            nodes.update(updatedNodes);
            edges.update(updatedEdges);
        }}

        function resetView() {{
            resetHighlight();
            closeInspector();
            document.getElementById('search-input').value = "";
            document.getElementById('cluster-filter').value = "all";
            hideLeafs = false;
            showSimilarity = true;
            document.getElementById('leaf-filter-btn').innerText = "Core Hubs Only";
            document.getElementById('leaf-filter-btn').classList.remove("btn-active");
            document.getElementById('sim-edges-btn').classList.add("btn-active");
            applyActiveFilters();
            network.fit({{ animation: {{ duration: 400, easingFunction: 'easeInOutQuad' }} }});
        }}

        // Slide-out Inspector Panel logic
        function showInspector(nodeId) {{
            const node = rawNodes.find(n => n.id === nodeId);
            if (!node) return;

            const inspector = document.getElementById('inspector');
            const badge = document.getElementById('insp-badge');
            const content = document.getElementById('insp-content');

            if (node.type === "paper") {{
                badge.innerText = "RESEARCH PAPER";
                badge.style.background = "rgba(255, 107, 107, 0.15)";
                badge.style.borderColor = "rgba(255, 107, 107, 0.4)";
                badge.style.color = "#ff6b6b";

                const authorsStr = (node.authors && node.authors.length) ? node.authors.slice(0, 4).join(", ") : "Quantitative Research Group";
                const conceptsPills = (node.concepts || []).map(c => 
                    `<span class="concept-pill" onclick="focusConcept('${{c}}')">${{c}}</span>`
                ).join("");

                const keywordsPills = (node.extracted_keywords || []).map(k =>
                    `<span class="keyword-pill" onclick="searchFromInspector('${{k}}')">${{k}}</span>`
                ).join("");

                content.innerHTML = `
                    <h2>${{node.full_title}}</h2>
                    <div class="inspector-meta">
                        <span>Score: <b>${{node.score}}/100</b></span> &bull;
                        <span>Category: <b>${{node.category}}</b></span> &bull;
                        <span>Source: <b>${{node.source.toUpperCase()}}</b></span>
                    </div>
                    <div class="inspector-meta">
                        <span>Authors: ${{authorsStr}}</span>
                    </div>

                    ${{node.hook ? `
                    <div class="card-section">
                        <h4>Executive Hook</h4>
                        <div class="hook-box">${{node.hook}}</div>
                    </div>` : ''}}

                    ${{node.breakthrough_summary ? `
                    <div class="card-section">
                        <h4>Mathematical Breakthrough</h4>
                        <p style="line-height: 1.45;">${{node.breakthrough_summary}}</p>
                    </div>` : ''}}

                    ${{node.takeaway ? `
                    <div class="card-section">
                        <h4>Practical Alpha & Takeaway</h4>
                        <p style="line-height: 1.45; color: #feca57;">${{node.takeaway}}</p>
                    </div>` : ''}}

                    ${{keywordsPills ? `
                    <div class="card-section">
                        <h4>Extracted Keywords</h4>
                        <div class="pill-container">${{keywordsPills}}</div>
                    </div>` : ''}}

                    <div class="card-section">
                        <h4>Core Quant Concepts</h4>
                        <div class="pill-container">${{conceptsPills || '<em>None tagged</em>'}}</div>
                    </div>

                    ${{node.pdf_url ? `
                    <div style="margin-top: 4px;">
                        <a href="${{node.pdf_url}}" target="_blank" class="btn-link">📄 Open Original Paper / PDF</a>
                    </div>` : ''}}
                `;
            }} else {{
                badge.innerText = "QUANT CONCEPT";
                badge.style.background = "rgba(43, 203, 186, 0.15)";
                badge.style.borderColor = "rgba(43, 203, 186, 0.4)";
                badge.style.color = "#2bcbba";

                const connectedPapers = network.getConnectedNodes(nodeId)
                    .map(nid => rawNodes.find(n => n.id === nid))
                    .filter(n => n && n.type === "paper");

                const papersListHtml = connectedPapers.map(p => `
                    <div style="padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <a href="javascript:void(0)" onclick="focusNode('${{p.id}}')" style="color:#93c5fd; text-decoration:none; font-weight:500;">
                            ${{p.full_title}}
                        </a>
                        <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 2px;">
                            Score: ${{p.score}}/100 &bull; ${{p.category}}
                        </div>
                    </div>
                `).join("");

                content.innerHTML = `
                    <h2>${{node.full_title.toUpperCase()}}</h2>
                    <div class="inspector-meta">
                        <span>Corpus Prevalence: <b>${{node.frequency}} papers</b></span> &bull;
                        <span>Research Cluster: <b>${{node.cluster_name}}</b></span>
                    </div>

                    <div class="card-section">
                        <h4>Connected Research Papers (${{connectedPapers.length}})</h4>
                        <div style="max-height: 280px; overflow-y: auto;">
                            ${{papersListHtml || '<em>No direct paper links.</em>'}}
                        </div>
                    </div>
                `;
            }}

            inspector.classList.add("open");
        }}

        function closeInspector() {{
            document.getElementById('inspector').classList.remove("open");
        }}

        function focusNode(nodeId) {{
            network.focus(nodeId, {{
                scale: 1.2,
                animation: {{ duration: 500, easingFunction: 'easeInOutQuad' }}
            }});
            network.selectNodes([nodeId]);
            highlightNeighborhood(nodeId);
            showInspector(nodeId);
        }}

        function focusConcept(conceptName) {{
            const conceptId = `concept:${{conceptName.toLowerCase().trim()}}`;
            const node = rawNodes.find(n => n.id === conceptId);
            if (node) {{
                focusNode(node.id);
            }}
        }}

        // Core Hubs & Similarity Edge Toggles
        let hideLeafs = false;
        let showSimilarity = true;

        function toggleLeafNodes() {{
            hideLeafs = !hideLeafs;
            const btn = document.getElementById('leaf-filter-btn');
            if (hideLeafs) {{
                btn.innerText = "Show All Concepts";
                btn.classList.add("btn-active");
            }} else {{
                btn.innerText = "Core Hubs Only";
                btn.classList.remove("btn-active");
            }}
            applyActiveFilters();
        }}

        function toggleSimEdges() {{
            showSimilarity = !showSimilarity;
            const btn = document.getElementById('sim-edges-btn');
            if (showSimilarity) {{
                btn.classList.add("btn-active");
            }} else {{
                btn.classList.remove("btn-active");
            }}
            applyActiveFilters();
        }}

        function applyActiveFilters() {{
            const selectedCluster = document.getElementById('cluster-filter').value;
            
            const updatedNodes = rawNodes.map(n => {{
                let visible = true;
                if (hideLeafs && n.is_leaf) visible = false;
                if (selectedCluster !== "all" && n.cluster_name !== selectedCluster) visible = false;
                
                return {{
                    id: n.id,
                    hidden: !visible,
                    opacity: 1.0
                }};
            }});

            const visibleNodeIds = new Set(updatedNodes.filter(n => !n.hidden).map(n => n.id));

            const updatedEdges = rawEdges.map(e => {{
                let visible = visibleNodeIds.has(e.from) && visibleNodeIds.has(e.to);
                if (!showSimilarity && e.type === "paper_similarity") visible = false;

                return {{
                    id: e.id,
                    hidden: !visible,
                    opacity: 1.0
                }};
            }});

            nodes.update(updatedNodes);
            edges.update(updatedEdges);
        }}

        // Quick Top-bar Search
        const searchInput = document.getElementById('search-input');
        searchInput.addEventListener('input', function(e) {{
            const query = e.target.value.toLowerCase().trim();
            if (!query) {{
                resetHighlight();
                return;
            }}

            const match = rawNodes.find(n => 
                n.full_title.toLowerCase().includes(query) || 
                n.label.toLowerCase().includes(query)
            );

            if (match) {{
                network.focus(match.id, {{
                    scale: 1.1,
                    animation: {{ duration: 400, easingFunction: 'easeInOutQuad' }}
                }});
                network.selectNodes([match.id]);
                highlightNeighborhood(match.id);
                showInspector(match.id);
            }}
        }});

        document.getElementById('cluster-filter').addEventListener('change', function(e) {{
            applyActiveFilters();
        }});

        // ==========================================
        // ⚡ RESEARCH & KEYWORD SEARCH ENGINE
        // ==========================================
        const tokenEngineOverlay = document.getElementById('token-engine-overlay');
        const tokenQueryInput = document.getElementById('token-query-input');
        let matchedTokenPaperIds = new Set();

        function toggleTokenEngine() {{
            const isOpen = tokenEngineOverlay.classList.contains('open');
            if (isOpen) {{
                tokenEngineOverlay.classList.remove('open');
            }} else {{
                tokenEngineOverlay.classList.add('open');
                tokenQueryInput.focus();
                applyTokenSearch();
            }}
        }}

        function closeTokenEngineOnBackdrop(e) {{
            if (e.target === tokenEngineOverlay) {{
                toggleTokenEngine();
            }}
        }}

        function searchFromInspector(keyword) {{
            if (!tokenEngineOverlay.classList.contains('open')) {{
                toggleTokenEngine();
            }}
            tokenQueryInput.value = keyword;
            applyTokenSearch();
        }}

        function clearTokenSearch() {{
            tokenQueryInput.value = "";
            document.getElementById('token-sort-select').value = "score_desc";
            applyTokenSearch();
        }}

        function applyTokenSearch() {{
            const query = tokenQueryInput.value.toLowerCase().trim();
            const sortBy = document.getElementById('token-sort-select').value;

            const paperNodes = rawNodes.filter(n => n.type === "paper");

            let matches = paperNodes.filter(p => {{
                if (!query) return true;

                // Match query in full title, concepts, extracted_keywords, abstract, hook, takeaway
                const titleMatch = p.full_title.toLowerCase().includes(query);
                const conceptMatch = (p.concepts || []).some(c => c.toLowerCase().includes(query));
                const keywordMatch = (p.extracted_keywords || []).some(k => k.toLowerCase().includes(query));
                const abstractMatch = (p.abstract || "").toLowerCase().includes(query);
                const hookMatch = (p.hook || "").toLowerCase().includes(query);

                return titleMatch || conceptMatch || keywordMatch || abstractMatch || hookMatch;
            }});

            // Sorting
            if (sortBy === "score_desc") {{
                matches.sort((a, b) => (b.score || 0) - (a.score || 0));
            }} else if (sortBy === "title_asc") {{
                matches.sort((a, b) => a.full_title.localeCompare(b.full_title));
            }} else if (sortBy === "relevance" && query) {{
                matches.sort((a, b) => {{
                    const scoreA = (a.full_title.toLowerCase().includes(query) ? 4 : 0) +
                                   ((a.concepts || []).some(c => c.toLowerCase().includes(query)) ? 2 : 0);
                    const scoreB = (b.full_title.toLowerCase().includes(query) ? 4 : 0) +
                                   ((b.concepts || []).some(c => c.toLowerCase().includes(query)) ? 2 : 0);
                    return scoreB - scoreA;
                }});
            }}

            matchedTokenPaperIds = new Set(matches.map(m => m.id));

            // Update stats
            document.getElementById('token-results-count').innerText = `Showing ${{matches.length}} matching papers`;

            // Render cards
            const listEl = document.getElementById('token-results-list');
            if (!matches.length) {{
                listEl.innerHTML = `<div style="text-align:center; padding:30px; color:var(--text-muted);">No papers match this search query.</div>`;
                return;
            }}

            listEl.innerHTML = matches.map(p => {{
                const scoreColor = p.score >= 90 ? '#ff6b6b' : (p.score >= 80 ? '#feca57' : '#54a0ff');
                const kwPills = (p.extracted_keywords || []).slice(0, 5).map(k =>
                    `<span class="keyword-pill" onclick="filterTokenQuery('${{k}}')">${{k}}</span>`
                ).join(" ");

                const conceptPills = (p.concepts || []).slice(0, 4).map(c =>
                    `<span class="concept-pill" onclick="filterTokenQuery('${{c}}')">${{c}}</span>`
                ).join(" ");

                return `
                    <div class="token-paper-card">
                        <div class="token-card-top">
                            <div class="token-card-title">${{p.full_title}}</div>
                            <span class="badge" style="background:${{scoreColor}}22; border-color:${{scoreColor}}; color:${{scoreColor}};">
                                ${{p.score}}/100
                            </span>
                        </div>

                        <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.75rem; color:var(--text-secondary);">
                            <span>Category: <b>${{p.category}}</b></span>
                            <span>Source: <b>${{p.source.toUpperCase()}}</b></span>
                        </div>

                        ${{p.hook ? `<div style="font-size:0.78rem; color:#cbd5e1; font-style:italic; border-left:2px solid var(--accent-gold); padding-left:8px; margin:2px 0;">${{p.hook}}</div>` : ''}}

                        ${{conceptPills ? `<div class="pill-container" style="margin-top:2px;">${{conceptPills}}</div>` : ''}}
                        ${{kwPills ? `<div class="pill-container">${{kwPills}}</div>` : ''}}

                        <div style="display:flex; justify-content:flex-end; gap:8px; margin-top:4px;">
                            <button class="btn" style="font-size:0.72rem; padding:4px 10px;" onclick="locatePaperFromEngine('${{p.id}}')">🎯 Locate in Graph</button>
                        </div>
                    </div>
                `;
            }}).join("");
        }}

        function filterTokenQuery(val) {{
            tokenQueryInput.value = val;
            applyTokenSearch();
        }}

        function locatePaperFromEngine(paperId) {{
            toggleTokenEngine();
            focusNode(paperId);
        }}

        function filterNetworkToTokenMatches() {{
            if (!matchedTokenPaperIds.size) return;
            toggleTokenEngine();

            const updatedNodes = rawNodes.map(n => {{
                let visible = false;
                if (n.type === "paper" && matchedTokenPaperIds.has(n.id)) visible = true;
                if (n.type === "concept") {{
                    // Show concept if connected to at least one matched paper
                    const connected = network.getConnectedNodes(n.id);
                    visible = connected.some(id => matchedTokenPaperIds.has(id));
                }}

                return {{
                    id: n.id,
                    opacity: visible ? 1.0 : 0.06,
                    font: {{ color: visible ? '#ffffff' : 'rgba(255,255,255,0.03)' }}
                }};
            }});

            const updatedEdges = rawEdges.map(e => {{
                const fromMatch = matchedTokenPaperIds.has(e.from);
                const toMatch = matchedTokenPaperIds.has(e.to);
                const visible = fromMatch || toMatch;

                return {{
                    id: e.id,
                    opacity: visible ? 0.8 : 0.02
                }};
            }});

            nodes.update(updatedNodes);
            edges.update(updatedEdges);
            document.getElementById('status-text').innerText = `Filtered canvas to ${{matchedTokenPaperIds.size}} search matches.`;
        }}

        // Keyboard Shortcuts: Ctrl+K or / opens Search Engine
        window.addEventListener('keydown', function(e) {{
            if ((e.ctrlKey && e.key === 'k') || (e.key === '/' && document.activeElement.tagName !== 'INPUT')) {{
                e.preventDefault();
                toggleTokenEngine();
            }} else if (e.key === 'Escape') {{
                if (tokenEngineOverlay.classList.contains('open')) {{
                    toggleTokenEngine();
                }} else {{
                    resetHighlight();
                    closeInspector();
                }}
            }}
        }});

        tokenQueryInput.addEventListener('input', applyTokenSearch);
    </script>
</body>
</html>
"""
