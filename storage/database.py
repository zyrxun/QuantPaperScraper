"""
Local SQLite database manager for storing papers, GLM evaluations, token stats, and Discord dispatch history.
"""

import sqlite3
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

class Database:
    def __init__(self, db_path: str = "data/papers.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Creates necessary database tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Papers table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                external_id TEXT UNIQUE NOT NULL,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                authors TEXT,
                abstract TEXT NOT NULL,
                category TEXT,
                published_date TEXT,
                pdf_url TEXT,
                local_pdf_path TEXT,
                doi TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Ensure category column exists in existing databases
            cursor.execute("PRAGMA table_info(papers)")
            cols = [col[1] for col in cursor.fetchall()]
            if "category" not in cols:
                cursor.execute("ALTER TABLE papers ADD COLUMN category TEXT")

            # Evaluations table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER NOT NULL,
                model_name TEXT,
                score INTEGER NOT NULL,
                hook TEXT,
                breakthrough_summary TEXT,
                takeaway TEXT,
                concepts TEXT,
                evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (paper_id) REFERENCES papers (id) ON DELETE CASCADE
            )
            """)

            # PDF Tokenization table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS pdf_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER UNIQUE NOT NULL,
                total_tokens INTEGER NOT NULL,
                section_breakdown TEXT,
                extracted_keywords TEXT,
                tokenized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (paper_id) REFERENCES papers (id) ON DELETE CASCADE
            )
            """)

            # Discord dispatch history
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS discord_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER NOT NULL,
                channel_id TEXT,
                message_id TEXT,
                posted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (paper_id) REFERENCES papers (id) ON DELETE CASCADE
            )
            """)

            # Graph entities and relations
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS graph_entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER NOT NULL,
                entity_name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                FOREIGN KEY (paper_id) REFERENCES papers (id) ON DELETE CASCADE
            )
            """)

            conn.commit()

    def paper_exists(self, external_id: str) -> bool:
        """Check if paper with given external ID has already been ingested."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM papers WHERE external_id = ?", (external_id,))
            return cursor.fetchone() is not None

    def save_paper(self, paper: Dict[str, Any]) -> int:
        """
        Saves a paper to the database if it doesn't already exist.
        Returns the paper's internal database ID.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            authors_json = json.dumps(paper.get("authors", []))
            cursor.execute("""
            INSERT INTO papers (external_id, source, title, authors, abstract, category, published_date, pdf_url, local_pdf_path, doi)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(external_id) DO UPDATE SET
                title=excluded.title,
                abstract=excluded.abstract,
                pdf_url=excluded.pdf_url,
                category=COALESCE(excluded.category, papers.category)
            RETURNING id
            """, (
                paper["external_id"],
                paper.get("source", "unknown"),
                paper["title"],
                authors_json,
                paper["abstract"],
                paper.get("category", "Quantitative Finance"),
                paper.get("published_date", ""),
                paper.get("pdf_url", ""),
                paper.get("local_pdf_path", None),
                paper.get("doi", "")
            ))
            row = cursor.fetchone()
            conn.commit()
            return row["id"] if row else -1

    def update_local_pdf_path(self, paper_id: int, local_pdf_path: str):
        """Update the saved local path of the downloaded PDF."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE papers SET local_pdf_path = ? WHERE id = ?", (local_pdf_path, paper_id))
            conn.commit()

    def save_evaluation(self, paper_id: int, eval_data: Dict[str, Any]):
        """Saves evaluation results from the GLM model."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            concepts_json = json.dumps(eval_data.get("concepts", []))
            cursor.execute("""
            INSERT INTO evaluations (paper_id, model_name, score, hook, breakthrough_summary, takeaway, concepts)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                paper_id,
                eval_data.get("model_name", "glm"),
                int(eval_data.get("score", 0)),
                eval_data.get("hook", ""),
                eval_data.get("breakthrough_summary", ""),
                eval_data.get("takeaway", ""),
                concepts_json
            ))

            # Store entities in graph_entities table
            for concept in eval_data.get("concepts", []):
                concept_clean = concept.strip()
                if concept_clean:
                    cursor.execute("""
                    INSERT INTO graph_entities (paper_id, entity_name, entity_type, weight)
                    VALUES (?, ?, 'concept', 1.0)
                    """, (paper_id, concept_clean.lower()))

            conn.commit()

    def save_token_info(self, paper_id: int, total_tokens: int, section_breakdown: Dict[str, int], keywords: List[str]):
        """Saves token counts and key terms derived from full-text PDF tokenization."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO pdf_tokens (paper_id, total_tokens, section_breakdown, extracted_keywords)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                total_tokens=excluded.total_tokens,
                section_breakdown=excluded.section_breakdown,
                extracted_keywords=excluded.extracted_keywords,
                tokenized_at=CURRENT_TIMESTAMP
            """, (
                paper_id,
                total_tokens,
                json.dumps(section_breakdown),
                json.dumps(keywords)
            ))

            for kw in keywords:
                cursor.execute("""
                INSERT INTO graph_entities (paper_id, entity_name, entity_type, weight)
                VALUES (?, ?, 'keyword', 0.8)
                """, (paper_id, kw.strip().lower()))

            conn.commit()

    def get_top_unposted_paper(self, min_score: int = 70) -> Optional[Dict[str, Any]]:
        """Finds the highest-rated paper that hasn't been posted to Discord yet."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT p.*, e.score, e.hook, e.breakthrough_summary, e.takeaway, e.concepts, e.model_name
            FROM papers p
            JOIN evaluations e ON p.id = e.paper_id
            LEFT JOIN discord_history d ON p.id = d.paper_id
            WHERE d.id IS NULL AND e.score >= ?
            ORDER BY e.score DESC, p.id DESC
            LIMIT 1
            """, (min_score,))
            row = cursor.fetchone()
            if not row:
                return None
            
            paper = dict(row)
            paper["authors"] = json.loads(paper["authors"]) if paper["authors"] else []
            paper["concepts"] = json.loads(paper["concepts"]) if paper["concepts"] else []
            return paper

    def mark_as_posted(self, paper_id: int, channel_id: str, message_id: str):
        """Records that a paper was posted to Discord to prevent duplicate sends."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO discord_history (paper_id, channel_id, message_id)
            VALUES (?, ?, ?)
            """, (paper_id, str(channel_id), str(message_id)))
            conn.commit()

    def get_top_papers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieves top-scoring papers from the archive."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT p.*, e.score, e.hook, e.breakthrough_summary, e.takeaway, e.concepts
            FROM papers p
            JOIN evaluations e ON p.id = e.paper_id
            ORDER BY e.score DESC, p.id DESC
            LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["authors"] = json.loads(item["authors"]) if item["authors"] else []
                item["concepts"] = json.loads(item["concepts"]) if item["concepts"] else []
                results.append(item)
            return results

    def get_all_papers_for_graph(self) -> List[Dict[str, Any]]:
        """Retrieves all papers with their associated concepts and evaluation data for graph construction."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT p.id, p.title, p.authors, p.abstract, p.source, p.published_date,
                   p.external_id, p.category, p.pdf_url,
                   e.score, e.hook, e.breakthrough_summary, e.takeaway, e.concepts,
                   pt.total_tokens, pt.section_breakdown, pt.extracted_keywords
            FROM papers p
            LEFT JOIN evaluations e ON p.id = e.paper_id
            LEFT JOIN pdf_tokens pt ON p.id = pt.paper_id
            """)
            papers = []
            for r in cursor.fetchall():
                p = dict(r)
                p["concepts"] = json.loads(p["concepts"]) if p.get("concepts") else []
                p["extracted_keywords"] = json.loads(p["extracted_keywords"]) if p.get("extracted_keywords") else []
                if p.get("authors") and p["authors"].strip().startswith("["):
                    try:
                        p["authors"] = json.loads(p["authors"])
                    except Exception:
                        p["authors"] = [p["authors"]]
                elif p.get("authors"):
                    p["authors"] = [p["authors"]]
                else:
                    p["authors"] = []
                p["section_breakdown"] = json.loads(p["section_breakdown"]) if p.get("section_breakdown") else {}
                papers.append(p)
            return papers

    def get_graph_entities(self) -> List[Dict[str, Any]]:
        """Retrieves entities and weights for building network edges."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT paper_id, entity_name, entity_type, weight
            FROM graph_entities
            """)
            return [dict(r) for r in cursor.fetchall()]

    def reindex_corpus_entities(self, force_all: bool = False) -> int:
        """
        Re-extracts high-quality domain concepts from paper titles, abstracts, and categories,
        replacing legacy identical fallback concepts in graph_entities and evaluations.
        """
        from graph.concept_extractor import extract_domain_concepts
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, title, abstract, category FROM papers")
            papers = cursor.fetchall()
            updated_count = 0

            fallback_set = {
                "stochastic volatility", "market microstructure", "risk-neutral pricing",
                "statistical arbitrage", "optimal execution"
            }

            for row in papers:
                pid, title, abstract, cat = row["id"], row["title"], row["abstract"], row["category"]
                # Check current evaluation concepts
                cursor.execute("SELECT concepts FROM evaluations WHERE paper_id = ?", (pid,))
                eval_row = cursor.fetchone()
                existing_concepts = json.loads(eval_row[0]) if eval_row and eval_row[0] else []

                is_stale_fallback = (set(existing_concepts) == fallback_set)

                if force_all or is_stale_fallback or not existing_concepts:
                    new_concepts = extract_domain_concepts(title, abstract, cat)
                    if new_concepts:
                        # Update evaluation
                        cursor.execute("""
                        UPDATE evaluations SET concepts = ? WHERE paper_id = ?
                        """, (json.dumps(new_concepts), pid))

                        # Delete old concept entities for this paper
                        cursor.execute("""
                        DELETE FROM graph_entities WHERE paper_id = ? AND entity_type = 'concept'
                        """, (pid,))

                        # Insert fresh concept entities
                        for c in new_concepts:
                            cursor.execute("""
                            INSERT INTO graph_entities (paper_id, entity_name, entity_type, weight)
                            VALUES (?, ?, 'concept', 1.0)
                            """, (pid, c.lower().strip()))

                        updated_count += 1

            # Clean up stopword / short noise from PDF keywords in graph_entities
            from graph.concept_extractor import STOPWORDS
            placeholders = ",".join("?" for _ in STOPWORDS)
            cursor.execute(f"""
            DELETE FROM graph_entities 
            WHERE entity_type = 'keyword' AND (entity_name IN ({placeholders}) OR length(entity_name) < 4)
            """, list(STOPWORDS))

            conn.commit()
            return updated_count

    def get_stats(self) -> Dict[str, Any]:
        """Provides summary metrics of the database corpus."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM papers")
            total_papers = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM evaluations")
            evaluated_papers = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*), COALESCE(SUM(total_tokens), 0) FROM pdf_tokens")
            token_stats = cursor.fetchone()
            tokenized_pdfs = token_stats[0]
            total_tokens = token_stats[1]

            cursor.execute("SELECT COUNT(*) FROM discord_history")
            posted_papers = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT entity_name) FROM graph_entities")
            unique_concepts = cursor.fetchone()[0]

            return {
                "total_papers": total_papers,
                "evaluated_papers": evaluated_papers,
                "tokenized_pdfs": tokenized_pdfs,
                "total_tokens": total_tokens,
                "posted_papers": posted_papers,
                "unique_concepts": unique_concepts
            }

    def get_detailed_stats(self) -> Dict[str, Any]:
        """
        Provides comprehensive database metrics for the dashboard,
        including category breakdowns, score distribution, and top concepts.
        """
        stats = self.get_stats()
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Category breakdown
            cursor.execute("""
            SELECT COALESCE(category, 'Quantitative Finance') as cat, COUNT(*) as cnt
            FROM papers
            GROUP BY cat
            ORDER BY cnt DESC
            """)
            stats["categories"] = {row["cat"]: row["cnt"] for row in cursor.fetchall()}

            # Score distribution
            cursor.execute("""
            SELECT 
                COUNT(CASE WHEN score >= 85 THEN 1 END) as elite,
                COUNT(CASE WHEN score >= 70 AND score < 85 THEN 1 END) as notable,
                COUNT(CASE WHEN score < 70 THEN 1 END) as screened,
                ROUND(AVG(score), 1) as avg_score
            FROM evaluations
            """)
            score_row = cursor.fetchone()
            stats["score_distribution"] = {
                "elite": score_row["elite"] if score_row else 0,
                "notable": score_row["notable"] if score_row else 0,
                "screened": score_row["screened"] if score_row else 0,
                "avg_score": score_row["avg_score"] if score_row and score_row["avg_score"] is not None else 0.0
            }

            # Top concepts
            cursor.execute("""
            SELECT entity_name, COUNT(*) as cnt
            FROM graph_entities
            WHERE entity_type = 'concept'
            GROUP BY entity_name
            ORDER BY cnt DESC
            LIMIT 10
            """)
            stats["top_concepts"] = [(row["entity_name"], row["cnt"]) for row in cursor.fetchall()]

            # Sources breakdown
            cursor.execute("""
            SELECT source, COUNT(*) as cnt
            FROM papers
            GROUP BY source
            """)
            stats["sources"] = {row["source"]: row["cnt"] for row in cursor.fetchall()}

            return stats
