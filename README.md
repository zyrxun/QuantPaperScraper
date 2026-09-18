<div align="center">

# 📈 QuantPaperBot
### Autonomous Quantitative Finance Paper Scraper, GLM Evaluator & Knowledge Engine

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Discord](https://img.shields.io/badge/Discord-discord.py%20v2.3%2B-5865F2.svg)](https://discord.com)
[![LLM](https://img.shields.io/badge/LLM-Zhipu%20AI%20GLM--5.3--plus-orange.svg)](https://open.bigmodel.cn/)
[![Data Sources](https://img.shields.io/badge/Data%20Sources-arXiv%20q--fin%20%7C%20OpenAlex-purple.svg)](https://arxiv.org/archive/q-fin)

*A production-grade Python system and 24/7 Discord bot that continuously harvests cutting-edge and classic quantitative finance papers, evaluates mathematical depth and alpha novelty with **GLM-5.3-Plus** (dynamically configurable), downloads full-text PDFs with polite multi-threading, tokenizes documents, builds an interactive **Quantitative Knowledge Graph**, and broadcasts the top curated pick daily.*

[Features](#-key-features) • [Architecture](#-architecture) • [One-Click Setup](#-one-click-setup) • [Configuration](#-configuration) • [Discord Commands](#-discord-commands) • [Push to GitHub](#-how-to-push-to-github)

---

</div>

## 📑 Table of Contents
- [🌟 Key Features](#-key-features)
- [🏗 Architecture](#-architecture)
- [⚡ One-Click Setup](#-one-click-setup)
- [⚙️ Configuration](#️-configuration)
  - [Discord Bot Setup](#1-discord-bot-setup)
  - [GLM API Key Setup](#2-glm-api-key-setup)
- [🤖 Discord Commands](#-discord-commands)
- [🖥️ Interactive Dashboard & 5-Option Menu](#️-interactive-terminal-dashboard--5-option-menu)
- [💻 CLI Usage](#-cli-usage)
- [🕸 Interactive Knowledge Graph](#-interactive-knowledge-graph)
- [📤 How to Push to GitHub](#-how-to-push-to-github)
- [🧪 Running Tests](#-running-tests)
- [📄 License](#-license)

---

## 🌟 Key Features

### 1. Dedicated Quantitative Finance Ingestion
- **arXiv `q-fin`**: Monitors all quantitative subcategories in real time:
  - `q-fin.TR` (Trading & Market Microstructure)
  - `q-fin.PM` (Portfolio Management)
  - `q-fin.CP` (Computational Finance)
  - `q-fin.PR` (Pricing of Securities)
  - `q-fin.RM` (Risk Management)
  - `q-fin.ST` (Statistical Finance)
  - `econ.EM` (Econometrics)
- **OpenAlex Index**: Surfaces classic, high-impact foundational literature (Black-Scholes, Rough Volatility, Pairs Trading, Limit Order Book Dynamics, Hawkes Processes).

### 2. GLM 5.3 Flash Quantitative Evaluation Engine
- Evaluates abstracts through the persona of a **Director of Quantitative Research at a Systematic Hedge Fund**.
- Scores papers strictly on **mathematical rigor, theoretical validity, microstructure insight, and alpha potential** (filtering out trivial or overfitted backtests).
- Generates structured JSON:
  - **Overall Score (1-100)**
  - **Attention-grabbing hook** for quantitative researchers
  - **Mathematical & Model Innovation** (2-3 sentences)
  - **Alpha & Practical Trading Implication** (1-2 sentences)
  - **Extracted Quant Concepts** (`rough volatility`, `optimal execution`, `hawkes process`, etc.)

### 3. Polite Multi-Threaded PDF Downloader
- Concurrent downloads via `ThreadPoolExecutor` with a synchronized rate-limiter and random jitter.
- Automatic exponential backoff on HTTP 429/503 errors to **prevent IP bans or cancellations from academic servers**.
- Validates PDF binary headers (`%PDF-`), rejecting HTML error/captcha traps.

### 4. Full-Text BPE Tokenizer
- Extracts full text and computes exact token counts using `tiktoken` (`cl100k_base` BPE).
- Profiles section token distributions (Abstract, Intro, Methods/Architecture, Results).
- Extracts high-density domain keywords for graph linking.

### 5. Interactive Knowledge & Alpha Graph
- Builds a multi-relational network connecting papers and quant concepts using `NetworkX`.
- Exports a standalone interactive HTML visualizer ([`data/graph.html`](data/graph.html)) with physics simulation, score color-coding, and cluster exploration.

### 6. Continuous 24/7 Discord Bot
- Runs an automated daily scheduler (`@tasks.loop(hours=24)`).
- Delivers rich Discord embeds with color-coded badges (`🔥 GROUNDBREAKING ALPHA`, `✨ HIGHLY INTERESTING MODEL`, `💡 NOTABLE QUANT RESEARCH`).
- Full support for interactive Slash Commands and DM customization.

---

## 🏗 Architecture

```mermaid
graph TD
    subgraph Quant Data Ingestion
        A1[arXiv q-fin Feed] --> SM[Scraper Manager]
        A2[OpenAlex Works API] --> SM
    end

    subgraph Evaluation & Processing
        SM --> DB[(SQLite Database\npapers.db)]
        SM --> GLM[GLM Abstract Evaluator\nglm-5.3-plus]
        GLM -->|Score & Alpha Analysis| DB
        DB -->|Candidate Batch| PDF[Polite Multi-Threaded\nPDF Downloader]
        PDF -->|Jitter & Backoff| TOK[tiktoken BPE Tokenizer]
        TOK -->|Token Stats & Keywords| DB
    end

    subgraph Graph & Visualization
        DB --> NX[NetworkX Knowledge Graph]
        NX --> HTML[Interactive Vis.js HTML Map\ndata/graph.html]
    end

    subgraph Broadcasting & Interaction
        DB --> BOT[Discord Bot / CLI Engine]
        BOT -->|Daily Top Pick Embed| DISCORD[Discord Server Channel]
        BOT -->|Slash & DM Commands| USER[User / Quant Team]
    end
```

---

## ⚡ One-Click Setup

Launch the application directly using the automated startup scripts (which check Python, initialize a virtual environment, install requirements, and set up `.env`):

### Windows (Double-Click or Command Prompt)
```cmd
start.bat
```
*(Or pass arguments: `start.bat --bulk 50` or `start.bat --search "limit order books"`)*

### Windows (PowerShell)
```powershell
.\start.ps1
```

### Linux / macOS / WSL (Bash)
```bash
chmod +x start.sh
./start.sh
```

---

## ⚙️ Configuration

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Edit `.env` with your API credentials:
```env
# Discord Bot Credentials (optional on disk; can be entered in-memory)
DISCORD_TOKEN=your_discord_bot_token_here
DISCORD_CHANNEL_ID=your_target_discord_channel_id_here

# GLM Configuration (Note: API keys can be entered safely in RAM upon startup)
GLM_API_KEY=
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
GLM_MODEL=glm-5.3-plus
```

### 1. Discord Bot Setup
1. Go to the [Discord Developer Portal](https://discord.com/developers/applications) and create a **New Application**.
2. Navigate to the **Bot** tab:
   - Click **Reset Token** and copy the token to `DISCORD_TOKEN` in `.env` (or input securely via interactive menu).
   - Enable **Message Content Intent** under *Privileged Gateway Intents*.
3. Go to **OAuth2 -> URL Generator**:
   - Scopes: `bot`, `applications.commands`
   - Permissions: `Send Messages`, `Embed Links`, `Attach Files`, `Read Message History`
   - Open the generated URL in your browser to invite the bot to your server.
4. Copy the Channel ID where you want daily papers posted into `DISCORD_CHANNEL_ID`.

### 2. GLM API Key Setup & Zero-Disk-Leak Security
- Obtain an API key from [Zhipu AI BigModel Platform](https://open.bigmodel.cn/).
- **Memory-Only Session Mode (Recommended)**: When launching `start.bat` or `python main.py`, the engine prompts for your key with masked input and keeps it strictly in volatile memory (RAM). It is **never saved to disk or .env**, keeping your local repository 100% clean and immune to accidental leaks when pushing code to GitHub.
- **Dynamic Model Selection**: By default, the engine uses **`glm-5.3-plus`**. Model names are never hardcoded; you can specify any model via `GLM_MODEL` environment variable, `config.yaml`, CLI flag (`--model`), or menu option `[5]`.
- *Note: If no API key is provided, the system falls back to an intelligent heuristic mock analyzer so you can test the entire pipeline locally without cost.*

---

## 🤖 Discord Commands

| Slash Command | Description |
| :--- | :--- |
| `/daily_test` | Immediately triggers the complete daily harvest, GLM evaluation, and posting cycle. |
| `/search <query> [limit]` | Searches arXiv & OpenAlex, evaluates papers with GLM, and returns embed cards. |
| `/filter [new_interests]` | Displays active filter settings or updates your topics of interest directly from chat. |
| `/top [count]` | Lists the highest-rated historical quant papers in the local archive. |
| `/graph` | Generates and uploads the interactive `quant_knowledge_graph.html` directly into Discord. |
| `/stats` | Displays local database counts, tokenized PDFs, total corpus tokens, and unique concepts. |

---

## 🖥️ Interactive Terminal Dashboard & 5-Option Menu

When you launch `start.bat` (Windows) or `./start.sh` (Linux/macOS), you are presented with a live, comprehensive quantitative finance dashboard:

```
╔════════════════════════════════════════════════════════════════════════════╗
║      📈 QUANTITATIVE FINANCE RESEARCH ENGINE & AUTONOMOUS ARCHIVE          ║
╚════════════════════════════════════════════════════════════════════════════╝
 📡 Status: Discord Bot: ✅ Configured | Target Channel: ✅ ID: 1234567890
            GLM Model:   ✅ Live (glm-5.3-plus) [🔒 Session RAM - 0 Disk Writes]
──────────────────────────────────────────────────────────────────────────────
 📚 Corpus Metrics:
    Total Ingested: 1,420 papers  |  Evaluated: 1,420  |  PDFs Tokenized: 380  |  Tokens: 4,120,400

 🏷️  Category Breakdown (7 categories in database):
    • q-fin.TR (Trading & Market Microstructure)       :   420 papers
    • q-fin.PM (Portfolio Management)                  :   310 papers
    • q-fin.CP (Computational Finance)                 :   250 papers
    • q-fin.RM (Risk Management)                       :   180 papers
    • q-fin.PR (Pricing of Securities)                 :   140 papers
    • q-fin.ST (Statistical Finance)                   :    80 papers
    • OpenAlex / Multi-Disciplinary Topics             :    40 papers

 🎯 Alpha & Quality Score Tiers (Model: glm-5.3-plus):
    • 💎 Alpha / Elite  (Score >= 85):    84 papers [Priority Discord broadcast candidates]
    • ⚡ Notable Alpha  (Score 70-84):   312 papers [High empirical/algorithmic depth]
    • ⚪ Screened Out   (Score < 70) : 1,024 papers [Archived with lower alpha score]
    • 📊 Mean Quality Score: 68.4 / 100

 🧠 Top Alpha Drivers & Concepts: optimal execution (18), order flow toxicity (15), deep reinforcement learning (14)
══════════════════════════════════════════════════════════════════════════════
  [1] 🤖 Launch 24/7 Discord Bot        (Scheduled daily posts & slash commands)
  [2] ⚡ Bulk Harvest Engine            (Ingest 50 to 10,000+ papers in succession with GLM)
  [3] 🔍 Search & Evaluate Quant Papers (On-demand topic search across arXiv & OpenAlex)
  [4] 🌐 Knowledge Graph & Analytics    (Open interactive HTML graph & corpus stats)
  [5] ⚙️  API Credentials & Bot Settings (Configure Tokens, GLM Key, Channel, Interests)
  [0] 🚪 Exit
══════════════════════════════════════════════════════════════════════════════
```

### ⚡ Bulk Succession Harvest Engine
Option `[2]` allows you to continuously ingest and evaluate tens, hundreds, or thousands of papers in succession:
- Paginates through arXiv query offsets (`start=0, 50, 100, 150...`) across all `q-fin` categories with polite 3.0s delays to respect arXiv rate limits.
- Paginates through OpenAlex pages for classic/high-citation literature.
- **Feeds each abstract directly to `glm-5.3-plus` (or your chosen model)** to evaluate alpha potential and math rigor in real time.
- Automatically queues accepted papers ($\ge 70$ score or custom threshold) for polite multithreaded PDF downloading and BPE tokenization.
- **Safe Interruption**: Press `Ctrl+C` at any point during a 5,000-paper harvest; the engine cleanly saves all evaluated papers into SQLite with zero corruption.

---

## 💻 CLI Usage

The system can also run directly from the command line or via cron tasks:

```bash
# Ingest and screen 250 papers in continuous succession with GLM 5.3 Flash
python main.py --bulk 250

# Run a single daily harvest cycle immediately
python main.py --daily

# Search for papers on a specific topic
python main.py --search "rough heston volatility" --limit 5

# Rebuild and export the interactive HTML knowledge graph
python main.py --graph

# View database and tokenization statistics
python main.py --stats

# Run the 24/7 Discord bot daemon
python main.py --bot
```

---

## 🕸 Interactive Knowledge Graph

The knowledge graph connects papers and quantitative finance concepts using shared mathematical foundations and Jaccard similarity:

1. Run `python main.py --graph` (or use `/graph` in Discord).
2. Open [`data/graph.html`](data/graph.html) in any web browser.
3. Features:
   - **Physics simulation**: Clusters related papers naturally around common alpha concepts.
   - **Score Color Badges**: Red (Score ≥ 90), Gold (80-89), Blue (< 80), Teal (Concepts).
   - **Rich Hover Cards**: Displays paper title, score, token count, and venue.

---

## 📤 How to Push to GitHub

This repository is already pre-configured with a `.gitignore` that safely excludes all sensitive API keys (`.env`), downloaded PDFs (`data/pdfs/`), virtual environments (`venv/`), and SQLite databases.

Follow these steps in your terminal to push this project to GitHub:

```bash
# 1. Initialize git repository (if not already initialized)
git init

# 2. Add all files (safe due to .gitignore)
git add .

# 3. Create your initial commit
git commit -m "feat: initial commit for QuantPaperBot"

# 4. Set default branch to main
git branch -M main

# 5. Add your GitHub remote repository (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/QuantPaperBot.git

# 6. Push code to GitHub
git push -u origin main
```

---

## 🧪 Running Tests

Run the automated test suite to verify database CRUD operations, GLM parsing, tokenization, and multithreading:

```bash
python -m unittest discover tests
```

---

## 📄 License

Distributed under the [MIT License](LICENSE). Free for personal and commercial research use.
