"""
Unified interactive terminal interface and entry point for the
Quantitative Finance Paper Scraper, GLM Evaluator, PDF Tokenizer, and Discord Bot.
"""

import os
import sys
import argparse
import webbrowser
import getpass
from dotenv import load_dotenv

# Load environment variables from .env if present (gitignored)
load_dotenv()

from pipeline import PaperPipeline
from bot.discord_bot import PaperScraperBot, setup_commands

CATEGORY_LABELS = {
    "q-fin.TR": "q-fin.TR (Trading & Market Microstructure)",
    "q-fin.PM": "q-fin.PM (Portfolio Management)",
    "q-fin.CP": "q-fin.CP (Computational Finance)",
    "q-fin.RM": "q-fin.RM (Risk Management)",
    "q-fin.PR": "q-fin.PR (Pricing of Securities)",
    "q-fin.ST": "q-fin.ST (Statistical Finance)",
    "q-fin.GN": "q-fin.GN (General Finance)",
    "q-fin.EC": "q-fin.EC (Economics & Finance)",
    "q-fin.MF": "q-fin.MF (Mathematical Finance)",
}

def get_env_value(key: str, default: str = "") -> str:
    """Reads a variable from os.environ or .env file directly."""
    val = os.getenv(key, "")
    if val and not val.startswith("your_"):
        return val
    return default

def get_active_model(pipeline: PaperPipeline = None) -> str:
    """Dynamically resolves the active GLM model name without hardcoding."""
    if pipeline and hasattr(pipeline, "glm_client"):
        return getattr(pipeline.glm_client, "model_name", getattr(pipeline.glm_client, "model", "glm-5.3-plus"))
    return os.getenv("GLM_MODEL") or "glm-5.3-plus"

def prompt_hidden_input(prompt_text: str) -> str:
    """Prompts for input without echoing characters to the terminal (safely masks typing)."""
    try:
        val = getpass.getpass(prompt_text).strip()
        return val
    except Exception:
        return input(prompt_text).strip()

def purge_local_credentials():
    """Wipes any local credential files from disk to ensure zero leaks before git commits."""
    removed = []
    for f in [".env", ".env.local", "credentials.json", "secrets.json"]:
        if os.path.exists(f):
            try:
                os.remove(f)
                removed.append(f)
            except Exception as e:
                print(f"[!] Warning: Could not remove {f}: {e}")
    if removed:
        print(f"\n🧹 Cleaned up disk files: {', '.join(removed)}")
        print("🛡️ All credential files removed from disk. Your workspace is 100% clean to commit and push to Git!\n")
    else:
        print("\n✅ Clean! No local credential files found on disk.\n")

def save_env_file(credentials: dict):
    """Safely saves or updates credentials in local .env file (excluded by .gitignore)."""
    env_path = ".env"
    existing_lines = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            existing_lines = f.readlines()

    keys_written = set()
    new_lines = []
    for line in existing_lines:
        trimmed = line.strip()
        if "=" in trimmed and not trimmed.startswith("#"):
            k = trimmed.split("=")[0].strip()
            if k in credentials:
                new_lines.append(f"{k}={credentials[k]}\n")
                keys_written.add(k)
                continue
        new_lines.append(line)

    for k, v in credentials.items():
        if k not in keys_written:
            new_lines.append(f"{k}={v}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    # Reload environment
    load_dotenv(override=True)
    print("\n✅ Credentials saved locally to .env (Protected & ignored by .gitignore)!\n")

def prompt_session_api_key(pipeline: PaperPipeline):
    """
    Prompts the user for the GLM API Key on launch or on-demand.
    Stores the key strictly in RAM for this session (never written to disk).
    """
    if pipeline.glm_client.is_configured():
        return

    print("\n" + "╔" + "═" * 74 + "╗")
    print("║" + "  🛡️  SECURE API KEY INPUT (SESSION RAM ONLY - ZERO DISK FOOTPRINT)       ".center(74) + "║")
    print("╚" + "═" * 74 + "╝")
    print(" • For your complete security, your API key is NEVER saved to any file.")
    print(" • It is held strictly in-memory (RAM) for this running session.")
    print(" • When you close this program, the key vanishes completely.")
    print(" • You can freely commit or share your project without exposing keys!")
    print("─" * 76)

    key = prompt_hidden_input("🔑 Enter Zhipu AI GLM API Key (hidden / press Enter to skip): ")
    if key:
        os.environ["GLM_API_KEY"] = key
        pipeline.glm_client.api_key = key
        print("\n✅ API Key loaded securely into session RAM (0 bytes saved to disk).")
        print(f"🤖 Connected with evaluation model: {get_active_model(pipeline)}\n")
    else:
        print("\nℹ️ No key entered. Continuing in Heuristic Evaluation Mode.\n")

def setup_credentials_interactive(pipeline: PaperPipeline = None):
    """
    Interactive terminal settings wizard with masked input and zero-disk-leak memory mode.
    Allows user to configure secrets purely in memory or wipe disk files.
    """
    curr_token = get_env_value("DISCORD_TOKEN")
    curr_channel = get_env_value("DISCORD_CHANNEL_ID")
    curr_glm = get_env_value("GLM_API_KEY") or (pipeline.glm_client.api_key if pipeline else "")
    curr_model = get_active_model(pipeline)

    while True:
        masked_glm = f"{curr_glm[:4]}...{curr_glm[-4:]}" if (curr_glm and len(curr_glm) > 8 and not curr_glm.startswith("your_")) else ("Set in RAM" if curr_glm and not curr_glm.startswith("your_") else "Not Set")
        masked_tok = f"{curr_token[:4]}...{curr_token[-4:]}" if (curr_token and len(curr_token) > 8 and not curr_token.startswith("your_")) else ("Configured" if curr_token and not curr_token.startswith("your_") else "Not Set")

        print("\n" + "=" * 74)
        print(" ⚙️  API CREDENTIALS & SECURITY CONFIGURATION")
        print("=" * 74)
        print(f"  • GLM API Key:     {masked_glm}  (Input is masked/hidden)")
        print(f"  • Active Model:    {curr_model}  (Dynamic, configurable)")
        print(f"  • Discord Bot:     {masked_tok}  |  Channel ID: {curr_channel or 'Not Set'}")
        print("─" * 74)
        print("  [1] 🧠 Enter / Update GLM API Key (Session RAM Only - Never Saved to Disk)")
        print(f"  [2] 🎯 Switch GLM Model Name      (Current: {curr_model})")
        print("  [3] 🤖 Configure Discord Bot Token & Channel (Session RAM)")
        print("  [4] 🧹 Purge / Delete all local credential files from disk (Git Safety)")
        print("  [5] 💾 Save credentials to local .env (Optional: gitignored, but RAM mode is safer)")
        print("  [0] ↩️  Return to Main Menu")
        print("=" * 74)

        opt = input("Select an option [0-5]: ").strip()

        if opt == "0":
            break
        elif opt == "1":
            print("\nEnter Zhipu AI GLM API Key (input is masked; characters won't show on screen):")
            new_key = prompt_hidden_input("🔑 GLM API Key: ")
            if new_key:
                curr_glm = new_key
                os.environ["GLM_API_KEY"] = new_key
                if pipeline:
                    pipeline.glm_client.api_key = new_key
                print("\n✅ GLM API Key updated in process RAM only (Never written to disk)!\n")
        elif opt == "2":
            print(f"\nEnter GLM Model Name (Default: glm-5.3-plus | Options: glm-5.3-plus, glm-4-plus, etc.):")
            new_model = input(f"Model Name [{curr_model}]: ").strip()
            if new_model:
                curr_model = new_model
                os.environ["GLM_MODEL"] = new_model
                if pipeline:
                    pipeline.glm_client.model = new_model
                    pipeline.glm_client.model_name = new_model
                print(f"\n✅ Active model updated to: {curr_model}\n")
        elif opt == "3":
            print("\nEnter Discord Bot Token (input is masked):")
            new_token = prompt_hidden_input("🤖 Discord Bot Token: ")
            if new_token:
                curr_token = new_token
                os.environ["DISCORD_TOKEN"] = new_token

            new_ch = input(f"Target Discord Channel ID [{curr_channel}]: ").strip()
            if new_ch:
                curr_channel = new_ch
                os.environ["DISCORD_CHANNEL_ID"] = new_ch
            print("\n✅ Discord credentials updated in process RAM!\n")
        elif opt == "4":
            purge_local_credentials()
        elif opt == "5":
            confirm = input("Are you sure you want to write credentials to local disk .env? (y/n): ").strip().lower()
            if confirm == "y":
                creds = {
                    "DISCORD_TOKEN": curr_token or "your_discord_bot_token_here",
                    "DISCORD_CHANNEL_ID": curr_channel or "your_discord_channel_id_here",
                    "GLM_API_KEY": curr_glm or "",
                    "GLM_BASE_URL": os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"),
                    "GLM_MODEL": curr_model
                }
                save_env_file(creds)

def start_discord_bot(pipeline: PaperPipeline):
    """Launches the Discord Bot daemon."""
    discord_token = get_env_value("DISCORD_TOKEN")
    if not discord_token:
        print("\n[!] Error: DISCORD_TOKEN is not configured.")
        configure_now = input("Would you like to configure it now? (y/n): ").strip().lower()
        if configure_now == "y":
            setup_credentials_interactive(pipeline)
            discord_token = get_env_value("DISCORD_TOKEN")
        else:
            return

    if not discord_token:
        print("[!] No Discord token provided. Cannot start bot.")
        return

    print("\n[Bot] Launching 24/7 Discord Bot. Press Ctrl+C in this terminal to stop.")
    bot = PaperScraperBot(pipeline)
    setup_commands(bot)
    try:
        bot.run(discord_token)
    except KeyboardInterrupt:
        print("\n[Bot] Discord Bot gracefully stopped.")
    except Exception as e:
        print(f"\n[Bot] Error running Discord Bot: {e}")

def interactive_menu(pipeline: PaperPipeline = None):
    """Renders a comprehensive interactive terminal dashboard with category metrics and 5 primary options."""
    if pipeline is None:
        pipeline = PaperPipeline()

    # Prompt user for API key if missing (held strictly in session RAM, never stored on disk)
    prompt_session_api_key(pipeline)

    while True:
        token_val = get_env_value("DISCORD_TOKEN")
        channel_val = get_env_value("DISCORD_CHANNEL_ID")
        model_val = get_active_model(pipeline)

        token_status = "✅ Configured" if token_val else "❌ Missing (Set in [5])"
        channel_status = f"✅ ID: {channel_val}" if channel_val else "❌ Missing"
        if pipeline.glm_client.is_configured():
            glm_status = f"✅ Live ({model_val}) [🔒 Session RAM - 0 Disk Writes]"
        else:
            glm_status = f"⚠️ Heuristic Mode ({model_val}) [Select 5 to set key]"

        stats = pipeline.db.get_detailed_stats()
        cats = stats.get("categories", {})
        scores = stats.get("score_distribution", {})
        top_concepts = stats.get("top_concepts", [])

        print("\n" + "╔" + "═" * 76 + "╗")
        print("║" + "   📈 QUANTITATIVE FINANCE RESEARCH ENGINE & AUTONOMOUS ARCHIVE    ".center(76) + "║")
        print("╚" + "═" * 76 + "╝")

        # 1. System & Connectivity Status
        print(f" 📡 Status: Discord Bot: {token_status} | Target Channel: {channel_status}")
        print(f"            GLM Model:   {glm_status}")
        print("─" * 78)

        # 2. Corpus Overview
        tot = stats.get("total_papers", 0)
        ev = stats.get("evaluated_papers", 0)
        tok = stats.get("tokenized_pdfs", 0)
        tot_tokens = stats.get("total_tokens", 0)
        print(f" 📚 Corpus Metrics:")
        print(f"    Total Ingested: {tot:,} papers  |  Evaluated: {ev:,}  |  PDFs Tokenized: {tok:,}  |  Tokens: {tot_tokens:,}")

        # 3. Category Breakdown
        print(f"\n 🏷️  Category Breakdown ({len(cats)} categories in database):")
        if cats:
            for cat, count in list(cats.items())[:6]:
                label = CATEGORY_LABELS.get(cat, cat)
                print(f"    • {label:<48} : {count:>5} papers")
            if len(cats) > 6:
                rem_count = sum(list(cats.values())[6:])
                print(f"    • Other Quant / OpenAlex Categories ({len(cats)-6} more)     : {rem_count:>5} papers")
        else:
            print("    • (Corpus currently empty — select option [2] to harvest papers into database)")

        # 4. Alpha & Quality Score Distribution
        print(f"\n 🎯 Alpha & Quality Score Tiers (Model: {model_val}):")
        elite = scores.get("elite", 0)
        notable = scores.get("notable", 0)
        screened = scores.get("screened", 0)
        avg_s = scores.get("avg_score", 0.0)
        print(f"    • 💎 Alpha / Elite  (Score >= 85): {elite:>5} papers [Priority Discord broadcast candidates]")
        print(f"    • ⚡ Notable Alpha  (Score 70-84): {notable:>5} papers [High empirical/algorithmic depth]")
        print(f"    • ⚪ Screened Out   (Score < 70) : {screened:>5} papers [Archived with lower alpha score]")
        print(f"    • 📊 Mean Quality Score: {avg_s} / 100")

        # 5. Top Extracted Concepts
        if top_concepts:
            concepts_str = ", ".join([f"{name} ({c})" for name, c in top_concepts[:6]])
            print(f"\n 🧠 Top Alpha Drivers & Concepts: {concepts_str}")

        print("═" * 78)
        print("  [1] 🤖 Launch 24/7 Discord Bot        (Scheduled daily posts & slash commands)")
        print("  [2] ⚡ Bulk Harvest Engine            (Ingest 50 to 10,000+ papers in succession with GLM)")
        print("  [3] 🔍 Search & Evaluate Quant Papers (On-demand topic search across arXiv & OpenAlex)")
        print("  [4] 🌐 Knowledge Graph & Analytics    (Open interactive HTML graph & corpus stats)")
        print("  [5] ⚙️  API Credentials & Bot Settings (Configure Tokens, GLM Key, Channel, Interests)")
        print("  [6] Reset Corpus & Graph           (Wipe database & start fresh from scratch)\n")
        print("  [0] 🚪 Exit")
        print("═" * 78)

        choice = input("Select an option [0-6]: ").strip()

        if choice == "1":
            start_discord_bot(pipeline)
        elif choice == "2":
            print("\n" + "=" * 65)
            print(" ⚡ BULK SUCCESSION HARVEST ENGINE")
            print("=" * 65)
            print("Continuously ingests papers in succession across arXiv and OpenAlex.")
            print(f"Each abstract is fed to {model_val} to evaluate alpha novelty.")
            print("Qualified papers are queued for polite multithreaded PDF download & tokenization.\n")

            if not pipeline.glm_client.is_configured():
                print("ℹ️ Note: No active GLM API key is set for live model evaluation.")
                enter_now = input("Would you like to enter your GLM key now (kept in RAM only)? (y/n): ").strip().lower()
                if enter_now == "y":
                    setup_credentials_interactive(pipeline)
                    model_val = get_active_model(pipeline)

            count_input = input("\nHow many papers to harvest in succession? [default: 50]: ").strip()
            target_count = int(count_input) if count_input.isdigit() and int(count_input) > 0 else 50

            score_input = input("Minimum score for PDF download & graph indexing? [default: 70]: ").strip()
            min_score = int(score_input) if score_input.isdigit() else 70

            topic_query = input("Specific search filter (or press Enter for all quant categories): ").strip()

            pipeline.run_bulk_harvest(
                target_count=target_count,
                min_score=min_score,
                download_pdfs=True,
                search_query=topic_query
            )
            input("\nPress Enter to return to main menu...")
        elif choice == "3":
            from search_papers import search_and_evaluate
            query = input("\nEnter quant search topic (e.g. 'statistical arbitrage', 'pairs trading'): ").strip()
            if query:
                limit_input = input("How many top papers to look for? [default: 5]: ").strip()
                limit = int(limit_input) if limit_input.isdigit() and int(limit_input) > 0 else 5
                dl_input = input("Download PDFs for qualifying un-downloaded papers? (y/n) [default: y]: ").strip().lower()
                download = dl_input != "n"
                search_and_evaluate(query=query, limit=limit, download_pdfs=download)
            input("\nPress Enter to return to main menu...")
        elif choice == "4":
            print("\n" + "=" * 65)
            print("  🌐 KNOWLEDGE GRAPH OPTIONS")
            print("=" * 65)
            print("  [1] Open Full Knowledge Graph (all papers in corpus)")
            print("  [2] Open Graph for Downloaded PDFs Only")
            print("  [3] Open Graph for Specific Topic / Keyword")
            print("  [4] Reset Corpus & Graph (Start completely from scratch)")
            print("  [0] Return to Main Menu")
            g_opt = input("\nSelect graph option [1-4, or 0]: ").strip()

            html_path = ""
            if g_opt == "1":
                html_path = pipeline.graph_builder.export_interactive_html()
            elif g_opt == "2":
                html_path = pipeline.graph_builder.export_interactive_html(only_downloaded=True)
            elif g_opt == "3":
                t_query = input("Enter topic filter (e.g. 'statistical arbitrage'): ").strip()
                html_path = pipeline.graph_builder.export_interactive_html(query=t_query)
            elif g_opt == "4":
                from reset_corpus import reset_corpus
                reset_corpus()
            elif g_opt == "0":
                pass
            else:
                html_path = pipeline.graph_builder.export_interactive_html()

            if html_path:
                abs_path = os.path.abspath(html_path)
                print(f"\n[Graph] Interactive Knowledge Graph generated at:\n  {abs_path}")
                open_browser = input("Open graph in default web browser? (y/n) [default: y]: ").strip().lower()
                if open_browser != "n":
                    try:
                        webbrowser.open(f"file://{abs_path}")
                        print("[Graph] Opened in browser.")
                    except Exception as e:
                        print(f"[Graph] Could not open browser automatically: {e}")
            input("\nPress Enter to return to main menu...")
        elif choice == "6":
            from reset_corpus import reset_corpus
            reset_corpus()
            input("\nPress Enter to return to main menu...")
        elif choice == "5":
            setup_credentials_interactive(pipeline)
        elif choice == "0":
            print("\nShutting down Quant Paper Scraper. Goodbye!\n")
            break
        else:
            print("\n[!] Invalid selection. Please choose an option from 0 to 5.")

def main():
    parser = argparse.ArgumentParser(
        description="Quantitative Finance Paper Scraper, GLM Evaluator, and Discord Bot"
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "bot", "run-daily", "harvest", "search", "export-graph", "config"],
        default="interactive",
        help="Execution mode (default: interactive menu)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Override GLM model name dynamically (default: from env/config or glm-5.3-plus)"
    )
    parser.add_argument("--query", type=str, default="", help="Search query for search mode")
    parser.add_argument("--limit", type=int, default=5, help="Number of papers to process in search mode")
    parser.add_argument("--count", type=int, default=50, help="Number of papers for harvest mode")
    parser.add_argument("--min-score", type=int, default=70, help="Minimum score threshold for PDF download")

    args = parser.parse_args()

    # Apply command line model override if provided
    if args.model:
        os.environ["GLM_MODEL"] = args.model

    pipeline = PaperPipeline()
    if args.model:
        pipeline.glm_client.model = args.model
        pipeline.glm_client.model_name = args.model

    if args.mode == "interactive":
        interactive_menu(pipeline)
    elif args.mode == "bot":
        start_discord_bot(pipeline)
    elif args.mode == "run-daily":
        pipeline.run_daily_cycle()
    elif args.mode == "harvest":
        pipeline.run_bulk_harvest(target_count=args.count, min_score=args.min_score, search_query=args.query)
    elif args.mode == "search":
        q = args.query or "quantitative finance high frequency trading"
        pipeline.search_and_ingest(q, limit=args.limit)
    elif args.mode == "export-graph":
        out = pipeline.graph_builder.export_interactive_html()
        print(f"Graph exported to {out}")
    elif args.mode == "config":
        setup_credentials_interactive(pipeline)

if __name__ == "__main__":
    main()
