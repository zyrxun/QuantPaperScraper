"""
Unified interactive terminal interface and entry point for the
Quantitative Finance Paper Scraper, GLM Evaluator, PDF Tokenizer, and Discord Bot.
"""

import os
import sys
import argparse
import webbrowser
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

from pipeline import PaperPipeline
from bot.discord_bot import PaperScraperBot, setup_commands

def get_env_value(key: str, default: str = "") -> str:
    """Reads a variable from os.environ or .env file directly."""
    val = os.getenv(key, "")
    if val and not val.startswith("your_"):
        return val
    return default

def save_env_file(credentials: dict):
    """Safely saves or updates credentials in .env file."""
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
    print("\n✅ Credentials saved successfully to .env!\n")

def setup_credentials_interactive():
    """Interactive terminal wizard to input Discord Token, Channel ID, and GLM Key."""
    print("\n" + "=" * 65)
    print(" 🔑 QUANT PAPER SCRAPER - CREDENTIALS & API CONFIGURATION")
    print("=" * 65)
    print("Press Enter without typing to keep existing / default values.\n")

    curr_token = get_env_value("DISCORD_TOKEN")
    curr_channel = get_env_value("DISCORD_CHANNEL_ID")
    curr_glm = get_env_value("GLM_API_KEY")
    curr_base = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
    curr_model = os.getenv("GLM_MODEL", "glm-4-plus")

    # 1. Discord Bot Token
    print("1. Discord Bot Token (from Discord Developer Portal -> Bot -> Token):")
    if curr_token:
        print(f"   [Current: {curr_token[:6]}...{curr_token[-4:]}]")
    new_token = input("   Enter Discord Token: ").strip()
    token_to_save = new_token if new_token else curr_token

    # 2. Discord Channel ID
    print("\n2. Target Discord Channel ID (where papers will be posted):")
    if curr_channel:
        print(f"   [Current: {curr_channel}]")
    new_channel = input("   Enter Channel ID: ").strip()
    channel_to_save = new_channel if new_channel else curr_channel

    # 3. GLM API Key
    print("\n3. GLM API Key (from Zhipu AI bigmodel.cn | Leave blank for offline mock mode):")
    if curr_glm:
        print(f"   [Current: {curr_glm[:6]}...{curr_glm[-4:]}]")
    new_glm = input("   Enter GLM API Key: ").strip()
    glm_to_save = new_glm if new_glm else curr_glm

    # 4. GLM Model
    print("\n4. GLM Model Name (e.g. glm-4-plus, glm-4, glm-5):")
    print(f"   [Current: {curr_model}]")
    new_model = input(f"   Enter Model Name [{curr_model}]: ").strip()
    model_to_save = new_model if new_model else curr_model

    creds = {
        "DISCORD_TOKEN": token_to_save or "your_discord_bot_token_here",
        "DISCORD_CHANNEL_ID": channel_to_save or "your_discord_channel_id_here",
        "GLM_API_KEY": glm_to_save or "your_glm_api_key_here",
        "GLM_BASE_URL": curr_base,
        "GLM_MODEL": model_to_save
    }
    save_env_file(creds)

def start_discord_bot(pipeline: PaperPipeline):
    """Launches the Discord Bot daemon."""
    discord_token = get_env_value("DISCORD_TOKEN")
    if not discord_token:
        print("\n[!] Error: DISCORD_TOKEN is not configured.")
        configure_now = input("Would you like to configure it now? (y/n): ").strip().lower()
        if configure_now == "y":
            setup_credentials_interactive()
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

def interactive_menu():
    """Renders a continuous interactive terminal menu."""
    pipeline = PaperPipeline()

    while True:
        token_val = get_env_value("DISCORD_TOKEN")
        channel_val = get_env_value("DISCORD_CHANNEL_ID")
        glm_val = get_env_value("GLM_API_KEY")
        model_val = os.getenv("GLM_MODEL", "glm-4-plus")

        token_status = "✅ Configured" if token_val else "❌ Missing (Set in [4])"
        channel_status = f"✅ ID: {channel_val}" if channel_val else "❌ Missing"
        glm_status = f"✅ Live ({model_val})" if glm_val else "⚠️ Mock Heuristic Mode (Set in [4])"

        print("\n" + "═" * 70)
        print("         📈 QUANTITATIVE FINANCE PAPER SCRAPER & ENGINE")
        print("═" * 70)
        print(f" Status: Discord Bot Token: {token_status} | Channel: {channel_status}")
        print(f"         GLM LLM Engine:    {glm_status}")
        print("─" * 70)
        print("  [1] 🤖 Start 24/7 Discord Bot (Daily scheduled posts & slash commands)")
        print("  [2] 🏆 Run Daily Paper Curation Cycle Now (Harvest, score & tokenize)")
        print("  [3] 🔍 Search Quant Papers on Demand (arXiv & OpenAlex)")
        print("  [4] 🔑 Configure API Keys & Discord Bot Token (.env)")
        print("  [5] 🌐 Export & Open Interactive Knowledge Graph (HTML)")
        print("  [6] 📊 View Corpus & SQLite Database Stats")
        print("  [7] 🎯 View / Update Quant Topic Filter Interests")
        print("  [8] 🧪 Run Automated Test Suite")
        print("  [0] 🚪 Exit")
        print("═" * 70)

        choice = input("Select an option [0-8]: ").strip()

        if choice == "1":
            start_discord_bot(pipeline)
        elif choice == "2":
            print("\n[Action] Running complete quant curation cycle...")
            top_paper = pipeline.run_daily_cycle()
            if top_paper:
                print("\n" + "─" * 60)
                print(f"🏆 TODAY'S TOP QUANT PICK: #{top_paper['id']} (Score: {top_paper['score']}/100)")
                print(f"Title: {top_paper['title']}")
                print(f"Hook:  {top_paper.get('hook')}")
                print(f"Math:  {top_paper.get('breakthrough_summary')}")
                print(f"Alpha: {top_paper.get('takeaway')}")
                if top_paper.get("total_tokens"):
                    print(f"Tokens: {top_paper['total_tokens']:,}")
                print("─" * 60)
            else:
                print("\nHarvest finished, but no new candidates exceeded the score threshold.")
            input("\nPress Enter to return to menu...")
        elif choice == "3":
            query = input("\nEnter quant search topic (e.g. 'limit order book execution'): ").strip()
            if query:
                limit_input = input("How many papers to evaluate? [default: 3]: ").strip()
                limit = int(limit_input) if limit_input.isdigit() else 3
                print(f"\n[Search] Querying arXiv and OpenAlex for '{query}'...")
                results = pipeline.search_and_ingest(query, limit=limit)
                print(f"\nFound and evaluated {len(results)} papers:")
                for p in results:
                    print(f"\n🌟 [{p['score']}/100] {p['title']}")
                    print(f"   Hook: {p.get('hook')}")
                    print(f"   Alpha: {p.get('takeaway')}")
                    print(f"   Concepts: {', '.join(p.get('concepts', []))}")
            input("\nPress Enter to return to menu...")
        elif choice == "4":
            setup_credentials_interactive()
            pipeline = PaperPipeline() # reload with new credentials
            input("\nPress Enter to return to menu...")
        elif choice == "5":
            html_path = pipeline.graph_builder.export_interactive_html()
            abs_path = os.path.abspath(html_path)
            print(f"\n[Graph] Exported interactive HTML to: {abs_path}")
            open_browser = input("Open graph in web browser now? (y/n): ").strip().lower()
            if open_browser == "y":
                webbrowser.open(f"file:///{abs_path}")
            input("\nPress Enter to return to menu...")
        elif choice == "6":
            stats = pipeline.db.get_stats()
            print("\n=== Corpus & Database Statistics ===")
            print(f"Total Papers Ingested:    {stats['total_papers']}")
            print(f"Evaluated by GLM:         {stats['evaluated_papers']}")
            print(f"Dispatched to Discord:    {stats['posted_papers']}")
            print(f"Full-text PDFs Tokenized: {stats['tokenized_pdfs']}")
            print(f"Total Token Count:        {stats['total_tokens']:,}")
            print(f"Unique Quant Concepts:    {stats['unique_concepts']}")
            print("====================================")
            input("\nPress Enter to return to menu...")
        elif choice == "7":
            filters = pipeline.config.get("filters", {})
            current = filters.get("interests", [])
            print("\nCurrent Active Quant Research Interests:")
            for i in current:
                print(f" • {i}")
            new_val = input("\nEnter new comma-separated interests (or press Enter to keep): ").strip()
            if new_val:
                new_list = [x.strip() for x in new_val.split(",") if x.strip()]
                pipeline.update_filter_interests(new_list)
                print("✅ Updated filter interests!")
            input("\nPress Enter to return to menu...")
        elif choice == "8":
            print("\n[Tests] Running automated unittest suite...")
            import unittest
            loader = unittest.TestLoader()
            suite = loader.discover("tests")
            runner = unittest.TextTestRunner(verbosity=2)
            runner.run(suite)
            input("\nPress Enter to return to menu...")
        elif choice == "0":
            print("\nExiting QuantPaperBot. Happy trading & research! 👋")
            sys.exit(0)
        else:
            print("\nInvalid choice. Please select from 0 to 8.")

def main():
    parser = argparse.ArgumentParser(
        description="Quantitative Finance Paper Scraper, GLM Evaluator, PDF Tokenizer & Knowledge Graph"
    )
    parser.add_argument("--bot", action="store_true", help="Start the 24/7 Discord bot directly")
    parser.add_argument("--daily", action="store_true", help="Run a single daily curation cycle")
    parser.add_argument("--search", type=str, default="", help="Search on-demand for a quant topic")
    parser.add_argument("--limit", type=int, default=5, help="Number of papers for search")
    parser.add_argument("--graph", action="store_true", help="Export the interactive knowledge graph")
    parser.add_argument("--stats", action="store_true", help="Display database statistics")
    parser.add_argument("--setup", action="store_true", help="Run interactive credentials setup")

    args = parser.parse_args()

    # Direct CLI commands if specified
    if args.setup:
        setup_credentials_interactive()
        return

    if args.stats:
        pipeline = PaperPipeline()
        stats = pipeline.db.get_stats()
        print("\n=== Corpus & Database Statistics ===")
        print(f"Total Papers Ingested:    {stats['total_papers']}")
        print(f"Evaluated by GLM:         {stats['evaluated_papers']}")
        print(f"Dispatched to Discord:    {stats['posted_papers']}")
        print(f"Full-text PDFs Tokenized: {stats['tokenized_pdfs']}")
        print(f"Total Token Count:        {stats['total_tokens']:,}")
        print(f"Unique Graph Concepts:    {stats['unique_concepts']}")
        print("====================================\n")
        return

    if args.graph:
        pipeline = PaperPipeline()
        path = pipeline.graph_builder.export_interactive_html()
        print(f"[Main] Interactive Knowledge Graph exported to: {os.path.abspath(path)}")
        return

    if args.search:
        pipeline = PaperPipeline()
        print(f"[Main] Searching quant papers for topic: '{args.search}'...")
        results = pipeline.search_and_ingest(args.search, limit=args.limit)
        print(f"\n[Main] Ingested and evaluated {len(results)} papers:")
        for r in results:
            print(f"\n🌟 [{r['score']}/100] {r['title']}")
            print(f"   Hook: {r['hook']}")
            print(f"   Alpha: {r['takeaway']}")
            print(f"   Concepts: {', '.join(r.get('concepts', []))}")
        return

    if args.daily:
        pipeline = PaperPipeline()
        print("[Main] Executing immediate daily curation pipeline...")
        top_paper = pipeline.run_daily_cycle()
        if top_paper:
            print(f"\n🏆 Daily Top Pick Selected: #{top_paper['id']}")
            print(f"Title: {top_paper['title']}")
            print(f"Score: {top_paper['score']}/100")
            print(f"Hook:  {top_paper.get('hook')}")
            print(f"Breakthrough: {top_paper.get('breakthrough_summary')}")
            print(f"Broader Impact: {top_paper.get('takeaway')}")
        return

    if args.bot:
        pipeline = PaperPipeline()
        start_discord_bot(pipeline)
        return

    # Default: Open the Rich Interactive Terminal Menu!
    interactive_menu()

if __name__ == "__main__":
    main()
