"""
Discord Bot providing continuous daily paper curation, slash commands,
DM configuration, on-demand search, and knowledge graph distribution.
"""

import os
import discord
from discord.ext import commands, tasks
from discord import app_commands
from typing import Optional
from pipeline import PaperPipeline

class PaperScraperBot(commands.Bot):
    def __init__(self, pipeline: PaperPipeline):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)
        self.pipeline = pipeline
        self.channel_id = int(os.getenv("DISCORD_CHANNEL_ID", 0) or 0)

    async def setup_hook(self):
        """Initializes background task loops and syncs application commands."""
        # Start the 24h continuous daily curation loop
        self.daily_paper_task.start()
        try:
            synced = await self.tree.sync()
            print(f"[DiscordBot] Synced {len(synced)} slash commands.")
        except Exception as e:
            print(f"[DiscordBot] Error syncing slash commands: {e}")

    async def on_ready(self):
        print(f"[DiscordBot] Logged in as {self.user} (ID: {self.user.id})")
        print(f"[DiscordBot] Target notification channel ID: {self.channel_id}")
        await self.change_presence(activity=discord.Activity(
            type=discord.ActivityType.watching,
            name="Curating Quant Finance papers 📈"
        ))

    def create_paper_embed(self, paper: dict) -> discord.Embed:
        """Constructs an aesthetic, rich Discord embed for curated quant finance papers."""
        score = paper.get("score", 0)
        
        # Determine aesthetic color badge based on score
        if score >= 90:
            color = discord.Color.from_rgb(255, 107, 107) # Coral Red / Gold
            badge = "🔥 GROUNDBREAKING ALPHA / THEORY"
        elif score >= 80:
            color = discord.Color.from_rgb(254, 202, 87) # Warm Gold
            badge = "✨ HIGHLY INTERESTING MODEL"
        else:
            color = discord.Color.from_rgb(84, 160, 255) # Ocean Blue
            badge = "💡 NOTABLE QUANT RESEARCH"

        embed = discord.Embed(
            title=paper.get("title", "Untitled Research"),
            url=paper.get("pdf_url") or paper.get("doi") or "https://arxiv.org",
            description=f"**{badge}** • **Score: {score}/100**\n\n*{paper.get('hook', '')}*",
            color=color
        )

        # Mathematical / Model Innovation
        if paper.get("breakthrough_summary"):
            embed.add_field(
                name="📐 Mathematical & Model Innovation",
                value=paper["breakthrough_summary"][:1020],
                inline=False
            )

        # Alpha & Trading Implication
        if paper.get("takeaway"):
            embed.add_field(
                name="📊 Alpha & Trading Implication",
                value=paper["takeaway"][:1020],
                inline=False
            )

        # Concepts / Keywords
        concepts = paper.get("concepts", [])
        if concepts:
            concept_str = " • ".join([f"`{c}`" for c in concepts[:6]])
            embed.add_field(name="🏷️ Quant Concepts", value=concept_str, inline=False)

        # Metadata
        authors = paper.get("authors", [])
        author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        tokens = paper.get("total_tokens")
        token_str = f"{tokens:,} tokens" if tokens else "Available in abstract"

        embed.add_field(name="✍️ Authors", value=author_str or "Unknown", inline=True)
        embed.add_field(name="📅 Published", value=paper.get("published_date") or "Recent", inline=True)
        embed.add_field(name="📊 Full PDF Tokens", value=token_str, inline=True)

        embed.set_footer(text=f"Source: {paper.get('source', '').upper()} | Evaluated by GLM")
        return embed

    @tasks.loop(hours=24)
    async def daily_paper_task(self):
        """Continuous scheduled background task that triggers once every 24 hours."""
        await self.wait_until_ready()
        print("[DiscordBot] Running scheduled daily curation task...")

        if not self.channel_id:
            print("[DiscordBot] WARNING: DISCORD_CHANNEL_ID not set in .env. Skipping message send.")
            return

        channel = self.get_channel(self.channel_id)
        if not channel:
            try:
                channel = await self.fetch_channel(self.channel_id)
            except Exception as e:
                print(f"[DiscordBot] Could not find channel {self.channel_id}: {e}")
                return

        # Run pipeline
        top_paper = self.pipeline.run_daily_cycle()
        if top_paper:
            embed = self.create_paper_embed(top_paper)
            try:
                msg = await channel.send(
                    content="🔔 **Today's Top Quantitative Finance Paper Pick**",
                    embed=embed
                )
                self.pipeline.db.mark_as_posted(top_paper["id"], str(self.channel_id), str(msg.id))
                print(f"[DiscordBot] Successfully posted paper #{top_paper['id']} to channel {self.channel_id}.")
            except Exception as e:
                print(f"[DiscordBot] Failed to send message: {e}")
        else:
            print("[DiscordBot] No paper reached the minimum score threshold today.")

    @daily_paper_task.before_loop
    async def before_daily_paper(self):
        await self.wait_until_ready()

def setup_commands(bot: PaperScraperBot):
    """Registers slash commands and prefix commands for bot interaction."""

    @bot.tree.command(name="daily_test", description="Immediately triggers the daily quant paper scrape and evaluation pipeline.")
    async def daily_test(interaction: discord.Interaction):
        await interaction.response.defer(thinking=True)
        top_paper = bot.pipeline.run_daily_cycle()
        if top_paper:
            embed = bot.create_paper_embed(top_paper)
            msg = await interaction.followup.send(content="✅ **Daily Quant Curation Cycle Completed**", embed=embed)
            bot.pipeline.db.mark_as_posted(top_paper["id"], str(interaction.channel_id), str(msg.id))
        else:
            await interaction.followup.send("Harvest completed, but no new unposted paper met the score threshold.")

    @bot.tree.command(name="search", description="Search for quant finance papers on arXiv & OpenAlex and evaluate them with GLM.")
    @app_commands.describe(query="Topic or keywords to search for", limit="Number of results (1-5)")
    async def search(interaction: discord.Interaction, query: str, limit: Optional[int] = 3):
        await interaction.response.defer(thinking=True)
        results = bot.pipeline.search_and_ingest(query, limit=limit or 3)
        if not results:
            await interaction.followup.send(f"No papers found for query: `{query}`.")
            return

        embeds = [bot.create_paper_embed(p) for p in results[:3]]
        await interaction.followup.send(
            content=f"🔍 **Found {len(results)} Quant Papers evaluated by GLM for:** `{query}`",
            embeds=embeds
        )

    @bot.tree.command(name="graph", description="Export and upload the interactive Quant Knowledge Graph.")
    async def graph(interaction: discord.Interaction):
        await interaction.response.defer(thinking=True)
        html_path = bot.pipeline.graph_builder.export_interactive_html()
        stats = bot.pipeline.db.get_stats()

        if os.path.exists(html_path):
            file = discord.File(html_path, filename="quant_knowledge_graph.html")
            embed = discord.Embed(
                title="📈 Quantitative Finance Knowledge & Alpha Graph",
                description=(
                    f"**Quant Research Corpus Metrics:**\n"
                    f"• **Papers Cataloged:** {stats['total_papers']}\n"
                    f"• **Full PDFs Tokenized:** {stats['tokenized_pdfs']}\n"
                    f"• **Total Corpus Tokens:** {stats['total_tokens']:,}\n"
                    f"• **Extracted Quant Concepts:** {stats['unique_concepts']}\n\n"
                    f"Download and open `quant_knowledge_graph.html` in any browser to explore the interactive visual map!"
                ),
                color=discord.Color.teal()
            )
            await interaction.followup.send(embed=embed, file=file)
        else:
            await interaction.followup.send("Could not generate knowledge graph.")

    @bot.tree.command(name="filter", description="View or update your paper curation interests and filter settings.")
    @app_commands.describe(new_interests="Comma-separated topics of interest (optional)")
    async def filter_cmd(interaction: discord.Interaction, new_interests: Optional[str] = None):
        if new_interests:
            interests_list = [i.strip() for i in new_interests.split(",") if i.strip()]
            bot.pipeline.update_filter_interests(interests_list)
            await interaction.response.send_message(
                f"✅ **Updated Curation Interests:**\n" + "\n".join([f"• {i}" for i in interests_list])
            )
        else:
            filters = bot.pipeline.config.get("filters", {})
            current = filters.get("interests", [])
            min_score = filters.get("min_score", 75)
            await interaction.response.send_message(
                f"📋 **Current Paper Filter Settings:**\n"
                f"• **Min Score:** `{min_score}/100`\n"
                f"• **Active Topic Interests:**\n" + "\n".join([f"  - {i}" for i in current]) +
                f"\n\n*To update, use `/filter new_interests: Topic 1, Topic 2, Topic 3` (in DM or channel)*"
            )

    @bot.tree.command(name="stats", description="View local database and tokenization statistics.")
    async def stats_cmd(interaction: discord.Interaction):
        stats = bot.pipeline.db.get_stats()
        embed = discord.Embed(
            title="📊 Paper Scraper & Knowledge Engine Stats",
            color=discord.Color.purple()
        )
        embed.add_field(name="📚 Total Papers", value=str(stats["total_papers"]), inline=True)
        embed.add_field(name="🤖 Evaluated by GLM", value=str(stats["evaluated_papers"]), inline=True)
        embed.add_field(name="📢 Posted to Discord", value=str(stats["posted_papers"]), inline=True)
        embed.add_field(name="📄 Tokenized PDFs", value=str(stats["tokenized_pdfs"]), inline=True)
        embed.add_field(name="🔤 Total Tokens", value=f"{stats['total_tokens']:,}", inline=True)
        embed.add_field(name="🧠 Unique Concepts", value=str(stats["unique_concepts"]), inline=True)
        await interaction.response.send_message(embed=embed)

    @bot.tree.command(name="top", description="View top rated historical papers in your database.")
    @app_commands.describe(count="Number of top papers to show (1-5)")
    async def top_cmd(interaction: discord.Interaction, count: Optional[int] = 3):
        papers = bot.pipeline.db.get_top_papers(limit=count or 3)
        if not papers:
            await interaction.response.send_message("No evaluated papers in the database yet.")
            return

        embeds = [bot.create_paper_embed(p) for p in papers]
        await interaction.response.send_message(content="🏆 **Top Rated Papers in Archive**", embeds=embeds)
