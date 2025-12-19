#!/usr/bin/env python3
"""
Tab Curator - AI-Powered Tab Management & Learning System
Uses Gemini Flash 2.0 for intelligent clustering and summarization
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re
from urllib.parse import urlparse

from google import genai
from google.genai import types
from bs4 import BeautifulSoup
import requests
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.prompt import Prompt, Confirm
from rich.progress import track
from rich.markdown import Markdown
import typer

# Initialize
app = typer.Typer(help="Tab Curator - Intelligent Tab Management System")
console = Console()

# Configuration
DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "tabs.db"
HIERARCHY_PATH = DATA_DIR / "hierarchy.json"
EXPORT_DIR = Path("exports")

# Create directories
for d in [DATA_DIR, CACHE_DIR, EXPORT_DIR]:
    d.mkdir(exist_ok=True)

# Initialize Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
    model = 'gemini-2.0-flash-exp'
else:
    console.print("[yellow]Warning: GEMINI_API_KEY not set. Set it with: export GEMINI_API_KEY='your-key'[/yellow]")
    client = None
    model = None


@dataclass
class Tab:
    url: str
    title: Optional[str] = None
    content: Optional[str] = None
    summary: Optional[str] = None
    category: Optional[str] = None
    status: str = "pending"  # pending, read, archived
    created_at: str = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class Database:
    """SQLite database for persistent storage"""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_db()
    
    def _init_db(self):
        """Create tables if they don't exist"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tabs (
                url TEXT PRIMARY KEY,
                title TEXT,
                content TEXT,
                summary TEXT,
                category TEXT,
                status TEXT DEFAULT 'pending',
                created_at TEXT,
                updated_at TEXT
            )
        """)
        self.conn.commit()
    
    def insert_tab(self, tab: Tab):
        """Insert or update a tab"""
        self.conn.execute("""
            INSERT OR REPLACE INTO tabs 
            (url, title, content, summary, category, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (tab.url, tab.title, tab.content, tab.summary, tab.category, 
              tab.status, tab.created_at, datetime.now().isoformat()))
        self.conn.commit()
    
    def get_tab(self, url: str) -> Optional[Tab]:
        """Get a tab by URL"""
        cursor = self.conn.execute(
            "SELECT url, title, content, summary, category, status, created_at FROM tabs WHERE url = ?",
            (url,)
        )
        row = cursor.fetchone()
        if row:
            return Tab(*row)
        return None
    
    def get_all_tabs(self, status: Optional[str] = None) -> List[Tab]:
        """Get all tabs, optionally filtered by status"""
        if status:
            cursor = self.conn.execute(
                "SELECT url, title, content, summary, category, status, created_at FROM tabs WHERE status = ?",
                (status,)
            )
        else:
            cursor = self.conn.execute(
                "SELECT url, title, content, summary, category, status, created_at FROM tabs"
            )
        return [Tab(*row) for row in cursor.fetchall()]
    
    def update_status(self, url: str, status: str):
        """Update tab status"""
        self.conn.execute(
            "UPDATE tabs SET status = ?, updated_at = ? WHERE url = ?",
            (status, datetime.now().isoformat(), url)
        )
        self.conn.commit()
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about tabs"""
        cursor = self.conn.execute("""
            SELECT status, COUNT(*) FROM tabs GROUP BY status
        """)
        stats = dict(cursor.fetchall())
        cursor = self.conn.execute("SELECT COUNT(*) FROM tabs")
        stats['total'] = cursor.fetchone()[0]
        return stats


class ContentFetcher:
    """Fetch and extract content from URLs"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
    
    def fetch(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Fetch title and content from URL"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style', 'nav', 'footer', 'header']):
                script.decompose()
            
            # Get title
            title = soup.find('title')
            title = title.get_text().strip() if title else urlparse(url).netloc
            
            # Get main content
            # Try to find main content areas
            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            
            if main_content:
                # Extract text
                text = main_content.get_text(separator='\n', strip=True)
                # Clean up excessive whitespace
                text = re.sub(r'\n\s*\n', '\n\n', text)
                text = text[:10000]  # Limit to first 10k chars
                return title, text
            
            return title, None
            
        except Exception as e:
            console.print(f"[red]Error fetching {url}: {e}[/red]")
            return None, None


class GeminiClusterer:
    """Use Gemini for intelligent clustering and summarization"""
    
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name
    
    def cluster_tabs(self, tabs: List[Tab]) -> Dict:
        """Create hierarchical clusters of tabs"""
        if not self.client:
            console.print("[red]Gemini API not configured. Please set GEMINI_API_KEY.[/red]")
            return {}
        
        # Break into batches for better clustering
        batch_size = 50
        all_categories = []
        
        for i in range(0, len(tabs), batch_size):
            batch = tabs[i:i + batch_size]
            console.print(f"[cyan]Processing batch {i//batch_size + 1}/{(len(tabs)-1)//batch_size + 1}...[/cyan]")
            
            # Prepare tab data for clustering
            tab_data = []
            for tab in batch:
                tab_data.append({
                    'url': tab.url,
                    'title': tab.title or 'Untitled',
                    'snippet': (tab.content or '')[:300]
                })
            
            prompt = f"""Analyze these {len(tab_data)} browser tabs and organize them into categories.

Tabs:
{json.dumps(tab_data, indent=2)}

Create a 2-level hierarchical structure:
- Level 1: Main categories (3-5 broad topics)
- Level 2: Individual tabs under each category

CRITICAL: Return ONLY valid JSON. No markdown, no explanation, no additional text.

{{
  "categories": [
    {{
      "name": "Category Name",
      "description": "Brief description",
      "tabs": ["url1", "url2", "url3"]
    }}
  ]
}}

Group by topic similarity (AI/ML, Business, Development, Health, Finance, etc.)"""

            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        response_mime_type="application/json"
                    )
                )
                text = response.text.strip()
                
                # Extract JSON from response
                if '```json' in text:
                    text = text.split('```json')[1].split('```')[0].strip()
                elif '```' in text:
                    text = text.split('```')[1].split('```')[0].strip()
                
                batch_hierarchy = json.loads(text)
                all_categories.extend(batch_hierarchy.get('categories', []))
                
            except Exception as e:
                console.print(f"[yellow]Warning: Error in batch {i//batch_size + 1}: {e}[/yellow]")
                # Create fallback category for this batch
                all_categories.append({
                    "name": f"Uncategorized Batch {i//batch_size + 1}",
                    "description": "Tabs that couldn't be automatically categorized",
                    "tabs": [tab.url for tab in batch]
                })
                continue
        
        # Now consolidate categories with similar names
        console.print("[cyan]Consolidating categories...[/cyan]")
        consolidated = self._consolidate_categories(all_categories)
        
        return {"categories": consolidated}
    
    def _consolidate_categories(self, categories: List[Dict]) -> List[Dict]:
        """Consolidate similar categories using fuzzy matching"""
        if not categories:
            return []
        
        # Use Gemini to intelligently merge similar categories
        if len(categories) <= 10:
            return categories
        
        console.print(f"[cyan]Using AI to merge {len(categories)} categories...[/cyan]")
        
        # Prepare category list
        cat_list = []
        for i, cat in enumerate(categories):
            cat_list.append({
                'id': i,
                'name': cat.get('name', 'Uncategorized'),
                'description': cat.get('description', ''),
                'tab_count': len(cat.get('tabs', []))
            })
        
        prompt = f"""You have {len(cat_list)} categories that need to be consolidated into 8-12 main categories.

Current categories:
{json.dumps(cat_list, indent=2)}

Create a mapping that groups similar categories together. Return ONLY valid JSON:

{{
  "mappings": [
    {{
      "new_name": "Consolidated Category Name",
      "new_description": "Brief description",
      "merge_ids": [1, 5, 8]
    }}
  ]
}}

Rules:
- Group semantically similar categories (e.g., "AI/ML Research" with "Artificial Intelligence")
- Keep distinct topics separate (e.g., don't merge "Business" with "Photography")
- Aim for 8-12 final categories
- Use clear, concise names"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json"
                )
            )
            
            text = response.text.strip()
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()
            
            mappings = json.loads(text)
            
            # Apply mappings
            used_ids = set()
            consolidated = []
            
            for mapping in mappings.get('mappings', []):
                merge_ids = mapping.get('merge_ids', [])
                all_tabs = []
                subcats = []
                
                for cat_id in merge_ids:
                    if cat_id < len(categories) and cat_id not in used_ids:
                        cat = categories[cat_id]
                        tabs = cat.get('tabs', [])
                        all_tabs.extend(tabs)
                        
                        if tabs:
                            subcats.append({
                                'name': cat.get('name', 'Untitled'),
                                'tabs': tabs
                            })
                        used_ids.add(cat_id)
                
                if all_tabs:
                    consolidated.append({
                        'name': mapping.get('new_name', 'Uncategorized'),
                        'description': mapping.get('new_description', ''),
                        'subcategories': subcats
                    })
            
            # Add any categories that weren't merged
            for i, cat in enumerate(categories):
                if i not in used_ids:
                    tabs = cat.get('tabs', [])
                    if tabs:
                        consolidated.append({
                            'name': cat.get('name', 'Uncategorized'),
                            'description': cat.get('description', ''),
                            'subcategories': [{
                                'name': cat.get('name', 'Untitled'),
                                'tabs': tabs
                            }]
                        })
            
            console.print(f"[green]Consolidated into {len(consolidated)} categories[/green]")
            return consolidated
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not consolidate categories: {e}[/yellow]")
            # Fallback: simple grouping
            return self._simple_consolidate(categories)
    
    def _simple_consolidate(self, categories: List[Dict]) -> List[Dict]:
        """Simple consolidation fallback"""
        category_map = {}
        
        for cat in categories:
            name = cat.get('name', 'Uncategorized')
            # Extract main topic (first part before dash or ampersand)
            main_topic = name.split('-')[0].split('&')[0].strip().lower()
            
            if main_topic not in category_map:
                category_map[main_topic] = {
                    'name': name.split('-')[0].split('&')[0].strip(),
                    'description': cat.get('description', ''),
                    'subcategories': []
                }
            
            tabs = cat.get('tabs', [])
            if tabs:
                category_map[main_topic]['subcategories'].append({
                    'name': name,
                    'tabs': tabs
                })
        
        return list(category_map.values())
    
    def summarize(self, tab: Tab) -> str:
        """Generate a summary of a tab's content"""
        if not self.client:
            return "Summary unavailable - Gemini API not configured"
        
        if not tab.content:
            return "No content available to summarize"
        
        prompt = f"""Summarize this article in 3-5 sentences. Focus on key insights and takeaways.

Title: {tab.title}
URL: {tab.url}

Content:
{tab.content[:3000]}

Provide a concise, informative summary that captures the main points."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            return f"Error generating summary: {e}"


class InteractiveExplorer:
    """Interactive terminal UI for exploring tabs"""
    
    def __init__(self, db: Database, clusterer: GeminiClusterer):
        self.db = db
        self.clusterer = clusterer
        self.hierarchy = self._load_hierarchy()
    
    def _load_hierarchy(self) -> Dict:
        """Load hierarchy from file"""
        if HIERARCHY_PATH.exists():
            with open(HIERARCHY_PATH, 'r') as f:
                return json.load(f)
        return {}
    
    def explore(self):
        """Main exploration loop"""
        if not self.hierarchy:
            console.print("[yellow]No hierarchy found. Run 'cluster' command first.[/yellow]")
            return
        
        self._show_stats()
        self._explore_categories(self.hierarchy.get('categories', []))
    
    def _show_stats(self):
        """Show statistics"""
        stats = self.db.get_stats()
        
        table = Table(title="Tab Statistics")
        table.add_column("Status", style="cyan")
        table.add_column("Count", style="magenta")
        
        table.add_row("Total", str(stats.get('total', 0)))
        table.add_row("Pending", str(stats.get('pending', 0)))
        table.add_row("Read", str(stats.get('read', 0)))
        table.add_row("Archived", str(stats.get('archived', 0)))
        
        console.print(table)
        console.print()
    
    def _explore_categories(self, categories: List[Dict]):
        """Explore categories"""
        while True:
            console.print(Panel.fit("🗂️  Categories", style="bold blue"))
            
            for i, cat in enumerate(categories, 1):
                pending = self._count_pending_in_category(cat)
                console.print(f"{i}. {cat['name']} - {cat.get('description', '')} ({pending} pending)")
            
            console.print("0. Exit")
            
            choice = Prompt.ask("Select category", default="0")
            
            if choice == "0":
                break
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(categories):
                    self._explore_subcategories(categories[idx])
            except ValueError:
                console.print("[red]Invalid choice[/red]")
    
    def _count_pending_in_category(self, category: Dict) -> int:
        """Count pending tabs in a category"""
        count = 0
        for subcat in category.get('subcategories', []):
            for url in subcat.get('tabs', []):
                tab = self.db.get_tab(url)
                if tab and tab.status == 'pending':
                    count += 1
        return count
    
    def _explore_subcategories(self, category: Dict):
        """Explore subcategories"""
        while True:
            console.print(Panel.fit(f"📁 {category['name']}", style="bold green"))
            
            subcats = category.get('subcategories', [])
            for i, subcat in enumerate(subcats, 1):
                pending = sum(1 for url in subcat.get('tabs', []) 
                            if self.db.get_tab(url) and self.db.get_tab(url).status == 'pending')
                console.print(f"{i}. {subcat['name']} ({pending} pending)")
            
            console.print("0. Back")
            
            choice = Prompt.ask("Select subcategory", default="0")
            
            if choice == "0":
                break
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(subcats):
                    self._explore_tabs(subcats[idx])
            except ValueError:
                console.print("[red]Invalid choice[/red]")
    
    def _explore_tabs(self, subcategory: Dict):
        """Explore individual tabs"""
        tabs = [self.db.get_tab(url) for url in subcategory.get('tabs', [])]
        tabs = [t for t in tabs if t]  # Filter None
        
        while tabs:
            console.print(Panel.fit(f"📄 {subcategory['name']}", style="bold yellow"))
            
            for i, tab in enumerate(tabs, 1):
                status_icon = "✅" if tab.status == "read" else "📦" if tab.status == "archived" else "⭕"
                console.print(f"{i}. {status_icon} {tab.title or tab.url[:50]}")
            
            console.print("0. Back")
            
            choice = Prompt.ask("Select tab to view", default="0")
            
            if choice == "0":
                break
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(tabs):
                    self._view_tab(tabs[idx])
                    # Refresh tab data
                    tabs[idx] = self.db.get_tab(tabs[idx].url)
            except ValueError:
                console.print("[red]Invalid choice[/red]")
    
    def _view_tab(self, tab: Tab):
        """View a single tab"""
        console.clear()
        console.print(Panel.fit(tab.title or "Untitled", style="bold magenta"))
        console.print(f"[cyan]URL:[/cyan] {tab.url}")
        console.print(f"[cyan]Status:[/cyan] {tab.status}")
        console.print()
        
        # Generate summary if not exists
        if not tab.summary and tab.content:
            console.print("[yellow]Generating summary...[/yellow]")
            tab.summary = self.clusterer.summarize(tab)
            self.db.insert_tab(tab)
        
        if tab.summary:
            console.print(Panel(Markdown(tab.summary), title="Summary", border_style="blue"))
        else:
            console.print("[yellow]No summary available[/yellow]")
        
        console.print()
        
        # Actions
        console.print("Actions:")
        console.print("1. Mark as read")
        console.print("2. Archive")
        console.print("3. Open in browser")
        console.print("0. Back")
        
        action = Prompt.ask("Choose action", default="0")
        
        if action == "1":
            self.db.update_status(tab.url, "read")
            console.print("[green]Marked as read[/green]")
        elif action == "2":
            self.db.update_status(tab.url, "archived")
            console.print("[green]Archived[/green]")
        elif action == "3":
            import webbrowser
            webbrowser.open(tab.url)


@app.command()
def import_tabs(
    files: List[Path] = typer.Argument(..., help="Files containing tab URLs")
):
    """Import tabs from exported URL files"""
    db = Database()
    fetcher = ContentFetcher()
    
    all_urls = []
    
    # Read all URLs from files
    for file_path in files:
        with open(file_path, 'r') as f:
            content = f.read()
            # Handle space-separated or newline-separated URLs
            urls = re.findall(r'https?://[^\s]+', content)
            all_urls.extend(urls)
    
    # Remove duplicates
    all_urls = list(set(all_urls))
    
    console.print(f"[cyan]Found {len(all_urls)} unique URLs[/cyan]")
    
    # Fetch and store tabs
    for url in track(all_urls, description="Importing tabs..."):
        # Check if already exists
        existing = db.get_tab(url)
        if existing:
            continue
        
        title, content = fetcher.fetch(url)
        tab = Tab(url=url, title=title, content=content)
        db.insert_tab(tab)
    
    console.print(f"[green]✓ Imported {len(all_urls)} tabs[/green]")


@app.command()
def cluster():
    """Cluster tabs into hierarchical categories"""
    db = Database()
    
    if not client:
        console.print("[red]Error: GEMINI_API_KEY not set[/red]")
        return
    
    tabs = db.get_all_tabs()
    
    if not tabs:
        console.print("[yellow]No tabs found. Run 'import-tabs' first.[/yellow]")
        return
    
    console.print(f"[cyan]Clustering {len(tabs)} tabs...[/cyan]")
    
    clusterer = GeminiClusterer(client, model)
    hierarchy = clusterer.cluster_tabs(tabs)
    
    # Save hierarchy
    with open(HIERARCHY_PATH, 'w') as f:
        json.dump(hierarchy, f, indent=2)
    
    console.print(f"[green]✓ Created hierarchy with {len(hierarchy.get('categories', []))} categories[/green]")
    
    # Show summary
    for cat in hierarchy.get('categories', [])[:10]:
        subcat_count = len(cat.get('subcategories', []))
        tab_count = sum(len(sc.get('tabs', [])) for sc in cat.get('subcategories', []))
        console.print(f"  • {cat['name']}: {tab_count} tabs in {subcat_count} subcategories")
    
    if len(hierarchy.get('categories', [])) > 10:
        console.print(f"  ... and {len(hierarchy.get('categories', [])) - 10} more categories")


@app.command()
def recluster():
    """Re-cluster with better consolidation (useful if you have too many categories)"""
    if not HIERARCHY_PATH.exists():
        console.print("[yellow]No hierarchy found. Run 'cluster' first.[/yellow]")
        return
    
    with open(HIERARCHY_PATH, 'r') as f:
        hierarchy = json.load(f)
    
    categories = hierarchy.get('categories', [])
    console.print(f"[cyan]Current: {len(categories)} categories[/cyan]")
    
    if not client:
        console.print("[red]Error: GEMINI_API_KEY not set[/red]")
        return
    
    clusterer = GeminiClusterer(client, model)
    
    # Flatten all tabs back into categories
    flat_categories = []
    for cat in categories:
        for subcat in cat.get('subcategories', []):
            flat_categories.append({
                'name': cat['name'],
                'description': cat.get('description', ''),
                'tabs': subcat.get('tabs', [])
            })
    
    # Re-consolidate
    consolidated = clusterer._consolidate_categories(flat_categories)
    
    # Save
    new_hierarchy = {'categories': consolidated}
    with open(HIERARCHY_PATH, 'w') as f:
        json.dump(new_hierarchy, f, indent=2)
    
    console.print(f"[green]✓ Consolidated into {len(consolidated)} categories[/green]")
    
    # Show summary
    for cat in consolidated:
        subcat_count = len(cat.get('subcategories', []))
        tab_count = sum(len(sc.get('tabs', [])) for sc in cat.get('subcategories', []))
        console.print(f"  • {cat['name']}: {tab_count} tabs in {subcat_count} subcategories")


@app.command()
def explore():
    """Interactively explore your tabs"""
    db = Database()
    clusterer = GeminiClusterer(client, model)
    explorer = InteractiveExplorer(db, clusterer)
    explorer.explore()


@app.command()
def export(
    format: str = typer.Option("notebooklm", help="Export format: notebooklm, markdown, json"),
    output: Path = typer.Option("export.txt", help="Output file"),
    status: str = typer.Option(None, help="Filter by status: pending, read, archived")
):
    """Export tabs to various formats"""
    db = Database()
    
    if status:
        tabs = db.get_all_tabs(status=status)
    else:
        tabs = db.get_all_tabs()
    
    output_path = EXPORT_DIR / output
    
    if format == "notebooklm":
        # Export as text suitable for NotebookLM
        with open(output_path, 'w') as f:
            f.write("# Tab Collection for Learning\n\n")
            
            for tab in tabs:
                f.write(f"## {tab.title or 'Untitled'}\n\n")
                f.write(f"URL: {tab.url}\n\n")
                if tab.summary:
                    f.write(f"{tab.summary}\n\n")
                f.write("---\n\n")
    
    elif format == "markdown":
        with open(output_path, 'w') as f:
            f.write("# My Tabs\n\n")
            for tab in tabs:
                f.write(f"- [{tab.title or tab.url}]({tab.url})\n")
    
    elif format == "json":
        with open(output_path, 'w') as f:
            data = [asdict(tab) for tab in tabs]
            json.dump(data, f, indent=2)
    
    elif format == "urls":
        # Simple list of URLs
        with open(output_path, 'w') as f:
            for tab in tabs:
                f.write(f"{tab.url}\n")
    
    console.print(f"[green]✓ Exported {len(tabs)} tabs to {output_path}[/green]")


@app.command()
def close_reviewed():
    """Generate a script to close all read/archived tabs in Chrome"""
    db = Database()
    reviewed = db.get_all_tabs(status="read") + db.get_all_tabs(status="archived")
    
    if not reviewed:
        console.print("[yellow]No reviewed tabs found. Mark some as read/archived first.[/yellow]")
        return
    
    # Export URLs
    urls_file = EXPORT_DIR / "reviewed_tabs.txt"
    with open(urls_file, 'w') as f:
        for tab in reviewed:
            f.write(f"{tab.url}\n")
    
    # Create AppleScript for macOS Chrome
    script_file = EXPORT_DIR / "close_reviewed_tabs.scpt"
    with open(script_file, 'w') as f:
        f.write('''-- Close reviewed tabs in Chrome
tell application "Google Chrome"
    set urlList to paragraphs of (read POSIX file "''' + str(urls_file.absolute()) + '''" as «class utf8»)
    
    repeat with w in windows
        set tabList to tabs of w
        repeat with t in tabList
            set tabURL to URL of t
            repeat with reviewedURL in urlList
                if tabURL contains reviewedURL then
                    close t
                    exit repeat
                end if
            end repeat
        end repeat
    end repeat
end tell
''')
    
    console.print(f"[green]✓ Created script to close {len(reviewed)} reviewed tabs[/green]")
    console.print(f"\n[cyan]To close these tabs in Chrome, run:[/cyan]")
    console.print(f"  osascript {script_file}")
    console.print(f"\n[yellow]⚠️  Make sure Chrome is running first![/yellow]")
    console.print(f"\n[dim]URLs saved to: {urls_file}[/dim]")


@app.command()
def list_reviewed(
    status: str = typer.Option("read", help="Status to list: read, archived, or both")
):
    """List all reviewed tabs with their URLs"""
    db = Database()
    
    if status == "both":
        tabs = db.get_all_tabs(status="read") + db.get_all_tabs(status="archived")
    else:
        tabs = db.get_all_tabs(status=status)
    
    if not tabs:
        console.print(f"[yellow]No {status} tabs found.[/yellow]")
        return
    
    table = Table(title=f"{status.capitalize()} Tabs ({len(tabs)})")
    table.add_column("Title", style="cyan", max_width=60)
    table.add_column("URL", style="dim", max_width=50)
    table.add_column("Category", style="green")
    
    for tab in tabs:
        table.add_row(
            tab.title or "Untitled",
            tab.url[:47] + "..." if len(tab.url) > 50 else tab.url,
            tab.category or "N/A"
        )
    
    console.print(table)
    
    # Ask if they want to export
    if Confirm.ask(f"\n[cyan]Export these {len(tabs)} URLs to a file?[/cyan]"):
        filename = f"{status}_tabs.txt"
        export_path = EXPORT_DIR / filename
        with open(export_path, 'w') as f:
            for tab in tabs:
                f.write(f"{tab.url}\n")
        console.print(f"[green]✓ Exported to {export_path}[/green]")


@app.command()
def query(question: str):
    """Ask questions about your tab collection"""
    db = Database()
    tabs = db.get_all_tabs()
    
    if not client:
        console.print("[red]Error: GEMINI_API_KEY not set[/red]")
        return
    
    # Prepare context
    context = "\n\n".join([
        f"Title: {tab.title}\nURL: {tab.url}\nSummary: {tab.summary or 'N/A'}"
        for tab in tabs[:50]  # Limit to avoid token limits
    ])
    
    prompt = f"""Based on this user's browser tab collection, answer their question:

Question: {question}

Tab Collection:
{context}

Provide a helpful answer based on the tabs."""

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )
        console.print(Panel(Markdown(response.text), title="Answer", border_style="green"))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def stats():
    """Show statistics about your tabs"""
    db = Database()
    stats = db.get_stats()
    
    table = Table(title="Tab Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="magenta")
    
    for key, value in stats.items():
        table.add_row(key.capitalize(), str(value))
    
    console.print(table)


if __name__ == "__main__":
    app()