# Tab Curator

Tab Curator is for when you have hundreds of open tabs and no clear way to get through them. It uses Google's Gemini Flash (configured as 3.0 in this version) to automatically organize your tabs by topic, providing a clean path to finally catching up on your reading list.

## Quick Start

### 1. Installation

Set up the project using `uv` or standard `pip`:

```bash
git clone https://github.com/yourusername/tab-curator.git
cd tab-curator
uv venv && source .venv/bin/activate
uv pip install google-genai beautifulsoup4 requests rich typer sqlalchemy
```

### 2. Configuration

Set your [Gemini API key](https://aistudio.google.com/apikey):

```bash
export GEMINI_API_KEY="your-api-key-here"
```

## Standard Workflow

Follow this sequence to organize your tabs and start catching up.

### 1. Export URLs
Extract open tabs from your browser into a text file. on macOS, use the provided script:
```bash
./export_chrome_tabs.sh > my_tabs.txt
```
*For iPhone, use "Share All Tabs" in Chrome and save the URLs to a text file.*

### 2. Import into Database
Process the text files to deduplicate URLs and fetch page titles/content.
```bash
python tab_curator.py import-tabs my_tabs.txt
```

### 3. Organize with AI
Group your tabs into logical categories. The AI processes tabs in batches of 50 to maintain context without hitting rate limits.
```bash
python tab_curator.py cluster
```

### 4. Explore and Catch Up
Launch the interactive CLI to browse categories, read summaries, and mark tabs as read.
```bash
python tab_curator.py explore
```

### 5. Cleanup
Generate a script to automatically close the tabs you've finished in Chrome.
```bash
python tab_curator.py close-reviewed
osascript exports/close_reviewed_tabs.scpt
```

## Command Reference

### import-tabs
```bash
python tab_curator.py import-tabs my_urls.txt
```
Reads URLs from text files, removes duplicates, and fetches page content for AI analysis.

### cluster
```bash
python tab_curator.py cluster
```
Uses Gemini to group your tabs into logical categories. If you have hundreds of tabs, it processes them in batches to keep things organized.

### explore
```bash
python tab_curator.py explore
```
The main way to catch up. Browse your categories, read AI-generated summaries, and mark items as "read" or "archived."
- **Numbers**: Select category or tab
- **0**: Go back
- **Actions**: Mark status or open in browser

### Advanced Tools

#### query
```bash
python tab_curator.py query "Which tabs mention machine learning?"
```
Ask natural language questions about your collection. AI uses your tab summaries to find answers.

#### export
```bash
python tab_curator.py export --format notebooklm --output notes.txt
```
Save your curated tabs for other apps. Use `notebooklm` for AI study guides, `markdown` for lists, or `json` for data projects.

#### close-reviewed
```bash
python tab_curator.py close-reviewed
```
Generates a script to automatically close tabs in Chrome that you've already marked as read or archived.

## Under the Hood

Tab Curator uses a two-stage process to handle large tab collections:
1. **Initial Grouping**: Tabs are grouped into specific topics in batches of 50.
2. **Consolidation**: A second pass merges similar topics into 8–12 high-level categories.

**Stack**: Built with Python, SQLite, and Google Gemini Flash.

## Project Structure

```text
tab-curator/
├── tab_curator.py         # Main app
├── export_chrome_tabs.sh  # macOS export script
├── data/
│   ├── tabs.db           # Tab database
│   ├── hierarchy.json    # Organized folder structure
│   └── cache/            # Cached page content
└── exports/              # Exported lists and scripts
```

## Use Cases

- **Research and Learning**: Organize papers into a curriculum. Export to NotebookLM for study guides.
- **Caught-up Professionals**: Catalog industry news. Use semantic search to find items saved weeks ago.

## Troubleshooting

### API Configuration
Make sure `GEMINI_API_KEY` is set in your terminal: `echo $GEMINI_API_KEY`.

### Fetching Errors
Some sites block automated scrapers. Tab Curator handles this by categorizing these tabs based on available metadata even if the full content can't be fetched.

---
MIT License.