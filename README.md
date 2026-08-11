# Marginalia

Marginalia is a private local search application for Zotero and Obsidian. It
creates searchable working copies and never edits the original libraries.

> **Supported release:** Windows 10/11 x64. macOS and Linux are not yet supported.

## Download and start

1. Open the [Windows beta Release](https://github.com/K-K-Szostak/marginalia-local-search/releases/tag/v0.1.0-beta.2).
2. Download **`Marginalia-Windows-v0.1.0-beta.2.zip`**. Do not download the
   automatically generated “Source code” archives.
3. Right-click the ZIP file, select **Extract All**, and open the extracted folder.
4. Double-click **`Start Marginalia.cmd`** and keep its command window open.
5. If Python 3.12 is missing, approve its official installation when Marginalia
   asks. Nothing is installed without confirmation.
6. Choose a Zotero data folder, an Obsidian vault, or both, then select
   **Copy and build my library**.

The first start needs internet access. Marginalia creates an isolated `.venv`
inside its own folder and downloads its two pinned Python dependencies there.

For a fuller nontechnical walkthrough, open **`START HERE.txt`** inside the
Windows package.

## AI is optional

BM25 keyword search does not need Ollama or any AI model. During setup you can:

- skip AI and use BM25;
- install Qwen3 Embedding 0.6B for semantic search;
- optionally install Gemma 4 12B for locally generated answers.

AI components can be installed later from **AI models** in the application.

## Privacy

Documents, indexes and model processing stay on the user's computer. Marginalia
works on private copies and does not modify the selected Zotero or Obsidian folders.

## Development

Requirements: Windows, Python 3.12, NumPy 2.3.5 and PyMuPDF 1.28.2.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

## License

Copyright © 2026 Marginalia contributors.

Marginalia is licensed under the GNU Affero General Public License version 3
only (`AGPL-3.0-only`) and is provided without warranty. See [LICENSE](LICENSE)
and [THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt).
