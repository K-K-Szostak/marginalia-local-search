# Marginalia

Marginalia is a private local search application for Zotero and Obsidian. It
creates searchable working copies and never edits the original libraries.

> **Shared beta package:** Windows 10/11, Linux and macOS use the same auditable source code and one download.

## Download and start

1. Open the [beta Release](https://github.com/K-K-Szostak/marginalia-local-search/releases/tag/v0.1.0-beta.5).
2. Download **`Marginalia-v0.1.0-beta.5.zip`**. Do not download the
   automatically generated “Source code” archives.
3. Extract the ZIP and open the extracted folder. The only files at this level
   are the three launchers and the `marginalia` application folder.
4. Start the launcher for your system and keep its terminal window open:
   - Windows: double-click **`Start Marginalia.cmd`**;
   - Linux: run **`Start Marginalia.sh`**;
   - macOS: double-click **`Start Marginalia.command`**.
5. Marginalia reuses Python 3.12 when available and explains what is missing
   before any optional installation.
6. Choose a Zotero data folder, an Obsidian vault, or both, then select
   **Copy and build my library**.

The first start needs internet access. Marginalia creates an isolated `.venv`
inside its own folder and downloads its two pinned Python dependencies there.

For a fuller walkthrough, open **`marginalia/START HERE.txt`** inside the package.

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

Requirements: Python 3.12, NumPy 2.3.5 and PyMuPDF 1.28.2 on Windows, Linux or macOS.

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
