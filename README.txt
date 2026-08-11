MARGINALIA — PRIVATE LOCAL RESEARCH LIBRARY

VERSION AND PLATFORM

This is Marginalia v0.1.0-beta.6. One transparent source package supports
Windows 10/11, Linux and macOS. Platform launchers share the same application
code stored in the marginalia subfolder.

QUICK START

1. Use Start Marginalia.cmd on Windows, Start Marginalia.sh on Linux, or
   Start Marginalia.command on macOS.
2. If Python 3.12 is missing, the launcher explains the platform-specific
   installation. Windows uses winget only after confirmation; macOS may use
   Homebrew only after confirmation; Linux leaves system packages to the user.
3. On the first run, the script creates .venv and installs the two pinned Python
   packages shown in requirements.txt.
4. Choose a Zotero data folder, an Obsidian vault, or both.
5. Click “Copy and build my library”. Keep the command window open while using the app.

The script reuses an existing Python 3.12 installation and never installs a second
copy when a suitable version is available. The .venv directory is an isolated
environment for Marginalia's pinned packages. Existing valid .venv environments are
reused. The command always runs the visible source files, so code changes are
auditable and visible after restarting.

To share Marginalia, send the release ZIP without private databases,
source_snapshots or .venv. The recipient needs internet access on the first run.
Python 3.12 is reused when present. After setup, ordinary BM25 use works offline.

SOURCE SAFETY AND REFRESHING

Marginalia never edits the selected folders. A refresh builds a new private snapshot
and new databases as a generation. The previous searchable generation remains intact
until every required database has passed integrity checks. Interrupted refreshes are
remembered and safely restarted the next time Marginalia opens.

Unchanged files can be reused from the previous private generation. Obsidian metadata
directories such as .obsidian, .git and .trash are not copied. After a successful
publication, superseded source generations are removed.

Zotero can normally remain open. Marginalia asks you to close it only if Zotero is
actively preventing SQLite from making a consistent backup.

Linked Zotero attachments are copied into the same private generation. Marginalia
tries to read Zotero's Linked Attachment Base Directory automatically; it can also be
selected explicitly in Source folders. Source folders may not contain Marginalia's
managed snapshot directory.

AI FEATURES (OPTIONAL)

BM25 keyword and exact-phrase search require no additional software. Semantic search
uses a dedicated local Ollama service and an embedding model. Generated answers use
the main Ollama service and a completion model. Either feature can work without the
other; unavailable semantic retrieval falls back to BM25, and an unavailable answer
model does not suppress search results.

Before semantic indexing, Marginalia offers an optional AI setup screen. It recommends
Qwen3 Embedding 0.6B, optionally offers Gemma 4 12B for generated answers, and always
allows setup to be skipped in favor of BM25. Downloads and semantic indexing show
progress. Models can be added later from AI models in the sidebar or from the final
entry in either model selector.
Every user-selected embedding model gets its own semantic index. After a library
refresh, Marginalia rebuilds every embedding index that was previously ready.
The current model and remaining model queue are saved during indexing, so restarting
the computer resumes the exact unfinished queue. When AI has been enabled, Marginalia
reuses an existing Ollama service or starts `ollama serve` itself if Ollama is not
already running. Ollama is not started automatically for BM25-only users.

PRIVACY AND STORAGE

Library text and model processing stay on this computer. Saved searches can contain
passages and generated answers. Marginalia retains at most 100 searches for 180 days;
“Clear all history” deletes them immediately and compacts the history database.

When a source changes, the setup screen warns that saved history can contain text from
the previous source. Stale semantic records are securely deleted after a successful
semantic refresh. If semantic indexing is unavailable, they remain hidden until a
later successful refresh can clean them.

LICENSE AND SOURCE CODE

Copyright (C) 2026 Marginalia contributors.

Marginalia is free software licensed under the GNU Affero General Public License,
version 3 only (SPDX: AGPL-3.0-only). You may use, study, share and modify it under
the terms in LICENSE. Marginalia is provided without warranty. The preferred source
for modification is https://github.com/K-K-Szostak/marginalia-local-search.
Third-party components retain their own licenses; see THIRD-PARTY-NOTICES.txt.

TROUBLESHOOTING

- Marginalia opens on an installation-specific localhost address.
- It never redirects to another Marginalia installation when a port is occupied.
- Detailed refresh output is written to refresh.log.
- The shared archive includes a Windows Tesseract OCR runtime. Linux and macOS use a
  local Tesseract installation available in PATH. Additional OCR languages are
  managed by that platform's Tesseract installation.
- The local Git history provides restore points for development changes.
