from __future__ import annotations

import json
import io
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import app.server as server
import build_semantic_index
import import_obsidian
import refresh_manager
import semantic_models
import source_manager


class AttachmentPathTests(unittest.TestCase):
    def test_managed_snapshot_attachment_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            attachment = base / "source_snapshots" / "generations" / "one" / "zotero" / "storage" / "A" / "paper.pdf"
            attachment.parent.mkdir(parents=True)
            attachment.write_bytes(b"pdf")
            with patch.object(server, "BASE", base):
                self.assertEqual(server.allowed_attachment_path(attachment.relative_to(base)), attachment.resolve())

    def test_path_outside_managed_copies_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            outside = base / "private.pdf"
            outside.write_bytes(b"pdf")
            with patch.object(server, "BASE", base):
                self.assertIsNone(server.allowed_attachment_path("private.pdf"))


class AnswerFallbackTests(unittest.TestCase):
    def test_unavailable_answer_service_does_not_disable_retrieval(self):
        with patch.object(server, "selected_answer_model", side_effect=ConnectionError("offline")):
            model, enabled, warning = server.answer_request_status("answer:model", True)
        self.assertEqual(model, "answer:model")
        self.assertFalse(enabled)
        self.assertIn("Search results are still available", warning)


class RefreshResumeTests(unittest.TestCase):
    def test_interrupted_publication_keeps_the_semantic_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path=Path(directory)/"state.json"
            state_path.write_text(json.dumps({
                "running":True,"phase":"publishing","resume_phase":"semantic",
                "semantic_queue":["embed-one:latest","embed-two:latest"],
                "semantic_model":"embed-one:latest",
            }),encoding="utf-8")
            original=dict(refresh_manager.STATE)
            try:
                refresh_manager.STATE.update(running=False,phase="idle",resume_phase="",
                  resume_required=False,semantic_queue=[],semantic_model="")
                with patch.object(refresh_manager,"STATE_PATH",state_path):
                    refresh_manager._restore_state()
                self.assertTrue(refresh_manager.STATE["resume_required"])
                self.assertEqual(refresh_manager.STATE["resume_phase"],"semantic")
                self.assertEqual(refresh_manager.STATE["semantic_queue"],
                                 ["embed-one:latest","embed-two:latest"])
            finally:
                refresh_manager.STATE.clear(); refresh_manager.STATE.update(original)

    def test_semantic_resume_finishes_the_persisted_model_queue(self):
        config={"zotero_path":"source","obsidian_path":""}
        original=dict(refresh_manager.STATE)
        try:
            refresh_manager.STATE.update(
                semantic_queue=["first-embed:latest","second-embed:latest"],
                semantic_model="first-embed:latest",semantic_completed=[],warnings=[],
                semantic_model_total=2,running=True,
            )
            with tempfile.TemporaryDirectory() as directory, \
                 patch.object(refresh_manager,"STATE_PATH",Path(directory)/"state.json"), \
                 patch.object(refresh_manager,"semantic_preflight",return_value=""), \
                 patch.object(refresh_manager,"run_stage") as run:
                refresh_manager._run_semantic_resume(config)
            self.assertEqual([call.args[3][1] for call in run.call_args_list],
                             ["first-embed:latest","second-embed:latest"])
            self.assertEqual(refresh_manager.STATE["semantic_queue"],[])
            self.assertEqual(refresh_manager.STATE["semantic_completed"],
                             ["first-embed:latest","second-embed:latest"])
        finally:
            refresh_manager.STATE.clear(); refresh_manager.STATE.update(original)

    def test_completed_semantic_models_are_remembered_before_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); model_paths={}
            for model,name in (("qwen3-embedding:0.6b","small.sqlite"),("qwen3-embedding:8b","large.sqlite"),("snowflake-arctic-embed2:latest","custom.sqlite")):
                path=root/name; database=sqlite3.connect(path)
                database.execute("CREATE TABLE semantic_index_info(key TEXT PRIMARY KEY,value TEXT)")
                database.executemany("INSERT INTO semantic_index_info VALUES(?,?)",(
                    ("model",model),("completed_at","2026-08-11T12:00:00Z")))
                database.commit(); database.close(); model_paths[model]=path
            with patch.object(refresh_manager,"SEMANTIC_MODELS",model_paths):
                self.assertEqual(set(refresh_manager.completed_semantic_models()),set(model_paths))
                refresh_manager.invalidate_semantic_indexes(model_paths.values())
                self.assertEqual(refresh_manager.completed_semantic_models(),[])

    def test_semantic_resume_skips_snapshot_and_core_stages(self):
        config = {"zotero_path": "source", "obsidian_path": ""}
        with patch.object(refresh_manager, "load_config", return_value=config), \
             patch.object(refresh_manager, "recover_interrupted_publication"), \
             patch.object(refresh_manager, "active_generation_matches", return_value=True), \
             patch.object(refresh_manager, "_run_semantic_resume") as resume, \
             patch.object(refresh_manager, "snapshot_sources") as snapshot:
            refresh_manager._run(resume_semantic=True)
        resume.assert_called_once_with(config, None)
        snapshot.assert_not_called()

    def test_embedding_installer_rejects_invalid_model_name(self):
        started, error = server.start_embedding_model_install("not a model")
        self.assertFalse(started)
        self.assertIn("Invalid", error)

    def test_embedding_installer_accepts_user_selected_model(self):
        model="snowflake-arctic-embed2:latest"
        original_install=dict(server.MODEL_INSTALL)
        try:
            with patch.object(server,"embedding_ollama_executable",return_value=Path("ollama.exe")), \
                 patch.object(server,"register_embedding_model") as register, \
                 patch.object(server.threading,"Thread") as thread:
                started,error=server.start_embedding_model_install(model)
            self.assertTrue(started); self.assertEqual(error,"")
            register.assert_called_once_with(model)
            self.assertEqual(server.EMBED_MODELS[model],semantic_models.index_path(model))
            thread.return_value.start.assert_called_once()
        finally:
            server.EMBED_MODELS.pop(model,None)
            server.MODEL_INSTALL.clear(); server.MODEL_INSTALL.update(original_install)

    def test_custom_embedding_model_has_stable_independent_paths(self):
        first="snowflake-arctic-embed2:latest"; second="mxbai-embed-large:latest"
        self.assertEqual(semantic_models.index_path(first),semantic_models.index_path(first))
        self.assertNotEqual(semantic_models.index_path(first),semantic_models.index_path(second))
        self.assertNotEqual(semantic_models.progress_path(first),semantic_models.progress_path(second))

    def test_refresh_rebuilds_every_previously_ready_custom_index(self):
        stage=("semantic","Updating semantic search","build_semantic_index.py",[])
        models=["qwen3-embedding:0.6b","qwen3-embedding:8b","snowflake-arctic-embed2:latest"]
        with patch.object(refresh_manager,"semantic_preflight",return_value=""), \
             patch.object(refresh_manager,"run_stage") as run:
            refresh_manager.rebuild_additional_semantic_models(
                stage,models,"qwen3-embedding:0.6b",9,9,{"snapshot_root":"copy"})
        rebuilt=[call.args[3][1] for call in run.call_args_list]
        self.assertEqual(rebuilt,["qwen3-embedding:8b","snowflake-arctic-embed2:latest"])

    def test_semantic_queue_runs_each_model_once_without_false_warning(self):
        stage=("semantic","Updating semantic search","build_semantic_index.py",[])
        original = dict(refresh_manager.STATE)
        try:
            refresh_manager.STATE.update(warnings=[],semantic_completed=[],semantic_model_total=0)
            with tempfile.TemporaryDirectory() as directory, \
                 patch.object(refresh_manager,"STATE_PATH",Path(directory)/"state.json"), \
                 patch.object(refresh_manager,"semantic_preflight",return_value=""), \
                 patch.object(refresh_manager,"run_stage") as run:
                refresh_manager._run_semantic_queue(
                    {},stage,["qwen3-embedding:0.6b"],9,9,None)
            run.assert_called_once_with(
                "semantic","Updating semantic search (qwen3-embedding:0.6b)",
                "build_semantic_index.py",["--model","qwen3-embedding:0.6b"],
                9,9,snapshot=None)
            self.assertEqual(refresh_manager.STATE["warnings"],[])
            self.assertEqual(refresh_manager.STATE["semantic_completed"],["qwen3-embedding:0.6b"])
        finally:
            refresh_manager.STATE.clear(); refresh_manager.STATE.update(original)

    def test_semantic_resume_passes_selected_model_to_indexer(self):
        config = {"zotero_path": "source", "obsidian_path": ""}
        with patch.object(refresh_manager, "semantic_preflight", return_value="") as preflight, \
             patch.object(refresh_manager, "run_stage") as run, \
             patch.object(refresh_manager, "update"), patch.object(refresh_manager, "update_activity"):
            refresh_manager._run_semantic_resume(config, "qwen3-embedding:8b")
        preflight.assert_called_once_with("qwen3-embedding:8b")
        self.assertEqual(run.call_args.args[3], ["--model", "qwen3-embedding:8b"])

    def test_skip_ai_is_persisted_for_the_current_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            original = dict(refresh_manager.STATE)
            try:
                refresh_manager.STATE.update(running=True, phase="ai_setup", ai_setup_required=True,
                                             ai_setup_skipped=False)
                with patch.object(refresh_manager, "STATE_PATH", Path(directory) / "state.json"):
                    self.assertTrue(refresh_manager.skip_semantic_setup())
                self.assertTrue(refresh_manager.STATE["ai_setup_skipped"])
                self.assertFalse(refresh_manager.STATE["ai_setup_required"])
            finally:
                refresh_manager.STATE.clear(); refresh_manager.STATE.update(original)

    def test_user_selected_embedding_model_releases_ai_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            original=dict(refresh_manager.STATE)
            try:
                refresh_manager.STATE.update(running=True,phase="ai_setup",ai_setup_required=True,
                  ai_setup_skipped=False,ai_setup_model="qwen3-embedding:0.6b")
                with patch.object(refresh_manager,"STATE_PATH",Path(directory)/"state.json"), \
                     patch.object(refresh_manager,"semantic_preflight",return_value=""), \
                     patch.object(refresh_manager,"set_active_embedding_model"):
                    self.assertTrue(refresh_manager.select_semantic_setup_model("test-embed:latest"))
                self.assertEqual(refresh_manager.STATE["ai_setup_model"],"test-embed:latest")
                self.assertFalse(refresh_manager.STATE["ai_setup_required"])
            finally:
                refresh_manager.AI_SETUP_EVENT.clear()
                refresh_manager.STATE.clear(); refresh_manager.STATE.update(original)


class SemanticRuntimeTests(unittest.TestCase):
    def test_model_identity_changes_chunk_hash(self):
        original=(build_semantic_index.EMBED_MODEL,build_semantic_index.MODEL_IDENTITY)
        try:
            build_semantic_index.EMBED_MODEL="embed:test"
            build_semantic_index.MODEL_IDENTITY="digest-one"
            first=build_semantic_index.chunk("one","metadata","Metadata",1,None,None,"Title","Text")
            build_semantic_index.MODEL_IDENTITY="digest-two"
            second=build_semantic_index.chunk("one","metadata","Metadata",1,None,None,"Title","Text")
            self.assertNotEqual(first["content_hash"],second["content_hash"])
        finally:
            build_semantic_index.EMBED_MODEL,build_semantic_index.MODEL_IDENTITY=original

    def test_updated_model_digest_marks_index_not_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); semantic=root/"semantic.sqlite"; clean=root/"clean.sqlite"
            db=sqlite3.connect(semantic); db.executescript(build_semantic_index.SCHEMA)
            db.execute("INSERT INTO semantic_chunks(chunk_key,content_hash,model,dimensions,kind,label,text,embedding) VALUES(?,?,?,?,?,?,?,?)",
              ("one","hash","embed:test",2,"metadata","Metadata","Text",np.zeros(2,dtype=np.float32).tobytes()))
            db.executemany("INSERT INTO semantic_index_info VALUES(?,?)",(
              ("model","embed:test"),("model_identity","old-digest"),("completed_at","now"),
              ("clean_text_completed_at","clean-now")))
            db.commit(); db.close()
            db=sqlite3.connect(clean); db.execute("CREATE TABLE clean_text_info(key TEXT PRIMARY KEY,value TEXT)")
            db.execute("INSERT INTO clean_text_info VALUES('completed_at','clean-now')"); db.commit(); db.close()
            with patch.object(server,"semantic_db_path",return_value=semantic),patch.object(server,"CLEAN_DB",clean):
                status=server.semantic_index_status("embed:test","new-digest")
            self.assertFalse(status["ready"]); self.assertFalse(status["model_current"])

    def test_existing_answer_ollama_is_reused(self):
        with patch.object(server, "ollama_names", return_value={"gemma4:12b"}), \
             patch.object(server.subprocess, "Popen") as popen:
            self.assertTrue(server.ensure_answer_ollama())
        popen.assert_not_called()

    def test_bulk_build_service_is_separate_from_answer_and_query_services(self):
        with patch.dict(os.environ, {
            "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
            "EMBED_OLLAMA_BASE_URL": "http://127.0.0.1:11435",
            "EMBED_BUILD_OLLAMA_BASE_URL": "http://127.0.0.1:11436",
        }, clear=False):
            candidates = refresh_manager.semantic_service_candidates()
        self.assertEqual(candidates, [
            ("gpu-preferred", "http://127.0.0.1:11436"),
            ("cpu-fallback", "http://127.0.0.1:11435"),
        ])

    def test_bulk_build_prefers_gpu_service_and_falls_back_to_query_cpu(self):
        model = "qwen3-embedding:0.6b"
        with patch.dict(os.environ, {
            "EMBED_BUILD_OLLAMA_BASE_URL": "http://127.0.0.1:11436",
            "EMBED_OLLAMA_BASE_URL": "http://127.0.0.1:11435",
        }, clear=False), patch.object(refresh_manager, "_ollama_models", return_value={model}):
            self.assertEqual(refresh_manager.choose_semantic_service(model)["role"], "gpu-preferred")

        def models(base_url, timeout=3):
            if base_url == "http://127.0.0.1:11436":
                raise ConnectionError("offline")
            return {model}

        with patch.dict(os.environ, {
            "EMBED_BUILD_OLLAMA_BASE_URL": "http://127.0.0.1:11436",
            "EMBED_OLLAMA_BASE_URL": "http://127.0.0.1:11435",
        }, clear=False), patch.object(refresh_manager, "_ollama_models", side_effect=models):
            selected = refresh_manager.choose_semantic_service(model)
        self.assertEqual(selected["role"], "cpu-fallback")
        self.assertEqual(selected["base_url"], "http://127.0.0.1:11435")

    def test_runtime_uses_ollama_vram_report_not_service_assumption(self):
        gpu = io.BytesIO(json.dumps({"models": [{
            "name": build_semantic_index.EMBED_MODEL, "size": 1000, "size_vram": 1000,
        }]}).encode())
        cpu = io.BytesIO(json.dumps({"models": [{
            "name": build_semantic_index.EMBED_MODEL, "size": 1000, "size_vram": 0,
        }]}).encode())
        with patch.object(build_semantic_index.urllib.request, "urlopen", return_value=gpu):
            self.assertEqual(build_semantic_index.embedding_runtime()["device"], "GPU")
        with patch.object(build_semantic_index.urllib.request, "urlopen", return_value=cpu):
            self.assertEqual(build_semantic_index.embedding_runtime()["device"], "CPU")

    def test_semantic_scoring_keeps_only_bounded_best_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "semantic.sqlite"
            db = sqlite3.connect(database)
            db.executescript(build_semantic_index.SCHEMA)
            for index in range(20):
                vector = np.asarray([float(index + 1), 1.0], dtype=np.float32)
                db.execute("""INSERT INTO semantic_chunks(
                  chunk_key,content_hash,model,dimensions,kind,label,title,text,embedding)
                  VALUES(?,?,?,?,?,?,?,?,?)""",(str(index),str(index),"qwen3-embedding:0.6b",2,
                  "document","PDF",f"Title {index}",f"Text {index}",vector.tobytes()))
            db.commit(); db.close()
            query=np.asarray([1.0,0.0],dtype=np.float32)
            with patch.object(server,"semantic_db_path",return_value=database):
                ranked=server.semantic_ranked_candidates("qwen3-embedding:0.6b",query,
                  {"document"},None,None,3)
            self.assertEqual(len(ranked),3)
            self.assertGreaterEqual(ranked[0][1],ranked[-1][1])


class ObsidianAssetTests(unittest.TestCase):
    def test_duplicate_filename_resolves_relative_to_open_note(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "obsidian.sqlite"
            db = sqlite3.connect(database)
            db.execute("CREATE TABLE obsidian_assets(relative_path TEXT PRIMARY KEY,filename TEXT,content_type TEXT,byte_size INTEGER,modified_at TEXT)")
            db.executemany("INSERT INTO obsidian_assets VALUES(?,?,?,?,?)", (
                ("First/image.png", "image.png", "image/png", 1, ""),
                ("Second/image.png", "image.png", "image/png", 1, ""),
            ))
            db.commit(); db.close()
            with patch.object(server, "OBSIDIAN_DB", database):
                record, resolution = server.obsidian_asset_record("image.png", "Second/note.md")
                missing, ambiguous = server.obsidian_asset_record("image.png", "")
            self.assertEqual(record[0], "Second/image.png")
            self.assertEqual(resolution, "exact")
            self.assertIsNone(missing)
            self.assertEqual(ambiguous, "ambiguous")


class SnapshotGenerationTests(unittest.TestCase):
    def test_source_containing_snapshot_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); vault=root/"vault"; managed=vault/"source_snapshots"
            vault.mkdir()
            with patch.object(source_manager,"SNAPSHOT_ROOT",managed), \
                 patch.object(source_manager,"CONFIG_PATH",root/"config.json"):
                with self.assertRaisesRegex(ValueError,"managed data"):
                    source_manager.save_config(obsidian_path=str(vault))

    def test_windows_snapshot_paths_use_extended_path_prefix(self):
        path = Path(r"C:\library\source_snapshots\generations\one\zotero\storage\A\paper.pdf")
        value = source_manager.filesystem_path(path)
        if os.name == "nt":
            self.assertTrue(value.startswith("\\\\?\\"))
        else:
            self.assertEqual(value, str(path.absolute()))

    @unittest.skipUnless(os.name == "nt", "Windows extended paths only")
    def test_extended_path_can_be_read_and_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generations = root / "generations"
            generation = generations / "generation"
            long_parent = generation / "obsidian" / ("a" * 100) / ("b" * 100)
            os.makedirs(source_manager.filesystem_path(long_parent), exist_ok=True)
            note = long_parent / (("c" * 80) + ".md")
            with open(source_manager.filesystem_path(note), "w", encoding="utf-8") as stream:
                stream.write("# Long note")
            self.assertGreater(len(str(note)), 260)
            self.assertEqual(import_obsidian.read_text(note), "# Long note")
            self.assertEqual(import_obsidian.file_stat(note).st_size, len("# Long note"))
            with patch.object(source_manager, "SNAPSHOT_GENERATIONS", generations):
                source_manager.discard_snapshot_generation({"snapshot_root": str(generation)})
            self.assertFalse(os.path.exists(source_manager.filesystem_path(generation)))

    def test_snapshot_skips_source_file_that_disappears_during_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            changing = source / ".zotero-ft-cache"
            changing.write_text("temporary", encoding="utf-8")

            def disappear(source_path, destination_path):
                Path(source_path).unlink()
                raise FileNotFoundError(2, "source changed", str(source_path))

            with patch.object(source_manager.shutil, "copy2", side_effect=disappear):
                result = source_manager.mirror_tree(source, target)

            self.assertEqual(result["skipped"], 1)
            self.assertFalse((target / changing.name).exists())

    def test_obsidian_snapshot_is_built_in_reserved_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vault = root / "vault"
            vault.mkdir(); (vault / "note.md").write_text("# Note")
            (vault / ".obsidian").mkdir(); (vault / ".obsidian" / "workspace").write_text("private")
            managed = root / "managed"
            generations = managed / "generations"
            config = {"zotero_path": "", "obsidian_path": str(vault)}
            with patch.object(source_manager, "ROOT", root), patch.object(source_manager, "SNAPSHOT_ROOT", managed), patch.object(source_manager, "SNAPSHOT_GENERATIONS", generations):
                generation = source_manager.reserve_snapshot_generation()
                result = source_manager.snapshot_sources(generation_root=generation, config=config)
            self.assertEqual(Path(result["snapshot_root"]), generation)
            self.assertTrue((generation / "obsidian" / "note.md").is_file())
            self.assertFalse((generation / "obsidian" / ".obsidian").exists())

    def test_linked_zotero_attachment_is_copied_into_private_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); zotero=root/"zotero"; zotero.mkdir()
            linked=root/"outside.pdf"; linked.write_bytes(b"private copy")
            db=sqlite3.connect(zotero/"zotero.sqlite")
            db.executescript("CREATE TABLE items(itemID INTEGER PRIMARY KEY,key TEXT); CREATE TABLE itemAttachments(itemID INTEGER,path TEXT,linkMode INTEGER);")
            db.execute("INSERT INTO items VALUES(1,'ABC123')")
            db.execute("INSERT INTO itemAttachments VALUES(1,?,2)",(str(linked),)); db.commit(); db.close()
            managed=root/"managed"; generations=managed/"generations"
            config={"zotero_path":str(zotero),"obsidian_path":""}
            with patch.object(source_manager,"ROOT",root),patch.object(source_manager,"SNAPSHOT_ROOT",managed),patch.object(source_manager,"SNAPSHOT_GENERATIONS",generations):
                generation=source_manager.reserve_snapshot_generation()
                source_manager.snapshot_sources(generation_root=generation,config=config)
            copied=generation/"zotero"/"linked_attachments"/"ABC123"/"outside.pdf"
            mapping=json.loads((generation/"zotero"/"linked_attachment_map.json").read_text(encoding="utf-8"))
            self.assertEqual(copied.read_bytes(),b"private copy")
            self.assertEqual(mapping["1"],"linked_attachments/ABC123/outside.pdf")

    def test_relative_linked_attachment_uses_configured_base_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); zotero=root/"zotero"; zotero.mkdir()
            linked_base=root/"linked"; (linked_base/"papers").mkdir(parents=True)
            (linked_base/"papers"/"article.pdf").write_bytes(b"relative linked file")
            db=sqlite3.connect(zotero/"zotero.sqlite")
            db.executescript("CREATE TABLE items(itemID INTEGER PRIMARY KEY,key TEXT); CREATE TABLE itemAttachments(itemID INTEGER,path TEXT,linkMode INTEGER);")
            db.execute("INSERT INTO items VALUES(1,'REL123')")
            db.execute("INSERT INTO itemAttachments VALUES(1,'attachments:papers/article.pdf',2)")
            db.commit(); db.close()
            managed=root/"managed"; generations=managed/"generations"
            config={"zotero_path":str(zotero),"obsidian_path":"",
                    "linked_attachment_base_path":str(linked_base)}
            with patch.object(source_manager,"ROOT",root),patch.object(source_manager,"SNAPSHOT_ROOT",managed),patch.object(source_manager,"SNAPSHOT_GENERATIONS",generations):
                generation=source_manager.reserve_snapshot_generation()
                source_manager.snapshot_sources(generation_root=generation,config=config)
            self.assertEqual((generation/"zotero"/"linked_attachments"/"REL123"/"article.pdf").read_bytes(),
                             b"relative linked file")


    def test_seeded_generation_breaks_hardlink_before_replacing_changed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / "previous"
            target = root / "target"
            source = root / "source"
            previous.mkdir(); source.mkdir()
            (previous / "paper.pdf").write_text("old", encoding="utf-8")
            source_manager._seed_tree(previous, target)
            (source / "paper.pdf").write_text("new and longer", encoding="utf-8")
            source_manager.mirror_tree(source, target)
            self.assertEqual((previous / "paper.pdf").read_text(encoding="utf-8"), "old")
            self.assertEqual((target / "paper.pdf").read_text(encoding="utf-8"), "new and longer")

    def test_obsidian_metadata_directories_are_pruned_before_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir(); (root / ".git" / "secret").write_text("x")
            (root / ".obsidian").mkdir(); (root / ".obsidian" / "workspace").write_text("x")
            (root / "Notes").mkdir(); (root / "Notes" / "kept.md").write_text("ok")
            files = source_manager.source_files(root, source_manager.OBSIDIAN_IGNORED_DIRECTORIES)
            self.assertEqual([path.relative_to(root).as_posix() for path in files], ["Notes/kept.md"])


class PublicationRecoveryTests(unittest.TestCase):
    @staticmethod
    def _database(path, statements):
        db = sqlite3.connect(path)
        db.executescript(statements)
        db.commit(); db.close()

    def test_complete_work_generation_is_published_with_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active_library = root / "library.sqlite"
            active_clean = root / "clean.sqlite"
            active_obsidian = root / "obsidian.sqlite"
            work_library = root / "library.next.sqlite"
            work_clean = root / "clean.next.sqlite"
            self._database(active_library, "CREATE TABLE old_marker(value TEXT); INSERT INTO old_marker VALUES('old');")
            self._database(active_clean, "CREATE TABLE old_marker(value TEXT);")
            self._database(work_library, "CREATE TABLE items(id); CREATE TABLE attachments(id); CREATE TABLE document_pages(id); CREATE TABLE document_search(id); CREATE TABLE item_search(id); CREATE TABLE new_marker(value TEXT); INSERT INTO new_marker VALUES('new');")
            self._database(work_clean, "CREATE TABLE clean_document_blocks(id); CREATE TABLE clean_document_search(id); CREATE TABLE clean_text_info(key TEXT,value TEXT); INSERT INTO clean_text_info VALUES('completed_at','now');")
            journal = root / "generation_publish.json"
            manifest = root / "library_generation.json"
            active = {"library": active_library, "clean": active_clean, "obsidian": active_obsidian}
            work = {"library": work_library, "clean": work_clean, "obsidian": root / "obsidian.next.sqlite"}
            with patch.object(refresh_manager, "PUBLISH_JOURNAL", journal), patch.object(refresh_manager, "GENERATION_MANIFEST", manifest), patch.object(refresh_manager, "ACTIVE_DATABASES", active), patch.object(refresh_manager, "WORK_DATABASES", work):
                published = refresh_manager.publish_work_generation({"zotero_path": "source", "obsidian_path": ""})
            self.assertEqual(set(published), {"library.sqlite", "clean.sqlite"})
            db = sqlite3.connect(active_library)
            self.assertEqual(db.execute("SELECT value FROM new_marker").fetchone()[0], "new")
            db.close()
            self.assertEqual(json.loads(manifest.read_text())["zotero_path"], "source")
            self.assertFalse(journal.exists())

    def test_publishing_phase_restores_database_and_manifest_backups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active = root / "active.sqlite"
            backup = root / "active.previous.sqlite"
            work = root / "active.next.sqlite"
            manifest = root / "library_generation.json"
            manifest_backup = root / "library_generation.previous.json"
            manifest_new = root / "library_generation.json.new"
            journal = root / "generation_publish.json"
            active.write_text("new database")
            backup.write_text("old database")
            manifest.write_text("new manifest")
            manifest_backup.write_text("old manifest")
            journal.write_text(json.dumps({
                "phase": "publishing",
                "entries": [{"active": str(active), "backup": str(backup), "work": str(work), "had_active": True}],
                "manifest": str(manifest), "manifest_backup": str(manifest_backup),
                "manifest_new": str(manifest_new), "had_manifest": True,
            }))
            with patch.object(refresh_manager, "PUBLISH_JOURNAL", journal), patch.object(refresh_manager, "GENERATION_MANIFEST", manifest):
                self.assertTrue(refresh_manager.recover_interrupted_publication())
            self.assertEqual(active.read_text(), "old database")
            self.assertEqual(manifest.read_text(), "old manifest")
            self.assertFalse(journal.exists())

    def test_damaged_journal_restores_known_backups_and_is_quarantined(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active = root / "library.sqlite"
            backup = root / "library.previous.sqlite"
            manifest = root / "library_generation.json"
            manifest_backup = root / "library_generation.previous.json"
            journal = root / "generation_publish.json"
            active.write_text("partial")
            backup.write_text("complete")
            manifest.write_text("partial manifest")
            manifest_backup.write_text("complete manifest")
            journal.write_text("not json")
            with patch.object(refresh_manager, "PUBLISH_JOURNAL", journal), patch.object(refresh_manager, "GENERATION_MANIFEST", manifest), patch.object(refresh_manager, "ACTIVE_DATABASES", {"library": active}):
                self.assertTrue(refresh_manager.recover_interrupted_publication())
            self.assertEqual(active.read_text(), "complete")
            self.assertEqual(manifest.read_text(), "complete manifest")
            self.assertFalse(journal.exists())
            self.assertEqual(len(list(root.glob("generation_publish.damaged-*.json"))), 1)

    def test_committed_phase_keeps_new_database(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active = root / "active.sqlite"
            backup = root / "active.previous.sqlite"
            journal = root / "generation_publish.json"
            active.write_text("new database")
            backup.write_text("old database")
            journal.write_text(json.dumps({
                "phase": "committed",
                "entries": [{"active": str(active), "backup": str(backup), "work": str(root / "missing.next.sqlite"), "had_active": True}],
            }))
            with patch.object(refresh_manager, "PUBLISH_JOURNAL", journal):
                self.assertTrue(refresh_manager.recover_interrupted_publication())
            self.assertEqual(active.read_text(), "new database")
            self.assertFalse(backup.exists())


class HistoryRetentionTests(unittest.TestCase):
    def test_history_is_limited_to_newest_entries(self):
        db = sqlite3.connect(":memory:")
        db.execute("CREATE TABLE search_history(id TEXT PRIMARY KEY,updated_at TEXT NOT NULL)")
        db.executemany("INSERT INTO search_history VALUES(?,?)", ((str(index), f"2099-01-{index + 1:02d}T00:00:00Z") for index in range(4)))
        with patch.object(server, "HISTORY_MAX_ENTRIES", 2), patch.object(server, "HISTORY_RETENTION_DAYS", 10000):
            server.prune_history(db)
        self.assertEqual([row[0] for row in db.execute("SELECT id FROM search_history ORDER BY updated_at")], ["2", "3"])
        db.close()


if __name__ == "__main__":
    unittest.main()
