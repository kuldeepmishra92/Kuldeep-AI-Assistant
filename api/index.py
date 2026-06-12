import os
import sys
import uuid
import time
import shutil
import tempfile
import functools
from pathlib import Path
from datetime import datetime
from flask import (
    Flask, request, jsonify, render_template,
    Response, send_from_directory, abort, make_response,
)
from flask_cors import CORS
from flask_compress import Compress

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
config.validate_config()

from rag.document_loader import load_and_chunk_file
from rag.vector_store    import (
    add_documents, get_document_count, clear_store,
    get_unique_sources, delete_source, source_exists,
    get_source_chunk_count, get_source_preview,
)
from rag.retriever       import invalidate_bm25_cache
from orchestrator.graph  import run_chat
from utils.logger        import get_logger
from utils.knowledge_loader import download_knowledge_from_hf
from keep_alive          import start_keep_alive

logger = get_logger(__name__)

app = Flask(__name__, static_folder="../static", template_folder="../templates")
CORS(app)
Compress(app)

# ── Admin auth ───────────────────────────────────────────────
def require_admin(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        # Check cookie (set after /admin login) or header
        key = (
            request.cookies.get("admin_key")
            or request.headers.get("X-Admin-Key")
            or request.args.get("key")
        )
        if key != config.ADMIN_SECRET_KEY:
            abort(403)
        return f(*args, **kwargs)
    return decorated

# ── Startup: auto-index all knowledge & uploads ─────────────
def auto_index_knowledge() -> None:
    """Index all knowledge + uploaded files. Knowledge files are ALWAYS re-indexed
    (delete old + re-add) so edits are always picked up on restart."""
    knowledge_dir = Path(config.KNOWLEDGE_DIR)
    uploads_dir   = Path(config.UPLOADS_DIR)
    total_indexed = 0

    # ── Permanent knowledge files: FORCE re-index every startup ──
    knowledge_files = [
        f for f in knowledge_dir.iterdir()
        if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTENSIONS
        and not f.name.startswith(".")
    ]
    if knowledge_files:
        logger.info(f"Auto-index [permanent]: re-indexing {len(knowledge_files)} file(s) in {knowledge_dir}")
        for fpath in knowledge_files:
            try:
                # Delete old chunks first so updates are always picked up
                delete_source(fpath.name)
                chunks = load_and_chunk_file(str(fpath), source_name=fpath.name)
                if chunks:
                    added = add_documents(chunks)
                    logger.info(f"    ✓ {fpath.name}: {added} chunks indexed.")
                    total_indexed += added
            except Exception as e:
                logger.error(f"  ✗ Failed to index {fpath.name}: {e}")
    else:
        logger.info(f"Auto-index [permanent]: no files in {knowledge_dir}")

    # ── Uploaded files: skip if already indexed ───────────────
    upload_files = [
        f for f in uploads_dir.iterdir()
        if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTENSIONS
        and not f.name.startswith(".")
    ]
    if upload_files:
        logger.info(f"Auto-index [uploaded]: scanning {len(upload_files)} file(s) in {uploads_dir}")
        for fpath in upload_files:
            if source_exists(fpath.name):
                logger.info(f"    → {fpath.name}: already indexed, skipping.")
                continue
            try:
                chunks = load_and_chunk_file(str(fpath), source_name=fpath.name)
                if chunks:
                    added = add_documents(chunks)
                    logger.info(f"    ✓ {fpath.name}: {added} chunks indexed.")
                    total_indexed += added
            except Exception as e:
                logger.error(f"  ✗ Failed to index {fpath.name}: {e}")
    else:
        logger.info(f"Auto-index [uploaded]: no files in {uploads_dir}")

    if total_indexed > 0:
        invalidate_bm25_cache()
        logger.info(f"Auto-index complete. {total_indexed} chunks added/updated.")
    else:
        logger.info("Auto-index complete. All files up to date.")

# Run at startup: download private knowledge (HF) then index
with app.app_context():
    download_knowledge_from_hf()   # ← pulls from private HF Dataset (no-op locally)
    auto_index_knowledge()

# ── Health check & keep-alive ping ──────────────────────────
@app.route("/ping")
def ping():
    return jsonify({"status": "alive", "service": "Kuldeep AI"}), 200

# ── Public routes ────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/static/<path:path>")
def send_static(path):
    return send_from_directory("../static", path)

@app.route("/api/stats", methods=["GET"])
def get_stats():
    try:
        chunks = get_document_count()
    except Exception:
        chunks = 0
    return jsonify({"total_chunks": chunks})

@app.route("/api/chat", methods=["POST"])
def chat():
    data       = request.json
    query      = data.get("message")
    session_id = data.get("session_id", str(uuid.uuid4())[:8])

    if not query:
        return jsonify({"error": "Message is required"}), 400

    def generate():
        try:
            start   = time.perf_counter()
            result  = run_chat(query=query, session_id=session_id)
            elapsed = round(time.perf_counter() - start, 2)

            # Log routing decision visibly in console
            agent_used = result.get("agent_used", "unknown")
            route      = result.get("route", "unknown")
            logger.info(f"━━━ [{session_id}] Route: {route} → {agent_used} ({elapsed}s) ━━━")

            # Stream words — agent info hidden from public response
            words = result["response"].split(" ")
            for i, w in enumerate(words):
                yield w + (" " if i < len(words) - 1 else "")
                time.sleep(0.005)  # 5ms — 3x faster than before

        except Exception as e:
            logger.error(f"[/api/chat] Error: {e}", exc_info=True)
            yield f"⚠️ Server error: {str(e)}"

    return Response(generate(), mimetype="text/event-stream")


# ── Admin page ────────────────────────────────────────────────
@app.route("/admin")
def admin():
    key = request.args.get("key", "")
    if key != config.ADMIN_SECRET_KEY:
        return render_template("admin_denied.html"), 403
    resp = make_response(render_template("admin.html"))
    # Set session cookie (1 day)
    resp.set_cookie(
        "admin_key", config.ADMIN_SECRET_KEY,
        max_age=86400, httponly=True, samesite="Lax",
    )
    return resp

# ── Admin API: force reindex all ─────────────────────────────
@app.route("/api/admin/reindex-all", methods=["POST"])
@require_admin
def reindex_all():
    try:
        logger.info("Manual reindex triggered via admin API.")
        auto_index_knowledge()
        chunks = get_document_count()
        return jsonify({"status": "ok", "total_chunks": chunks})
    except Exception as e:
        logger.error(f"Reindex failed: {e}")
        return jsonify({"error": str(e)}), 500

# ── Admin API: stats ──────────────────────────────────────────
@app.route("/api/admin/stats", methods=["GET"])
@require_admin
def admin_stats():
    try:
        chunks = get_document_count()
        sources = get_unique_sources()

        # Classify sources into knowledge vs uploads
        knowledge_files = [
            f.name for f in Path(config.KNOWLEDGE_DIR).iterdir()
            if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTENSIONS
        ]
        upload_files = [
            f.name for f in Path(config.UPLOADS_DIR).iterdir()
            if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTENSIONS
        ]

        return jsonify({
            "total_chunks":    chunks,
            "total_docs":      len(sources),
            "knowledge_count": len(knowledge_files),
            "upload_count":    len(upload_files),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Admin API: list documents ─────────────────────────────────
@app.route("/api/documents", methods=["GET"])
@require_admin
def get_documents():
    try:
        sources    = get_unique_sources()
        knowledge_names = {
            f.name for f in Path(config.KNOWLEDGE_DIR).iterdir()
            if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTENSIONS
        }
        upload_names = {
            f.name for f in Path(config.UPLOADS_DIR).iterdir()
            if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTENSIONS
        }

        docs = []
        for src in sources:
            # Determine disk path
            if src in knowledge_names:
                disk_path = Path(config.KNOWLEDGE_DIR) / src
                origin = "knowledge"
            elif src in upload_names:
                disk_path = Path(config.UPLOADS_DIR) / src
                origin = "uploaded"
            else:
                disk_path = None
                origin = "unknown"

            size_bytes = disk_path.stat().st_size if disk_path and disk_path.exists() else 0
            chunk_count = get_source_chunk_count(src)

            docs.append({
                "name":        src,
                "origin":      origin,        # "knowledge" | "uploaded" | "unknown"
                "size_bytes":  size_bytes,
                "chunk_count": chunk_count,
                "file_type":   Path(src).suffix.lower().lstrip("."),
            })

        return jsonify({"documents": docs})
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return jsonify({"error": str(e)}), 500

# ── Admin API: view document content ─────────────────────────
@app.route("/api/document/view", methods=["GET"])
@require_admin
def view_document():
    source_name = request.args.get("filename")
    if not source_name:
        return jsonify({"error": "filename is required"}), 400

    # Try to read actual file first (most accurate)
    for folder in [config.KNOWLEDGE_DIR, config.UPLOADS_DIR]:
        fpath = Path(folder) / source_name
        if fpath.exists() and fpath.suffix.lower() == ".txt":
            return jsonify({
                "filename": source_name,
                "content":  fpath.read_text(encoding="utf-8", errors="ignore"),
                "source":   "file",
            })
        elif fpath.exists() and fpath.suffix.lower() == ".pdf":
            preview = get_source_preview(source_name)
            return jsonify({
                "filename": source_name,
                "content":  preview,
                "source":   "chunks",
                "note":     "PDF — showing indexed text preview",
            })

    # Fallback: reconstruct from ChromaDB chunks
    preview = get_source_preview(source_name)
    return jsonify({
        "filename": source_name,
        "content":  preview or "No content found.",
        "source":   "chunks",
    })

# ── Admin API: update / edit document ─────────────────────────
@app.route("/api/document/update", methods=["POST"])
@require_admin
def update_document():
    """
    Two modes:
      1. TXT inline edit: JSON body with {filename, content}
      2. File re-upload:  multipart form with file
    """
    # Mode 1: inline text update
    if request.is_json:
        data    = request.json
        filename = data.get("filename")
        content  = data.get("content")
        if not filename or content is None:
            return jsonify({"error": "filename and content required"}), 400

        # Find existing file location
        target = None
        for folder in [config.KNOWLEDGE_DIR, config.UPLOADS_DIR]:
            fpath = Path(folder) / filename
            if fpath.exists():
                target = fpath
                break

        if not target:
            # Default to uploads dir
            target = Path(config.UPLOADS_DIR) / filename

        # Write updated content
        target.write_text(content, encoding="utf-8")

        # Re-index: delete old chunks → re-index file
        delete_source(filename)
        chunks = load_and_chunk_file(str(target), source_name=filename)
        count  = add_documents(chunks)
        invalidate_bm25_cache()

        logger.info(f"Document '{filename}' updated inline. {count} chunks re-indexed.")
        return jsonify({
            "message":      f"'{filename}' updated and re-indexed successfully.",
            "chunk_count":  count,
        })

    # Mode 2: file re-upload
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f        = request.files["file"]
    filename = f.filename
    suffix   = Path(filename).suffix.lower()

    if suffix not in config.SUPPORTED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: {suffix}"}), 400

    # Save to uploads dir (replaces old file)
    save_path = Path(config.UPLOADS_DIR) / filename
    f.save(str(save_path))

    # Re-index
    delete_source(filename)
    chunks = load_and_chunk_file(str(save_path), source_name=filename)
    count  = add_documents(chunks)
    invalidate_bm25_cache()

    logger.info(f"Document '{filename}' re-uploaded. {count} chunks re-indexed.")
    return jsonify({
        "message":     f"'{filename}' updated and re-indexed.",
        "chunk_count": count,
    })

# ── Admin API: upload new document(s) ────────────────────────
@app.route("/api/upload", methods=["POST"])
@require_admin
def upload_files():
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files         = request.files.getlist("files")
    results       = []
    total_chunks  = 0

    for f in files:
        filename = f.filename
        suffix   = Path(filename).suffix.lower()

        if suffix not in config.SUPPORTED_EXTENSIONS:
            results.append({"file": filename, "status": "skipped", "reason": f"Unsupported type {suffix}"})
            continue

        # ── Save permanently to uploads/ ──
        save_path = Path(config.UPLOADS_DIR) / filename
        f.save(str(save_path))

        # ── If already indexed, re-index (update) ──
        if source_exists(filename):
            logger.info(f"  '{filename}' already indexed — re-indexing (update).")
            delete_source(filename)

        try:
            chunks = load_and_chunk_file(str(save_path), source_name=filename)
            count  = add_documents(chunks)
            invalidate_bm25_cache()
            total_chunks += count
            results.append({"file": filename, "status": "indexed", "chunks": count})
            logger.info(f"Uploaded & indexed: {filename} ({count} chunks)")
        except Exception as e:
            logger.error(f"Error indexing {filename}: {e}")
            results.append({"file": filename, "status": "error", "reason": str(e)})

    return jsonify({
        "message":      f"Processed {len(files)} file(s).",
        "total_chunks": total_chunks,
        "results":      results,
    })

# ── Admin API: delete document ────────────────────────────────
@app.route("/api/delete_document", methods=["DELETE"])
@require_admin
def delete_document():
    data        = request.json or {}
    source_name = data.get("filename")
    if not source_name:
        return jsonify({"error": "filename is required"}), 400

    # Remove from ChromaDB
    ok = delete_source(source_name)
    if ok:
        invalidate_bm25_cache()

    # Remove from disk (uploads only — don't delete permanent knowledge files)
    disk_path = Path(config.UPLOADS_DIR) / source_name
    if disk_path.exists():
        disk_path.unlink()
        logger.info(f"Disk file deleted: {disk_path}")
    else:
        # Check if it's a knowledge file (can delete from ChromaDB but not disk)
        knowledge_path = Path(config.KNOWLEDGE_DIR) / source_name
        if knowledge_path.exists():
            logger.warning(
                f"'{source_name}' is a permanent knowledge file — "
                f"removed from ChromaDB but NOT from disk. It will re-index on restart."
            )
            return jsonify({
                "message": (
                    f"'{source_name}' removed from active knowledge base. "
                    f"Note: this is a permanent file — it will be re-indexed on next server restart."
                )
            })

    return jsonify({"message": f"'{source_name}' deleted successfully."})

# ── Admin API: clear all uploads ──────────────────────────────
@app.route("/api/clear", methods=["POST"])
@require_admin
def clear_database():
    data          = request.json or {}
    clear_uploads_only = data.get("uploads_only", False)

    try:
        if clear_uploads_only:
            # Only clear uploaded docs, keep permanent knowledge
            upload_files = [
                f.name for f in Path(config.UPLOADS_DIR).iterdir()
                if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTENSIONS
            ]
            for fname in upload_files:
                delete_source(fname)
                fpath = Path(config.UPLOADS_DIR) / fname
                if fpath.exists():
                    fpath.unlink()
            invalidate_bm25_cache()
            # Re-index permanent knowledge
            auto_index_knowledge()
            logger.info(f"Cleared {len(upload_files)} uploaded file(s).")
            return jsonify({
                "message": f"Cleared {len(upload_files)} uploaded file(s). Permanent knowledge preserved.",
                "total_chunks": get_document_count(),
            })
        else:
            clear_store()
            invalidate_bm25_cache()
            logger.info("Full knowledge base cleared.")
            return jsonify({"message": "Knowledge base cleared.", "total_chunks": 0})
    except Exception as e:
        logger.error(f"Error clearing KB: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
