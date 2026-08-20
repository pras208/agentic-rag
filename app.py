import os
import uuid
import logging
import yaml
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename

from embedding import Embedder, VectorStore
from document import DocumentProcessor
from llm import BedrockClient
from agent import AgentOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
try:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    logger.error("config.yaml not found")
    raise
except yaml.YAMLError as e:
    logger.error(f"Error parsing config.yaml: {e}")
    raise

# Initialize Flask app
app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = config["app"]["max_upload_size"] * 1024 * 1024
app.config["UPLOAD_FOLDER"] = "uploads"

# Create upload folder
Path("uploads").mkdir(exist_ok=True)

# Initialize components with error handling
try:
    logger.info("Initializing components...")
    embedder = Embedder(config["embedding"]["model_name"])
    vector_store = VectorStore(
        dimension=config["embedding"]["dimension"],
        index_path=config["vector_store"]["index_path"],
        metadata_path=config["vector_store"]["metadata_path"]
    )
    document_processor = DocumentProcessor(
        chunk_size=config["document_processing"]["chunk_size"],
        chunk_overlap=config["document_processing"]["chunk_overlap"],
        max_workers=config["document_processing"]["max_workers"]
    )
    bedrock = BedrockClient(
        region=config["bedrock"]["region"]
    )
    agent = AgentOrchestrator(embedder, vector_store, bedrock)
    logger.info("Components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")
    raise

# Thread pool for async document processing
executor = ThreadPoolExecutor(max_workers=config["document_processing"]["max_workers"])

@app.route("/")
def index():
    """Serve UI."""
    return render_template("index.html")

@app.route("/api/status")
def status():
    """Get system status."""
    return jsonify({
        "indexed_documents": vector_store.get_size(),
        "embedding_model": config["embedding"]["model_name"]
    })

@app.route("/api/upload", methods=["POST"])
def upload():
    """Handle document upload and start background processing."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Validate file type
        allowed_extensions = {".pdf", ".txt", ".md"}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            return jsonify({
                "error": f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            }), 400

        # Validate file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Seek back to start
        max_size_bytes = config["app"]["max_upload_size"] * 1024 * 1024
        if file_size > max_size_bytes:
            return jsonify({
                "error": f"File too large. Max size: {config['app']['max_upload_size']}MB"
            }), 413

        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        logger.info(f"File saved: {filename}")

        # Start background processing
        job_id = str(uuid.uuid4())

        def process_and_index() -> dict:
            """Process file and add to vector store."""
            try:
                logger.info(f"Processing document: {filename} (job_id: {job_id})")
                chunks = document_processor.process_file(file_path, job_id)

                # Generate embeddings
                texts = [chunk["text"] for chunk in chunks]
                embeddings = embedder.embed_batch(texts)

                # Prepare metadata
                metadatas = [
                    {
                        **chunk,
                        "source_file": filename
                    }
                    for chunk in chunks
                ]

                # Add to vector store
                vector_store.add_documents(embeddings, metadatas)
                logger.info(f"Successfully indexed {len(chunks)} chunks from {filename}")
                return {"status": "success", "chunks_indexed": len(chunks)}
            except Exception as e:
                logger.error(f"Error processing file {filename}: {str(e)}")
                return {"status": "error", "message": str(e)}

        # Submit to thread pool
        executor.submit(process_and_index)

        return jsonify({"status": "pending", "job_id": job_id, "message": "File queued for processing"})
    except Exception as e:
        logger.error(f"Upload endpoint error: {str(e)}")
        return jsonify({"status": "error", "message": "Failed to process upload"}), 500

@app.route("/api/query", methods=["POST"])
def query():
    """Handle user query with streaming response."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body required"}), 400

        query_text: Optional[str] = data.get("query", "").strip()
        model_id: Optional[str] = data.get("model", "").strip()

        # Validate inputs
        if not query_text:
            return jsonify({"error": "Query text is required"}), 400
        if len(query_text) > 5000:
            return jsonify({"error": "Query too long (max 5000 characters)"}), 413
        if not model_id:
            return jsonify({"error": "Model ID is required"}), 400

        logger.info(f"Processing query with model: {model_id}")

        def generate():
            """Stream response from agent."""
            try:
                # Update the model in the Bedrock client
                bedrock.model_id = model_id
                token_count = 0
                for token in agent.reason(query_text):
                    yield f"data: {token}\n\n"
                    token_count += 1
                logger.info(f"Query completed with {token_count} tokens")
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Error during query streaming: {str(e)}")
                yield f"data: ERROR: {str(e)}\n\n"

        return Response(generate(), mimetype="text/event-stream")
    except Exception as e:
        logger.error(f"Query endpoint error: {str(e)}")
        return jsonify({"error": "Failed to process query"}), 500

@app.route("/api/status")
def status():
    """Get system status."""
    try:
        return jsonify({
            "status": "ok",
            "indexed_documents": vector_store.get_size(),
            "embedding_model": config["embedding"]["model_name"]
        })
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({"error": "Failed to get status"}), 500

@app.route("/api/job-status/<job_id>")
def job_status(job_id: str):
    """Get processing job status."""
    try:
        if not job_id or len(job_id) != 36:  # UUID length
            return jsonify({"error": "Invalid job ID"}), 400
        status_data = document_processor.get_job_status(job_id)
        return jsonify(status_data)
    except Exception as e:
        logger.error(f"Error getting job status for {job_id}: {str(e)}")
        return jsonify({"error": "Failed to get job status"}), 500

if __name__ == "__main__":
    app.run(
        host=config["app"]["host"],
        port=config["app"]["port"],
        debug=config["app"]["debug"]
    )