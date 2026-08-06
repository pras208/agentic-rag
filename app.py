import os
import uuid
import asyncio
import yaml
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename

from embedding import Embedder, VectorStore
from document import DocumentProcessor
from llm import BedrockClient
from agent import AgentOrchestrator

# Load configuration
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Initialize Flask app
app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = config["app"]["max_upload_size"] * 1024 * 1024
app.config["UPLOAD_FOLDER"] = "uploads"

# Create upload folder
Path("uploads").mkdir(exist_ok=True)

# Initialize components
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
    region=config["bedrock"]["region"],
    model_id=config["bedrock"]["model_id"]
)
agent = AgentOrchestrator(embedder, vector_store, bedrock)

@app.route("/")
def index():
    """Serve UI."""
    return render_template("index.html")

@app.route("/api/status")
def status():
    """Get system status."""
    return jsonify({
        "indexed_documents": vector_store.get_size(),
        "embedding_model": config["embedding"]["model_name"],
        "bedrock_model": config["bedrock"]["model_id"]
    })

@app.route("/api/upload", methods=["POST"])
def upload():
    """Handle document upload and start async processing."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Validate file type
    allowed_extensions = {".pdf", ".txt", ".md"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        return jsonify({"error": f"Unsupported file type. Allowed: {allowed_extensions}"}), 400

    # Save file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Start async processing
    job_id = str(uuid.uuid4())

    # Process in background using asyncio
    async def process_and_index():
        try:
            chunks = await document_processor.process_file_async(file_path, job_id)

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

            return {"status": "success", "chunks_indexed": len(chunks)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # Run async task
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(process_and_index())
        loop.close()

        return jsonify({**result, "job_id": job_id})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/query", methods=["POST"])
def query():
    """Handle user query with streaming response."""
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Query required"}), 400

    query_text = data["query"]

    def generate():
        """Stream response from agent."""
        try:
            for token in agent.reason(query_text):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"

    return Response(generate(), mimetype="text/event-stream")

@app.route("/api/job-status/<job_id>")
def job_status(job_id):
    """Get processing job status."""
    status = document_processor.get_job_status(job_id)
    return jsonify(status)

if __name__ == "__main__":
    app.run(
        host=config["app"]["host"],
        port=config["app"]["port"],
        debug=config["app"]["debug"]
    )
