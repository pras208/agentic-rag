from typing import Generator, Dict
from embedding import Embedder, VectorStore
from llm import BedrockClient
from .prompts import RETRIEVAL_PROMPT_TEMPLATE, DIRECT_PROMPT_TEMPLATE

class AgentOrchestrator:
    """Simple reasoning loop for agentic RAG."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore, bedrock: BedrockClient):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bedrock = bedrock

    def reason(self, query: str, max_iterations: int = 5) -> Generator:
        """Agentic reasoning loop - decide whether to retrieve or answer directly."""
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Simple decision logic: retrieve if documents are indexed and query seems document-related
            if self.vector_store.get_size() > 0 and self._should_retrieve(query):
                context = self._retrieve_context(query)
                if context:
                    prompt = RETRIEVAL_PROMPT_TEMPLATE.format(query=query, context=context)
                else:
                    prompt = DIRECT_PROMPT_TEMPLATE.format(query=query)
            else:
                prompt = DIRECT_PROMPT_TEMPLATE.format(query=query)

            # Stream response from Bedrock
            for token in self.bedrock.invoke_stream(prompt):
                yield token

            break  # Simple POC: one iteration

    def _should_retrieve(self, query: str) -> bool:
        """Heuristic to decide if query needs document retrieval."""
        keywords = ["what", "how", "explain", "describe", "tell", "where", "when"]
        return any(kw in query.lower() for kw in keywords)

    def _retrieve_context(self, query: str, k: int = 5) -> str:
        """Retrieve relevant context from vector store."""
        query_embedding = self.embedder.embed_text(query)
        results = self.vector_store.search(query_embedding, k=k)

        if not results:
            return ""

        context_parts = []
        for result in results:
            context_parts.append(f"- {result['text']}")

        return "\n".join(context_parts)
