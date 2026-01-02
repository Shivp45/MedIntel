import os
import logging
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq
from openai import OpenAI as OAClient
import httpx

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class MedicalRAGPipeline:
    """RAG Pipeline with Groq primary and OpenAI fallback"""

    def __init__(self, chroma_dir: str = "embeddings/chroma", top_k: int = 5):
        self.chroma_dir = chroma_dir
        self.top_k = top_k

        # Initialize embeddings
        logger.info("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Load vector store
        self.vectorstore = self._load_vectorstore()

        # Initialize API keys
        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        self.groq_client = Groq(api_key=groq_key) if groq_key else None
        self.openai_client = OAClient(api_key=openai_key) if openai_key else None

    def _load_vectorstore(self) -> Optional[Chroma]:
        """Load existing ChromaDB vector store"""
        try:
            if not os.path.exists(self.chroma_dir):
                logger.error(f"Vector store not found at: {self.chroma_dir}")
                return None

            return Chroma(
                persist_directory=self.chroma_dir,
                embedding_function=self.embeddings,
                collection_name="medical_docs"
            )
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return None

    def _build_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """Construct prompt using retrieved context"""
        context_text = "\n\n".join([
            f"[Source: {c['metadata'].get('source','Unknown')} | Score: {c['similarity_score']:.4f}]\n{c['text']}"
            for c in context_chunks
        ])

        return f"""You are a Medical Research Assistant for doctors. Provide accurate, evidence-based answers using ONLY the context below.

IMPORTANT:
- If insufficient info, say so clearly.
- Highlight drug interactions or warnings if found.
- Cite sources by document name.
- Keep answer concise but medically precise.

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:"""

    def retrieve_context(self, query: str) -> List[Dict]:
        """Retrieve top-k relevant chunks"""
        if not self.vectorstore:
            return []
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=self.top_k)
            return [{"text": d.page_content, "metadata": d.metadata, "similarity_score": s} for d,s in results]
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []

    def _call_openai(self, prompt: str) -> Optional[str]:
        """Fallback OpenAI call with extended timeout safety"""
        if not self.openai_client:
            return None
        try:
            with httpx.Client(timeout=25.0) as client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI fallback failed: {e}")
            return None

    def generate_answer(self, query: str) -> Tuple[str, List[Dict], str]:
        """Full RAG flow: retrieve → generate → fallback"""
        context_chunks = self.retrieve_context(query)
        if not context_chunks:
            return "No relevant medical context found.", [], "none"

        prompt = self._build_prompt(query, context_chunks)

        # Try Groq
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500,
                top_p=0.9
            )
            answer = response.choices[0].message.content
            return answer, context_chunks, "Groq (Llama 3.1-8B)"
        except Exception as e:
            logger.warning("Groq failed, switching to OpenAI fallback...")
            answer = self._call_openai(prompt)
            return answer or "Both LLMs failed. Check API keys.", context_chunks, "OpenAI (fallback)"

    def check_system_health(self) -> Dict[str, bool]:
        """Check if vector DB and API keys exist"""
        return {
            "vectorstore": self.vectorstore is not None,
            "groq_api": bool(os.getenv("GROQ_API_KEY")),
            "openai_api": bool(os.getenv("OPENAI_API_KEY"))
        }

if __name__ == "__main__":
    pipe = MedicalRAGPipeline()
    q = "What are the common side effects of aspirin?"
    ans, ctx, llm = pipe.generate_answer(q)

    print("\nLLM Used:", llm)
    print("\nRetrieved Chunks:", len(ctx))
    print("\nAnswer:\n", ans[:500])
