import os
import logging
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_vector_index():
    """Verify ChromaDB index exists and is queryable"""
    
    chroma_dir = "embeddings/chroma"
    
    print("\n" + "=" * 60)
    print(" ChromaDB Vector Index Verification")
    print("=" * 60)
    
    # Check if directory exists
    if not os.path.exists(chroma_dir):
        print(f"\n Vector index not found at: {chroma_dir}")
        print("Please run: python backend/ingest_data.py")
        return False
    
    print(f"\n✅ Vector index directory exists: {chroma_dir}")
    
    # Load embeddings
    try:
        print("\n Loading embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print(" Embedding model loaded")
    except Exception as e:
        print(f" Error loading embeddings: {e}")
        return False
    
    # Load vector store
    try:
        print("\n  Loading ChromaDB vector store...")
        vectorstore = Chroma(
            persist_directory=chroma_dir,
            embedding_function=embeddings,
            collection_name="medical_docs"
        )
        print("✅ Vector store loaded")
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        return False
    
    # Get collection stats
    try:
        collection = vectorstore._collection
        count = collection.count()
        print(f"\n Vector Store Statistics:")
        print(f"  Total documents indexed: {count}")
        
        if count == 0:
            print("\n  Warning: Vector store is empty!")
            print("Please run: python backend/ingest_data.py")
            return False
        
    except Exception as e:
        print(f"⚠️  Could not get collection count: {e}")
    
    # Test retrieval
    try:
        print("\n🔎 Testing retrieval with sample query...")
        test_query = "What are common medical treatments?"
        results = vectorstore.similarity_search(test_query, k=3)
        
        print(f"✅ Retrieved {len(results)} results")
        print("\n Sample retrieved chunk:")
        if results:
            print(f"  Source: {results[0].metadata.get('source', 'Unknown')}")
            print(f"  Preview: {results[0].page_content[:200]}...")
        
    except Exception as e:
        print(f"❌ Error during test retrieval: {e}")
        return False
    
    print("\n" + "=" * 60)
    print(" Vector index verification complete!")
    print("=" * 60)
    print("\n🚀 System ready! Run: streamlit run app.py")
    
    return True


if __name__ == "__main__":
    success = verify_vector_index()
    exit(0 if success else 1)