#ingest_data.py
import os
import logging
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class MedicalDataIngestion:
    """Handles ingestion of medical PDFs into ChromaDB vector store"""
    
    def __init__(
        self,
        data_dir: str = "data",
        chroma_dir: str = "embeddings/chroma",
        chunk_size: int = 1500,
        chunk_overlap: int = 300
    ):
        self.data_dir = Path(data_dir)
        self.chroma_dir = Path(chroma_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embeddings model
        logger.info("Loading embedding model: sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def find_all_pdfs(self) -> List[Path]:
        """Recursively find all PDF files in data directory and subdirectories"""
        pdf_files = []
        
        if not self.data_dir.exists():
            logger.error(f"Data directory not found: {self.data_dir}")
            logger.info("Creating data directory...")
            self.data_dir.mkdir(parents=True, exist_ok=True)
            return pdf_files
        
        logger.info(f"Scanning for PDFs in: {self.data_dir}")
        
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_path = Path(root) / file
                    pdf_files.append(pdf_path)
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        return pdf_files
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Dict:
        """Extract text from a single PDF file using PyMuPDF"""
        doc = None
        try:
            doc = fitz.open(str(pdf_path))
            text = ""
            num_pages = len(doc)
            
            for page_num in range(num_pages):
                page = doc[page_num]
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n"
            
            # Get relative path for better source tracking
            try:
                relative_path = pdf_path.relative_to(self.data_dir)
            except ValueError:
                relative_path = pdf_path.name
            
            result = {
                "text": text.strip(),
                "source": str(relative_path),
                "full_path": str(pdf_path),
                "num_pages": num_pages
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return None
        
        finally:
            # Always close the document in the finally block
            if doc is not None:
                doc.close()
    
    def process_documents(self, pdf_files: List[Path]) -> List[Dict]:
        """Process all PDFs and create text chunks with metadata"""
        all_chunks = []
        
        logger.info(f"Processing {len(pdf_files)} PDF files...")
        
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            doc_data = self.extract_text_from_pdf(pdf_path)
            
            if not doc_data or not doc_data["text"]:
                logger.warning(f"Skipping empty document: {pdf_path}")
                continue
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(doc_data["text"])
            
            logger.info(f"Created {len(chunks)} chunks from {doc_data['source']}")
            
            # Add metadata to each chunk
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    "text": chunk,
                    "metadata": {
                        "source": doc_data["source"],
                        "full_path": doc_data["full_path"],
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                        "num_pages": doc_data["num_pages"]
                    }
                }
                all_chunks.append(chunk_data)
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def create_vector_store(self, chunks: List[Dict]) -> Chroma:
        """Create and persist ChromaDB vector store"""
        
        # Create directory if it doesn't exist
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating ChromaDB vector store at: {self.chroma_dir}")
        
        # Prepare texts and metadatas
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Create vector store
        vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            persist_directory=str(self.chroma_dir),
            collection_name="medical_docs"
        )
        
        logger.info("Vector store created and persisted successfully")
        return vectorstore
    
    def ingest(self) -> bool:
        """Main ingestion pipeline"""
        try:
            logger.info("=" * 60)
            logger.info("Starting Medical Data Ingestion Pipeline")
            logger.info("=" * 60)
            
            # Find all PDFs
            pdf_files = self.find_all_pdfs()
            
            if not pdf_files:
                logger.error("No PDF files found. Please add PDFs to the data/ directory")
                return False
            
            # Process documents
            chunks = self.process_documents(pdf_files)
            
            if not chunks:
                logger.error("No chunks created. Check PDF content")
                return False
            
            # Create vector store
            vectorstore = self.create_vector_store(chunks)
            
            logger.info("=" * 60)
            logger.info("Ingestion completed successfully!")
            logger.info(f"Total documents processed: {len(pdf_files)}")
            logger.info(f"Total chunks indexed: {len(chunks)}")
            logger.info(f"Vector store location: {self.chroma_dir}")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)
            return False


def main():
    """Run the ingestion pipeline"""
    ingestion = MedicalDataIngestion()
    success = ingestion.ingest()
    
    if success:
        print("\n Data ingestion completed successfully!")
        print(f"Vector database saved to: embeddings/chroma/")
        print("\n You can now run the Streamlit app: streamlit run app.py")
    else:
        print("\n Data ingestion failed. Check logs for details.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())