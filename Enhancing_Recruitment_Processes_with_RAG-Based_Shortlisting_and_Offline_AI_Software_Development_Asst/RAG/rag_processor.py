import os
import pickle
import numpy as np
import faiss
from ollama import embeddings, chat
import re
import time
import requests

# Try to import optional dependencies with fallbacks
try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

class RAGProcessor:
    def __init__(self):
        # Initialize RAG processor with empty documents and no index
        self.documents = []  # List to store document chunks
        self.embeddings = None  # Will store document embeddings
        self.index = None  # Will store FAISS index for similarity search
        self.model = "llama3"  # Default model for generation
        self.embedding_model = "nomic-embed-text"  # Default model for embeddings
        
    def load_documents(self, data_folder):
        """Load all resumes from the data folder"""
        self.documents = []
        
        # Check if data folder exists
        if not os.path.exists(data_folder):
            print(f"Data folder '{data_folder}' does not exist. Creating it.")
            os.makedirs(data_folder)
            return
            
        # Check if data folder is empty
        if not os.listdir(data_folder):
            print(f"Data folder '{data_folder}' is empty. Please add some resumes.")
            return
            
        # Process each file in the data folder
        for filename in os.listdir(data_folder):
            filepath = os.path.join(data_folder, filename)
            content = ""
            
            try:
                # Handle different file types
                if filename.endswith('.pdf'):
                    content = self.extract_text_from_pdf(filepath)
                elif filename.endswith('.docx'):
                    content = self.extract_text_from_docx(filepath)
                elif filename.endswith('.doc'):
                    print(f"Warning: .doc files are not supported. Please convert {filename} to .docx")
                    continue
                elif filename.endswith('.txt'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                else:
                    print(f"Skipping unsupported file type: {filename}")
                    continue
                    
                # Skip if no content was extracted
                if not content.strip():
                    print(f"Warning: No content extracted from {filename}")
                    continue
                    
                # Clean and chunk the content
                chunks = self.chunk_text(content, filename)
                self.documents.extend(chunks)
                print(f"Processed {filename}: {len(chunks)} chunks")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
                
        print(f"Loaded {len(self.documents)} total document chunks")
    
    def extract_text_from_pdf(self, filepath):
        """Extract text from PDF files using available libraries"""
        text = ""
        
        # Try PyMuPDF first (most reliable)
        if HAS_PYMUPDF:
            try:
                with fitz.open(filepath) as doc:
                    for page in doc:
                        text += page.get_text() + "\n"
                return text
            except Exception as e:
                print(f"PyMuPDF failed: {e}")
        
        # Fall back to PyPDF2
        if HAS_PYPDF2:
            try:
                with open(filepath, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                return text
            except Exception as e:
                print(f"PyPDF2 failed: {e}")
        
        raise Exception("No PDF processing library available. Install PyMuPDF or PyPDF2.")
    
    def extract_text_from_docx(self, filepath):
        """Extract text from DOCX files"""
        if not HAS_DOCX:
            raise Exception("python-docx library not available. Install it with: pip install python-docx")
            
        doc = docx.Document(filepath)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    def chunk_text(self, text, filename, chunk_size=1000, overlap=200):
        """Split text into overlapping chunks"""
        # Clean the text by removing extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # If text is shorter than chunk size, return it as a single chunk
        if len(text) <= chunk_size:
            return [{
                'text': text,
                'source': filename,
                'start_idx': 0,
                'end_idx': len(text)
            }]
        
        chunks = []
        start = 0
        
        # Create chunks with overlap
        while start < len(text):
            end = start + chunk_size
            if end > len(text):
                end = len(text)
            
            # Try to end at a sentence boundary if possible
            if end < len(text):
                # Look for sentence endings near the chunk end
                for punctuation in ['. ', '! ', '? ', '\n']:
                    last_pos = text.rfind(punctuation, start, end)
                    if last_pos != -1 and last_pos > start + chunk_size / 2:
                        end = last_pos + len(punctuation)
                        break
            
            # Extract chunk and add to list
            chunk = text[start:end]
            chunks.append({
                'text': chunk,
                'source': filename,
                'start_idx': start,
                'end_idx': end
            })
            
            # Move start position, considering overlap
            start = end - overlap if end - overlap > start else end
            
        return chunks
    
    def generate_embeddings(self):
        """Generate embeddings for all document chunks"""
        if not self.documents:
            raise ValueError("No documents loaded. Please load documents first.")
            
        texts = [doc['text'] for doc in self.documents]
        
        # Generate embeddings using Ollama with retry logic
        print("Generating embeddings...")
        
        # Fix: Ollama embeddings API expects a different format
        all_embeddings = []
        max_retries = 3
        
        # Generate embeddings for each text chunk
        for i, text in enumerate(texts):
            for attempt in range(max_retries):
                try:
                    print(f"Generating embedding {i+1}/{len(texts)}...")
                    response = embeddings(model=self.embedding_model, prompt=text)
                    all_embeddings.append(response['embedding'])
                    break
                except requests.exceptions.ConnectionError:
                    if attempt < max_retries - 1:
                        print(f"Connection error, retrying in 5 seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(5)
                    else:
                        raise Exception("Failed to connect to Ollama after multiple attempts. Make sure Ollama is running.")
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Error generating embedding, retrying... (Attempt {attempt + 1}/{max_retries}): {e}")
                        time.sleep(5)
                    else:
                        raise
        
        # Convert to numpy array for FAISS
        self.embeddings = np.array(all_embeddings, dtype=np.float32)
        
        # Create FAISS index for similarity search
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        
        print(f"Created index with {self.index.ntotal} vectors")
    
    def save_index(self, emb_file='embeddings.pkl', index_file='index.faiss'):
        """Save embeddings and FAISS index to disk"""
        with open(emb_file, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings
            }, f)
        
        faiss.write_index(self.index, index_file)
        print(f"Saved embeddings to {emb_file} and index to {index_file}")
    
    def load_index(self, emb_file='embeddings.pkl', index_file='index.faiss'):
        """Load embeddings and FAISS index from disk"""
        if not os.path.exists(emb_file) or not os.path.exists(index_file):
            return False
            
        with open(emb_file, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.embeddings = data['embeddings']
        
        self.index = faiss.read_index(index_file)
        print(f"Loaded {self.index.ntotal} vectors from index")
        return True
    
    def search(self, query, k=5):
        """Search for relevant document chunks"""
        # Generate query embedding
        response = embeddings(model=self.embedding_model, prompt=query)
        query_embedding = np.array([response['embedding']], dtype=np.float32)
        
        # Search the index for similar documents
        distances, indices = self.index.search(query_embedding, k)
        
        # Retrieve the relevant documents
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents) and idx >= 0:  # Ensure index is valid
                doc = self.documents[idx]
                results.append({
                    'text': doc['text'],
                    'source': doc['source'],
                    'score': float(distance)
                })
        
        return results
    
    def generate_response(self, query, context):
        """Generate a response using Ollama based on query and context"""
        # Prepare context text from search results
        context_text = "\n\n".join([f"From {doc['source']}:\n{doc['text']}" for doc in context])
        
        # Create the prompt with context and query
        prompt = f"""Based on the following resume information, answer the user's question.

Resume Context:
{context_text}

User Question: {query}

Please provide a helpful answer based only on the resume information above. 
If the information isn't available in the resumes, politely state that you cannot answer 
based on the available resume data.if answers is in points or multiple points, use points format (e.x 1,2,3)proper formatting. """

        # Get response from Ollama
        response = chat(model=self.model, messages=[{
            'role': 'user',
            'content': prompt
        }])
        
        return response['message']['content']