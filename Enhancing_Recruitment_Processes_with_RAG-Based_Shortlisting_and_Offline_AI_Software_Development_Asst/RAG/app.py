from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_processor import RAGProcessor
import os

app = Flask(__name__)
CORS(app)

# Initialize RAG processor
rag_processor = RAGProcessor()

# Check if we have saved index, otherwise process documents
if not rag_processor.load_index():
    print("No existing index found. Processing documents...")
    rag_processor.load_documents('data')
    rag_processor.generate_embeddings()
    rag_processor.save_index()

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        # Search for relevant documents
        results = rag_processor.search(query, k=5)
        
        # Generate response
        response = rag_processor.generate_response(query, results)
        
        return jsonify({
            'response': response,
            'sources': [{'source': r['source'], 'score': r['score']} for r in results]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reload', methods=['POST'])
def reload_documents():
    try:
        rag_processor.load_documents('data')
        rag_processor.generate_embeddings()
        rag_processor.save_index()
        return jsonify({'message': 'Documents reloaded successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)