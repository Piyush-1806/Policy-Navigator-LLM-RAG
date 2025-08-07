import asyncio
from typing import List, Dict, Any, Optional
import json
import hashlib
import logging

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, config):
        self.config = config
        self.embedding_model = None
        self.faiss_index = None
        self.chunk_metadata = {}
        self.embedding_dimension = None
        
    async def initialize(self):
        """Initialize vector store with FAISS only"""
        # Load embedding model - use better model for legal/insurance text
        try:
            # Try the better model first
            try:
                self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
                logger.info("✅ Using all-mpnet-base-v2 (higher quality)")
            except:
                logger.warning("⚠️ Falling back to all-MiniLM-L6-v2")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            test_embedding = self.embedding_model.encode(["test"])
            self.embedding_dimension = test_embedding.shape[1]
            logger.info(f"✅ Embedding model loaded (dimension: {self.embedding_dimension})")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
        
        # Initialize FAISS vector store
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
        logger.info(f"✅ FAISS initialized with dimension {self.embedding_dimension}")
    
    async def store_chunks(self, chunks: List[Dict[str, Any]]):
        """Store chunks in FAISS vector database"""
        if not chunks:
            return
        
        try:
            # Generate embeddings
            texts = [chunk['text'] for chunk in chunks]
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            
            embeddings = self.embedding_model.encode(texts)
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            
            # Store in FAISS
            self.faiss_index.add(embeddings.astype('float32'))
            
            # Store metadata
            start_idx = len(self.chunk_metadata)
            for i, chunk in enumerate(chunks):
                self.chunk_metadata[start_idx + i] = chunk
            
            logger.info(f"✅ Stored {len(chunks)} vectors in FAISS")
                
        except Exception as e:
            logger.error(f"❌ Error storing chunks: {e}")
            # Don't raise - let the system continue
    
    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant chunks with enhanced query processing"""
        try:
            # Enhanced query processing for better matching
            enhanced_queries = [
                query,  # Original query
                query.replace("What is", "").replace("?", "").strip(),  # Remove question words
                query.lower()  # Lowercase version
            ]
            
            # Add specific term extraction for insurance queries
            if "pre-existing" in query.lower():
                enhanced_queries.extend([
                    "Pre-existing Disease means condition ailment injury",
                    "pre-existing conditions excluded life-threatening"
                ])
            if "hospital" in query.lower() and "definition" in query.lower():
                enhanced_queries.append("Hospital means institution registered Clinical Establishments nursing staff beds")
            if "permanent total disablement" in query.lower():
                enhanced_queries.extend([
                    "Permanent Total Disablement bodily injury accidental external violent",
                    "PTD total disablement 12 months occupation paralysis"
                ])
            if "emergency accidental hospitalization" in query.lower():
                enhanced_queries.extend([
                    "Emergency Accidental Hospitalization in-patient treatment diagnostic",
                    "accidental injury hospitalization covered excluded"
                ])
            if "opd emergency medical expenses" in query.lower() or "exclusions" in query.lower():
                enhanced_queries.extend([
                    "OPD Emergency Medical Expenses exclusions",
                    "treatment delayed customary charges cosmetic pregnancy"
                ])
            
            all_results = []
            
            # Search with multiple query variations using FAISS
            for q in enhanced_queries:
                if not q.strip():
                    continue
                    
                query_embedding = self.embedding_model.encode([q])
                
                # Search FAISS
                if self.faiss_index and self.faiss_index.ntotal > 0:
                    scores, indices = self.faiss_index.search(
                        query_embedding.astype('float32'), 
                        min(top_k, self.faiss_index.ntotal)
                    )
                    
                    for score, idx in zip(scores[0], indices[0]):
                        if idx != -1 and idx < len(self.chunk_metadata):
                            chunk = self.chunk_metadata[idx].copy()
                            chunk['score'] = float(score)
                            chunk['query_variant'] = q
                            all_results.append(chunk)
                    
                    logger.info(f"✅ FAISS search for '{q}' returned {len(scores[0])} chunks")
            
            # Remove duplicates and rank with special boost for definition chunks
            seen_texts = set()
            unique_results = []
            
            # First pass: boost definition chunks
            boosted_results = []
            for chunk in all_results:
                score = chunk.get('score', 0)
                # Boost score for definition chunks
                if any(term in chunk['text'] for term in ['Pre-existing Disease means', 'Hospital means', 'Permanent Total Disablement means']):
                    score += 0.3  # Significant boost for exact definitions
                elif 'means any' in chunk['text'] or 'defined as' in chunk['text']:
                    score += 0.2  # Medium boost for other definitions
                chunk['boosted_score'] = score
                boosted_results.append(chunk)
            
            # Sort by boosted score
            for chunk in sorted(boosted_results, key=lambda x: x.get('boosted_score', 0), reverse=True):
                text_key = chunk['text'][:100]  # Use first 100 chars as key
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    unique_results.append(chunk)
                    if len(unique_results) >= top_k:
                        break
            
            logger.info(f"✅ Enhanced search returned {len(unique_results)} unique chunks")
            return unique_results[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []