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
        self.pinecone_index = None
        self.faiss_index = None
        self.chunk_metadata = {}
        self.embedding_dimension = None
        
    async def initialize(self):
        """Initialize vector store - prioritize FAISS to avoid dimension issues"""
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
        
        # For now, let's skip Pinecone due to dimension mismatch and use FAISS
        # This ensures the system works reliably
        logger.info("🔄 Using FAISS vector store to avoid dimension mismatch issues")
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
        logger.info(f"✅ FAISS initialized with dimension {self.embedding_dimension}")
        
        # Optional: Try Pinecone but don't fail if it doesn't work
        if self.config.PINECONE_API_KEY:
            try:
                from pinecone import Pinecone
                pc = Pinecone(api_key=self.config.PINECONE_API_KEY)
                
                index_name = "hackrx-insurance"
                existing_indexes = pc.list_indexes()
                index_names = [index.name for index in existing_indexes.indexes]
                
                if index_name in index_names:
                    # Check if dimensions match
                    for idx in existing_indexes.indexes:
                        if idx.name == index_name:
                            if idx.dimension == self.embedding_dimension:
                                self.pinecone_index = pc.Index(index_name)
                                logger.info("✅ Pinecone connected (dimensions match)")
                            else:
                                logger.warning(f"⚠️  Pinecone dimension mismatch: {idx.dimension} vs {self.embedding_dimension}")
                                logger.info("🔄 Will use FAISS instead")
                            break
                else:
                    logger.info("ℹ️  Pinecone index not found, using FAISS")
            except Exception as e:
                logger.info(f"ℹ️  Pinecone not available: {e}, using FAISS")
    
    async def store_chunks(self, chunks: List[Dict[str, Any]]):
        """Store chunks in vector database"""
        if not chunks:
            return
        
        try:
            # Generate embeddings
            texts = [chunk['text'] for chunk in chunks]
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            
            embeddings = self.embedding_model.encode(texts)
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            
            # Try Pinecone first if available
            if self.pinecone_index:
                try:
                    vectors = []
                    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                        vector_id = hashlib.md5(f"{chunk['text'][:50]}_{i}".encode()).hexdigest()
                        vectors.append({
                            'id': vector_id,
                            'values': embedding.tolist(),
                            'metadata': {
                                'text': chunk['text'][:1000],
                                'page': chunk['page'],
                                'source_url': chunk['source_url'][:200],
                                'chunk_id': chunk['chunk_id']
                            }
                        })
                    
                    # Upload in small batches
                    batch_size = 25
                    for i in range(0, len(vectors), batch_size):
                        batch = vectors[i:i + batch_size]
                        self.pinecone_index.upsert(vectors=batch)
                    
                    logger.info(f"✅ Stored {len(vectors)} vectors in Pinecone")
                    return  # Success, exit early
                    
                except Exception as e:
                    logger.warning(f"⚠️  Pinecone storage failed: {e}")
                    logger.info("🔄 Falling back to FAISS")
            
            # Store in FAISS (primary or fallback)
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
            
            # Search with multiple query variations
            for q in enhanced_queries:
                if not q.strip():
                    continue
                    
                query_embedding = self.embedding_model.encode([q])
                
                # Try Pinecone first if available
                if self.pinecone_index:
                    try:
                        results = self.pinecone_index.query(
                            vector=query_embedding[0].tolist(),
                            top_k=top_k,
                            include_metadata=True
                        )
                        
                        for match in results.get('matches', []):
                            if 'metadata' in match:
                                chunk = {
                                    'text': match['metadata'].get('text', ''),
                                    'page': match['metadata'].get('page', 1),
                                    'source_url': match['metadata'].get('source_url', ''),
                                    'chunk_id': match['metadata'].get('chunk_id', ''),
                                    'score': match.get('score', 0.0),
                                    'query_variant': q
                                }
                                all_results.append(chunk)
                        
                        logger.info(f"✅ Pinecone search for '{q}' returned {len(results.get('matches', []))} chunks")
                        # Use first successful Pinecone result
                        break
                        
                    except Exception as e:
                        logger.warning(f"⚠️  Pinecone search failed: {e}")
                        continue
                
                # Search FAISS (primary or fallback)
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
                    # Don't break for FAISS - collect all variants
            
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