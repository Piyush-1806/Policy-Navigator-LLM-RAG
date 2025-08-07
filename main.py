import os
import asyncio
import hashlib
import json
import time
import logging
from typing import List, Dict, Any, Optional
import traceback
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import uvicorn
import redis.asyncio as redis

from document_processor import DocumentProcessor
from vector_store import VectorStore
from llm_client import LLMClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration with validation
class Config:
    API_KEY = os.getenv("API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
    
    def validate(self):
        """Validate configuration"""
        if not self.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not set - using fallback logic")
        if not self.PINECONE_API_KEY:
            logger.warning("PINECONE_API_KEY not set - using FAISS fallback")

config = Config()
config.validate()

# Global components
document_processor = DocumentProcessor()
vector_store = VectorStore(config)
llm_client = LLMClient(config)

# Redis connection for caching
redis_client = None

# Request tracking
active_requests = {}

# Pydantic models for exact API specification matching HackRx requirements
class HackRxRequest(BaseModel):
    documents: str = Field(..., description="Single document URL")
    questions: List[str] = Field(..., min_length=1, max_length=20, description="List of questions")
    
    @field_validator('documents')
    @classmethod
    def validate_document_url(cls, v: str) -> str:
        if not v.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid URL: {v}")
        return v
    
    @field_validator('questions')
    @classmethod
    def validate_questions(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("At least one question is required")
        for question in v:
            if len(question.strip()) < 5:
                raise ValueError("Questions must be at least 5 characters long")
        return v

class HackRxResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answer strings")

# Lifespan manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global redis_client
    
    logger.info("Starting up LLM Insurance Query System")
    
    try:
        # Initialize Redis
        redis_client = redis.from_url(config.REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e} - Continuing without cache")
        redis_client = None
    
    # Initialize components
    await document_processor.initialize()
    await vector_store.initialize()
    await llm_client.initialize()
    
    logger.info("System initialization complete")
    
    yield
    
    # Shutdown
    if redis_client:
        await redis_client.close()
    logger.info("System shutdown complete")

# Security with better error handling
security = HTTPBearer(auto_error=False)  # Don't auto-raise errors

def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Verify Bearer token with proper error handling"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if credentials.credentials != config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# FastAPI app with enhanced configuration and lifespan
app = FastAPI(
    title="LLM Insurance Query System",
    description="AI-powered system for processing natural language queries over insurance documents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Enhanced health check with component status"""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {
            "document_processor": "ready",
            "vector_store": "ready" if vector_store.embedding_model else "not_ready",
            "llm_client": "ready" if llm_client.model else "fallback_mode",
            "redis": "ready" if redis_client else "disabled"
        },
        "active_requests": len(active_requests)
    }
    return health_status

@app.get("/metrics")
async def get_metrics(token: str = Depends(verify_token)):
    """System metrics endpoint"""
    return {
        "active_requests": len(active_requests),
        "cache_status": "enabled" if redis_client else "disabled",
        "vector_store_type": "pinecone" if vector_store.pinecone_index else "faiss",
        "llm_status": "gemini" if llm_client.model else "fallback"
    }

async def get_cache_key(documents: List[str], questions: List[str]) -> str:
    """Generate cache key for request"""
    content = f"{documents[0] if documents else ''}:{sorted(questions)}"
    return f"hackrx:{hashlib.sha256(content.encode()).hexdigest()[:16]}"

async def cache_get(key: str) -> Optional[Dict]:
    """Get from cache"""
    if not redis_client:
        return None
    try:
        result = await redis_client.get(key)
        return json.loads(result) if result else None
    except Exception as e:
        logger.warning(f"Cache get error: {e}")
        return None

async def cache_set(key: str, value: Dict, ttl: int = None):
    """Set to cache"""
    if not redis_client:
        return
    try:
        ttl = ttl or config.CACHE_TTL
        await redis_client.setex(key, ttl, json.dumps(value))
    except Exception as e:
        logger.warning(f"Cache set error: {e}")

@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_run(
    request: HackRxRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Main HackRx API endpoint matching exact specification"""
    request_id = hashlib.md5(f"{time.time()}:{id(request)}".encode()).hexdigest()[:8]
    start_time = time.time()
    
    # Track active request
    active_requests[request_id] = {
        "start_time": start_time,
        "document": request.documents,
        "questions": len(request.questions)
    }
    
    try:
        logger.info(f"[{request_id}] Processing document: {request.documents[:100]}... with {len(request.questions)} questions")
        
        # Check cache first
        cache_key = await get_cache_key([request.documents], request.questions)
        cached_result = await cache_get(cache_key)
        if cached_result:
            logger.info(f"[{request_id}] Returning cached result")
            return HackRxResponse(answers=cached_result["answers"])
        
        # Step 1: Process the document
        try:
            chunks = await document_processor.process_document(request.documents)
        except Exception as e:
            logger.error(f"[{request_id}] Failed to process document {request.documents}: {e}")
            return HackRxResponse(answers=["Failed to process document. Please check document URL and try again."] * len(request.questions))
        
        if not chunks:
            return HackRxResponse(answers=["No content found in the provided document."] * len(request.questions))
        
        logger.info(f"[{request_id}] Successfully processed {len(chunks)} chunks from document")
        
        # Step 2: Store chunks in vector database
        await vector_store.store_chunks(chunks)
        
        # Step 3: Process each question with parallel processing
        question_tasks = []
        for i, question in enumerate(request.questions):
            task = process_single_question_text(question, request_id, i)
            question_tasks.append(task)
        
        # Execute question processing in parallel
        answers = await asyncio.gather(*question_tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        final_answers = []
        for i, result in enumerate(answers):
            if isinstance(result, Exception):
                logger.error(f"[{request_id}] Failed to process question {i}: {result}")
                final_answers.append("Unable to process this question due to technical issues. Please try again.")
            else:
                final_answers.append(result)
        
        processing_time = time.time() - start_time
        
        response_data = {"answers": final_answers}
        
        # Cache the result
        background_tasks.add_task(cache_set, cache_key, response_data)
        
        logger.info(f"[{request_id}] Request completed in {processing_time:.2f} seconds")
        
        return HackRxResponse(answers=final_answers)
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"[{request_id}] Error processing request: {str(e)}")
        logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")
        
        return HackRxResponse(answers=["System error occurred while processing your request. Please try again later."] * len(request.questions))
    
    finally:
        # Remove from active requests
        active_requests.pop(request_id, None)

async def process_single_question_text(question: str, request_id: str, question_idx: int) -> str:
    """Process a single question and return answer as string"""
    try:
        logger.info(f"[{request_id}] Processing question {question_idx + 1}: {question[:50]}...")
        
        # Retrieve relevant chunks with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                relevant_chunks = await vector_store.search(question, top_k=5)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                logger.warning(f"[{request_id}] Retry {attempt + 1} for vector search: {e}")
                await asyncio.sleep(0.5 * (attempt + 1))
        
        if not relevant_chunks:
            return "No relevant information found in the provided document for this query."
        
        # Generate answer using LLM with timeout
        try:
            answer = await asyncio.wait_for(
                llm_client.generate_text_answer(question, relevant_chunks),
                timeout=25.0  # Leave 5 seconds buffer for overall 30s limit
            )
            return answer
        except asyncio.TimeoutError:
            logger.error(f"[{request_id}] LLM timeout for question: {question[:50]}")
            return "Request timeout - please try with a simpler question or fewer documents."
        
    except Exception as e:
        logger.error(f"[{request_id}] Error processing question '{question[:50]}': {str(e)}")
        return "Unable to process this question due to technical issues. Please rephrase and try again."

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return {
        "success": False,
        "error": {
            "code": exc.status_code,
            "message": exc.detail
        },
        "timestamp": time.time()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return {
        "success": False,
        "error": {
            "code": 500,
            "message": "Internal server error"
        },
        "timestamp": time.time()
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    workers = int(os.getenv("WORKERS", 1))
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port,
        workers=workers,
        access_log=True,
        log_level="info"
    )