# LLM-Powered Intelligence Query-Retrieval System for HackRx 6.0

A FastAPI-based system for processing natural language queries over insurance documents, specifically designed for the HackRx 6.0 competition.

## 🎯 Competition Overview

This system implements an LLM-powered intelligent query-retrieval system that processes large documents and makes contextual decisions for insurance, legal, HR, and compliance domains.

**Competition Requirements:**
- Process PDFs, DOCX, and email documents
- Handle policy/contract data efficiently  
- Parse natural language queries
- Use embeddings (FAISS/Pinecone) for semantic search
- Implement clause retrieval and matching
- Provide explainable decision rationale
- Output structured responses

## 🏗️ System Architecture

```
[Document Input] → [LLM Parser] → [Embedding Search] → [Clause Matching] → [Logic Evaluation] → [Response]
```

**Components:**
1. **Document Processor**: Downloads and parses PDFs, DOCX, emails
2. **Text Chunking**: Semantic segmentation with overlap
3. **Vector Search**: Pinecone (primary) / FAISS (fallback)
4. **LLM Integration**: Google Gemini-2.0-flash-exp
5. **Answer Generation**: Context-aware responses

## 🚀 API Specification

### Endpoint
```
POST /api/v1/hackrx/run
```

### Headers
```
Authorization: Bearer hackrx-api-key
Content-Type: application/json
```

### Request Format
```json
{
  "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=...",
  "questions": [
    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
    "What is the waiting period for pre-existing diseases (PED) to be covered?",
    "Does this policy cover maternity expenses, and what are the conditions?"
  ]
}
```

### Response Format
```json
{
  "answers": [
    "A grace period of thirty days is provided for premium payment after the due date.",
    "There is a waiting period of thirty-six (36) months of continuous coverage for pre-existing diseases.",
    "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy."
  ]
}
```

## 🛠️ Tech Stack

- **Backend**: FastAPI + Uvicorn + Gunicorn
- **LLM**: Google Gemini-2.0-flash-exp
- **Vector DB**: Pinecone (primary) / FAISS (fallback)
- **Document Processing**: PyMuPDF, python-docx, mailparser
- **Embeddings**: sentence-transformers (all-mpnet-base-v2)
- **Caching**: Redis
- **Deployment**: Docker + Heroku/Railway/Render

## 🏃‍♂️ Quick Start

### Local Development

1. **Clone and Setup**
   ```bash
   git clone <repository>
   cd hackrx-llm-system
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Environment Configuration**
   ```bash
   # Create .env file
   GEMINI_API_KEY=your_gemini_key_here
   PINECONE_API_KEY=your_pinecone_key_here
   API_KEY=hackrx-api-key
   ```

3. **Run Server**
   ```bash
   python main.py
   # Server runs on http://localhost:8000
   ```

4. **Test API**
   ```bash
   python test_script.py
   ```

### Docker Deployment

```bash
# Build and run with Docker
docker build -t hackrx-api .
docker run -p 8000:8000 --env-file .env hackrx-api
```

### Platform Deployment

**Heroku:**
```bash
heroku create your-app-name
heroku config:set GEMINI_API_KEY=your_key
heroku config:set PINECONE_API_KEY=your_key
git push heroku main
```

**Railway/Render:**
- Connect GitHub repository
- Set environment variables
- Deploy automatically

## 📊 Performance & Evaluation

### Target Metrics
- **Response Time**: < 30 seconds
- **Accuracy**: High precision in query understanding
- **Token Efficiency**: Optimized LLM usage
- **Scalability**: Handles concurrent requests

### Evaluation Criteria
- **Accuracy**: Precision of query understanding and clause matching
- **Token Efficiency**: Optimized LLM token usage and cost-effectiveness  
- **Latency**: Response speed and real-time performance
- **Reusability**: Code modularity and extensibility
- **Explainability**: Clear decision reasoning and clause traceability

## 🔧 Configuration

### Environment Variables
```bash
GEMINI_API_KEY=<required>         # Google Gemini API key
PINECONE_API_KEY=<optional>       # Pinecone vector DB (FAISS fallback)
API_KEY=hackrx-api-key           # Bearer token for auth
REDIS_URL=redis://localhost:6379  # Caching (optional)
PORT=8000                        # Server port
```

### System Tuning
- **Chunk Size**: ~1000 tokens per chunk
- **Top-K Retrieval**: 5 most relevant chunks
- **Concurrent Requests**: Up to 10 simultaneous
- **Cache TTL**: 1 hour for repeated requests

## 🧪 Testing

### Health Check
```bash
curl http://localhost:8000/health
```

### Full Test Suite
```bash
python test_script.py
```

### Manual Testing
```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Authorization: Bearer hackrx-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/policy.pdf",
    "questions": ["What is the grace period for premium payment?"]
  }'
```

## 📁 Project Structure

```
hackrx-llm-system/
├── main.py                 # FastAPI application
├── document_processor.py   # Document parsing & chunking
├── vector_store.py        # Embedding & search logic
├── llm_client.py          # Gemini LLM integration
├── test_script.py         # API testing suite
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Multi-service setup
├── .env.example          # Environment template
├── deploy.sh             # Deployment automation script
└── README.md            # This file
```

## 🚀 Deployment Guide

### Quick Deployment Checklist
- [ ] Set all required environment variables
- [ ] Verify API is accessible via HTTPS
- [ ] Test with sample requests
- [ ] Confirm response format matches specification
- [ ] Validate response time < 30 seconds

### Automated Deployment
```bash
# Make deploy script executable
chmod +x deploy.sh

# Run deployment script
./deploy.sh

# Choose deployment option:
# 1. Deploy to Heroku (automated)
# 2. Railway deployment instructions
# 3. Render deployment instructions
# 4. Run local tests only
# 5. Docker build and test
```

### Manual Heroku Deployment
```bash
# Login and create app
heroku login
heroku create your-hackrx-app

# Set environment variables
heroku config:set GEMINI_API_KEY=your_key_here
heroku config:set PINECONE_API_KEY=your_key_here

# Deploy
git add .
git commit -m "Deploy HackRx API"
git push heroku main

# Test deployed API
curl https://your-hackrx-app.herokuapp.com/health
```

### Railway Deployment
1. Connect GitHub repository to Railway
2. Set environment variables in dashboard:
   - `GEMINI_API_KEY`
   - `PINECONE_API_KEY`
   - `API_KEY=hackrx-api-key`
3. Deploy automatically

### Render Deployment
1. Connect GitHub repository to Render
2. Set build command: `pip install -r requirements.txt && python -m spacy download en_core_web_sm`
3. Set start command: `gunicorn main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT`
4. Set environment variables
5. Deploy

## 🎯 HackRx 6.0 Submission

### Submission Format

**Webhook URL:**
```
https://your-domain.com/api/v1/hackrx/run
```

**Tech Stack Description:**
```
FastAPI + Gemini-2.0-flash + Pinecone vector search + PyMuPDF + sentence-transformers embeddings
```

### Pre-Submission Checklist

- [x] **API Endpoint**: `POST /api/v1/hackrx/run` implemented
- [x] **Authentication**: Bearer token authentication working
- [x] **HTTPS**: Deployed with SSL certificate
- [x] **Request Format**: Accepts single `documents` string and `questions` array
- [x] **Response Format**: Returns `{"answers": ["string1", "string2"]}` format
- [x] **Response Time**: < 30 seconds response guarantee
- [x] **Error Handling**: Graceful failure and meaningful error messages
- [x] **Documentation**: Clear API specification and usage examples

### Expected Evaluation Process

1. **Request Sent**: Platform sends POST to your webhook
2. **Document Processing**: Your API downloads and parses documents
3. **Query Processing**: System processes questions using LLM
4. **Response Returned**: API returns answers array
5. **Evaluation**: Platform scores answer quality and performance

## 🏆 Key Features

### Advanced Document Processing
- **Multi-format Support**: PDFs, DOCX, emails
- **Robust Parsing**: Handles corrupted/complex documents
- **Page-aware Chunking**: Preserves document structure
- **Metadata Extraction**: Tracks source locations

### Intelligent Query Processing
- **Entity Extraction**: Identifies key entities (age, procedures, amounts)
- **Semantic Search**: Vector similarity matching
- **Context Ranking**: Smart chunk prioritization
- **LLM Integration**: Advanced reasoning with Gemini

### Production Features
- **Caching System**: Redis-based response caching
- **Error Recovery**: Robust fallback mechanisms
- **Monitoring**: Health checks and metrics
- **Scalability**: Concurrent request handling

### Pinecone Integration (Updated)
- **Latest API**: Uses `pinecone-client>=3.0.0`
- **Simplified Setup**: No environment parameter needed
- **Auto Fallback**: FAISS backup if Pinecone fails
- **Free Tier Optimized**: Works within free tier limits

## 🔍 Troubleshooting

### Common Issues

**Issue**: `PineconeException: Environment parameter is deprecated`
- **Fix**: Remove `PINECONE_ENVIRONMENT` from your `.env` file

**Issue**: `Authentication failed`
- **Fix**: Verify your API keys are correct and active

**Issue**: `Import error: No module named 'pinecone'`
- **Fix**: Update to latest requirements: `pip install pinecone-client>=3.0.0`

**Issue**: `Request timeout`
- **Fix**: Check document URL accessibility and LLM API limits

### Debug Steps
1. Check health endpoint: `curl http://localhost:8000/health`
2. Verify environment variables are set
3. Run test script: `python test_script.py`
4. Check application logs for detailed error messages

## 🤝 Support

For technical issues or questions:
1. Check the health endpoint: `/health`
2. Review logs for error details
3. Verify environment variables are set
4. Test with the provided test script

## 📝 License

Built for HackRx 6.0 Competition - Educational/Competition Use

---

## 🎉 Ready to Compete!

Your system is now ready for the HackRx 6.0 competition! 

### Final Steps:
1. **Deploy** to your chosen platform (Heroku/Railway/Render)
2. **Test** the deployed URL with the test script
3. **Submit** your webhook URL to the competition platform
4. **Monitor** performance and iterate as needed

### Submission URL Format:
```
https://your-app-name.herokuapp.com/api/v1/hackrx/run
https://your-app-name.railway.app/api/v1/hackrx/run
https://your-app-name.onrender.com/api/v1/hackrx/run
```

### Competition Tips:
- **Test thoroughly** with different document types
- **Monitor response times** to stay under 30 seconds
- **Check logs** regularly for any issues
- **Keep API keys secure** and never commit them to git

Good luck! 🏆 May your LLM-powered system excel in the competition!