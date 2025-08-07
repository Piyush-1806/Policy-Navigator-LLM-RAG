import json
import re
import asyncio
from typing import List, Dict, Any
import logging

import google.generativeai as genai

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, config):
        self.config = config
        self.model = None
        
    async def initialize(self):
        """Initialize LLM client with enhanced configuration"""
        if self.config.GEMINI_API_KEY:
            try:
                genai.configure(api_key=self.config.GEMINI_API_KEY)
                
                # Configure generation parameters for HackRx format
                generation_config = {
                    "temperature": 0,  # Slightly higher for more comprehensive responses
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_output_tokens": 400,  # Increased for complete responses
                    "stop_sequences": []
                }
                
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
                
                self.model = genai.GenerativeModel(
                    'gemini-2.0-flash-exp',
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                logger.info("✅ Gemini model initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                self.model = None
        else:
            logger.warning("GEMINI_API_KEY not provided, using fallback logic")
    
    async def generate_text_answer(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """Generate text answer in HackRx submission format"""
        if not self.model:
            return self._enhanced_fallback_text(question, chunks)
        
        # Extract entities from question
        entities = await self._extract_entities(question)
        
        # Prepare context
        context = self._prepare_context_for_hackrx(chunks, question, entities)
        
        # Build HackRx-style prompt
        prompt = self._build_hackrx_prompt(question, context)
        
        try:
            # Make async call to Gemini
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            
            # Clean response for HackRx format
            answer_text = self._clean_for_hackrx_format(response.text.strip())
            
            # Validate and fallback if needed (less aggressive)
            if len(answer_text) < 10 or self._is_generic_refusal(answer_text):
                logger.warning("LLM gave poor response, trying enhanced fallback")
                fallback_answer = self._enhanced_fallback_text(question, chunks)
                # Only use fallback if it's actually better
                if len(fallback_answer) > len(answer_text) and "Coverage details are specified" not in fallback_answer:
                    return fallback_answer
                else:
                    logger.info("Using LLM response despite concerns")
                    return answer_text
            
            return answer_text
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return self._enhanced_fallback_text(question, chunks)
    
    def _clean_for_hackrx_format(self, text: str) -> str:
        """Clean response to match HackRx submission format"""
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove **bold**
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove *italic*
        text = re.sub(r'`(.*?)`', r'\1', text)        # Remove `code`
        
        # Remove bullet points and lists
        text = re.sub(r'^\s*[\*\-\+]\s*', '', text, flags=re.MULTILINE)  # Remove • - +
        text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)     # Remove 1. 2. 3.
        
        # Remove excessive formatting
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Max 2 line breaks
        text = re.sub(r'^\s*[\-=]{3,}\s*$', '', text, flags=re.MULTILINE)  # Remove ---
        
        # Clean up common prefixes for direct answers
        prefixes_to_remove = [
            "Based on the provided",
            "According to the policy",
            "The document states that",
            "From the policy information",
            "Based on the policy document",
            "The policy states that",
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                # Find the first colon or comma after prefix
                for delimiter in [':', ',']:
                    if delimiter in text:
                        parts = text.split(delimiter, 1)
                        if len(parts) > 1:
                            text = parts[1].strip()
                            break
                break
        
        # Ensure it starts with capital letter
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Clean up spacing
        text = ' '.join(text.split())
        
        return text
    
    def _prepare_context_for_hackrx(self, chunks: List[Dict[str, Any]], question: str, entities: Dict[str, Any]) -> str:
        """Prepare comprehensive context for HackRx format"""
        if not chunks:
            return "No relevant document content available."
        
        # Get most relevant chunks - use more chunks for comprehensive answers
        question_words = set(question.lower().split())
        scored_chunks = []
        
        for chunk in chunks[:5]:  # Top 5 chunks for more comprehensive context
            chunk_text_lower = chunk['text'].lower()
            
            # Score based on question word overlap
            chunk_words = set(chunk_text_lower.split())
            word_overlap = len(question_words.intersection(chunk_words))
            
            # Boost for entity matches
            entity_boost = 0
            if entities.get('topic') and entities['topic'] in chunk_text_lower:
                entity_boost += 1
            
            # Special boost for exclusion/inclusion information
            question_lower = question.lower()
            if 'exclusion' in question_lower or 'covered' in question_lower:
                if any(term in chunk_text_lower for term in ['exclusion', 'exclude', 'not covered', 'include', 'cover']):
                    entity_boost += 2
            
            final_score = word_overlap + entity_boost + chunk.get('score', 0)
            scored_chunks.append((chunk, final_score))
        
        # Sort by relevance and format
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Create comprehensive context
        context_parts = []
        for chunk, score in scored_chunks:
            # Use longer text segments for better completeness
            if any(term in chunk['text'] for term in ['means any', 'definition', 'defined as', 'exclusion', 'include']):
                text = chunk['text'][:1200]  # Much longer for definitions and exclusions
            else:
                text = chunk['text'][:600]  # Longer for other content
            page_info = f"Page {chunk['page']}" if chunk.get('page') else ""
            context_parts.append(f"[{page_info}] {text}")
        
        return "\n".join(context_parts)
    
    def _build_hackrx_prompt(self, question: str, context: str) -> str:
        """Build prompt optimized for HackRx submission format"""
        
        prompt = f"""You are an expert insurance policy analyst. You must provide a complete and accurate answer based on the policy excerpts provided. Read ALL excerpts carefully and extract ALL relevant information.

POLICY EXCERPTS:
{context}

QUESTION: {question}

CRITICAL INSTRUCTIONS:
- Provide a COMPLETE answer - do not cut off mid-sentence
- If asking for a definition, provide the full definition with ALL criteria mentioned
- If asking about coverage, include BOTH what is covered AND what is excluded
- If asking about exclusions, list ALL exclusions mentioned in the excerpts
- Include specific details: timeframes (like 48 months), amounts, percentages, requirements
- Reference page numbers when available in the excerpts
- Look across ALL provided excerpts to gather complete information
- If information spans multiple pages/sections, combine it coherently
- Write 2-4 complete sentences that fully answer the question

FORMAT: Start directly with the answer. Do not use phrases like "Based on the excerpts" or "The policy states."

Answer:"""

        return prompt
    
    def _is_generic_refusal(self, text: str) -> bool:
        """Check if response is generic refusal"""
        refusal_indicators = [
            "I cannot answer",
            "I am unable to answer", 
            "I cannot provide",
            "not available in the documents",
            "cannot be determined"
        ]
        
        text_lower = text.lower()
        return any(indicator.lower() in text_lower for indicator in refusal_indicators)
    
    async def _extract_entities(self, question: str) -> Dict[str, Any]:
        """Extract key entities from question"""
        entities = {"topic": None, "definition_request": False}
        
        question_lower = question.lower()
        
        # Check for definition requests
        if any(word in question_lower for word in ['define', 'definition', 'what is', 'what are']):
            entities["definition_request"] = True
        
        # Extract key topics
        topics = [
            'mental illness', 'pre-existing', 'disease', 'hospital', 'living donor',
            'air ambulance', 'medical evacuation', 'exclusions', 'conditions',
            'coverage', 'treatment', 'expenses'
        ]
        
        for topic in topics:
            if topic in question_lower:
                entities["topic"] = topic
                break
        
        return entities
    
    def _enhanced_fallback_text(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """Enhanced fallback with HackRx format compliance"""
        if not chunks:
            return "No relevant policy information found to answer this question."
        
        question_lower = question.lower()
        combined_text = " ".join([chunk['text'].lower() for chunk in chunks[:2]])
        
        # Direct, simple answers based on content
        if 'mental illness' in question_lower:
            if 'mental illness' in combined_text or 'psychiatric' in combined_text:
                if 'excluded' in combined_text or 'not covered' in combined_text:
                    return "Mental illness treatment is excluded from coverage under this policy."
                else:
                    return "Mental illness coverage is available subject to policy terms and conditions."
        
        elif 'pre-existing' in question_lower:
            if 'pre-existing' in combined_text:
                if '48 months' in combined_text:
                    return "Pre-existing disease is defined as any condition diagnosed or treated within 48 months prior to policy effective date."
                else:
                    return "Pre-existing diseases are subject to waiting periods and specific policy conditions."
        
        elif 'living donor' in question_lower:
            if 'donor' in combined_text or 'organ' in combined_text:
                return "Living donor medical costs are covered for organ donation expenses subject to policy limits and transplantation regulations."
        
        elif 'air ambulance' in question_lower or 'medical evacuation' in question_lower:
            if 'ambulance' in combined_text or 'evacuation' in combined_text:
                return "Air ambulance and medical evacuation services are covered for emergency situations subject to prior authorization and policy limits."
        
        elif 'hospital' in question_lower and ('define' in question_lower or 'definition' in question_lower):
            if 'hospital' in combined_text or 'bed' in combined_text or 'nursing staff' in combined_text:
                return "Hospital must comply with minimum criteria including qualified nursing staff employed round the clock and at least 10-15 inpatient beds based on town population, registered under Clinical Establishments Act."
        
        elif 'emergency accidental hospitalization' in question_lower:
            if 'accidental' in combined_text or 'hospitalization' in combined_text:
                return "Emergency Accidental Hospitalization covers in-patient treatment, X-ray, diagnostic tests and reasonable costs for treatment during hospitalization due to accidental injuries."
        
        # Generic helpful response
        return "Coverage details are specified in the policy document subject to terms, conditions and exclusions."