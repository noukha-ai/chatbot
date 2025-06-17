import gradio as gr
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch
import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4
from enum import Enum

# Load environment variables
load_dotenv()

# Hardcoded configuration values
QDRANT_COLLECTION_NAME = "all_tasks_data"
QDRANT_VECTOR_SIZE = 384
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.0-flash"  # Updated to Gemini model name
SEARCH_DEFAULT_K = 15
SEARCH_MAX_QUERY_VARIATIONS = 5
SEARCH_MAX_UNIQUE_DOCS = 20
GENERATION_TEMPERATURE = 0.3
GENERATION_TOP_P = 0.9
GENERATION_TOP_K = 40
LOGGING_FILE = "chatbot.log"
LOGGING_LEVEL = "INFO"

# Database fields with enhanced categorization
DATABASE_FIELDS = {
    "client": ["client_id", "client_name", "client_stage_id", "client_industry_id", 
               "client_sales_owner_id", "client_cs_owner_id", "client_recurring_revenue", 
               "client_cs_owner", "client_sales_owner"],
    "project": ["project_id", "project_name", "project_description", "project_stage_id", 
                "project_priority_id", "project_planned_start", "project_planned_end", 
                "project_actual_start", "project_actual_end", "project_delivery_manager_id", 
                "project_delivery_manager"],
    "usecase": ["usecase_id", "usecase_name", "usecase_description", "usecase_stage_id", 
                "usecase_priority_id", "usecase_planned_start", "usecase_planned_end", 
                "usecase_actual_start", "usecase_actual_end", "usecase_delivery_manager_id", 
                "usecase_delivery_manager"],
    "phase": ["phase_id", "phase_name"],
    "task": ["task_id", "task_name", "task_description", "task_stage_id", "task_priority_id", 
             "task_planned_start", "task_planned_end", "task_actual_start", "task_actual_end", 
             "task_owner_id", "task_owner"],
    "other": ["custom_fields"]
}

class QueryType(Enum):
    """Enumeration for query types"""
    DATABASE_REQUIRED = "DATABASE_REQUIRED"
    DIRECT_ANSWER = "DIRECT_ANSWER"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGGING_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TransformersEmbeddings(Embeddings):
    """Custom embeddings class using transformers"""
    
    def __init__(self, model_name: str = SENTENCE_TRANSFORMER_MODEL):
        self.model_name = model_name
        self._load_model()
    
    def _load_model(self):
        """Load the transformer model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            logger.info(f"Successfully loaded {self.model_name} model")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()[0].tolist()

class DocumentProcessor:
    """Handles document processing and formatting"""
    
    @staticmethod
    def clean_json_artifacts(text: str) -> str:
        """Clean JSON artifacts from text fields"""
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        
        # Remove common JSON artifacts
        text = re.sub(r'\{"root":\s*\{[^}]*\}\}', '', text)
        text = re.sub(r'\[\{"children":[^}]*\}\]', '', text)
        text = re.sub(r'"paragraph",\s*"version":\s*1', '', text)
        text = re.sub(r'"direction":\s*"ltr"', '', text)
        text = re.sub(r'"format":\s*""', '', text)
        text = re.sub(r'"indent":\s*0', '', text)
        text = re.sub(r'"type":\s*"root"', '', text)
        text = re.sub(r'\{[^}]*"version":\s*1[^}]*\}', '', text)
        
        # Clean up whitespace and special characters
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[{}"\[\],:]', ' ', text)
        text = text.strip()
        
        return text if text else "Not specified"
    
    @staticmethod
    def format_date(date_str: str) -> str:
        """Format date string to readable format"""
        if not date_str or date_str == "Not specified":
            return "Not specified"
        
        try:
            # Handle various date formats
            if 'T' in date_str:
                date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                return date_obj.strftime("%B %d, %Y")
            return date_str
        except:
            return date_str
    
    @staticmethod
    def calculate_days_difference(start_date: str, end_date: str) -> Optional[int]:
        """Calculate days between two dates"""
        try:
            if not start_date or not end_date or start_date == "Not specified" or end_date == "Not specified":
                return None
            
            start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            return (end - start).days
        except:
            return None

class QueryAnalyzer:
    """Analyzes queries to determine if database fetch is required"""
    
    def __init__(self, response_generator):
        self.response_generator = response_generator
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine processing approach"""
        
        analysis_prompt = f"""
You are an intelligent query analyzer for a project management chatbot. Your job is to determine whether a user query requires database search or can be answered directly.

**User Query:** "{query}"
**Current Date:** {datetime.today().strftime("%Y-%m-%d")}

**System Context:**
You are a project management assistant that can access data about:
- Clients (names, industries, revenue, owners, stages)
- Projects (names, descriptions, stages, priorities, timelines, managers)
- Use Cases (project components with their own timelines and managers) 
- Phases (use case components)
- Tasks (specific work items with owners, priorities, timelines)

**Decision Framework:**

**DATABASE_REQUIRED** - Query needs data from the database:
- Asks about specific entities (client names, project names, task details)
- Requests current status, progress, or timeline information
- Asks for lists, summaries, or analysis of project data
- Inquires about team members, assignments, or ownership
- Asks about deadlines, overdue items, or scheduling
- Requests performance metrics or business insights
- Any question that requires specific project management data

**DIRECT_ANSWER** - Can answer without database (but still in scope):
- General project management advice or best practices
- How to use project management concepts
- Definitions of project management terms
- General workflow or methodology questions
- System capabilities or feature explanations
- Simple greetings with context transition to PM topics

**OUT_OF_SCOPE** - Not related to project management:
- Personal questions unrelated to work
- General knowledge questions (weather, news, etc.)
- Technical support for non-PM software
- Medical, legal, or financial advice
- Entertainment, jokes, or casual conversation without PM context
- Questions about topics completely unrelated to project management

**Analysis Rules:**
1. If the query mentions specific names, dates, or project details → DATABASE_REQUIRED
2. If it's a general PM question that can be answered with knowledge → DIRECT_ANSWER  
3. If it's completely unrelated to project management → OUT_OF_SCOPE
4. When in doubt between DATABASE_REQUIRED and DIRECT_ANSWER, choose DATABASE_REQUIRED
5. Be strict about OUT_OF_SCOPE - only project management related queries are acceptable

**Output Format:**
Return ONLY a JSON object:
{{
    "query_type": "DATABASE_REQUIRED|DIRECT_ANSWER|OUT_OF_SCOPE",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of the decision",
    "entities_detected": ["list", "of", "detected", "entities"],
    "requires_db_fetch": true/false,
    "suggested_response_approach": "Brief description of how to handle this query"
}}

**Examples:**

Query: "Show me all overdue tasks for Project Alpha"
{{
    "query_type": "DATABASE_REQUIRED",
    "confidence": 0.95,
    "reasoning": "Asks for specific data about Project Alpha's overdue tasks",
    "entities_detected": ["Project Alpha", "overdue tasks"],
    "requires_db_fetch": true,
    "suggested_response_approach": "Search database for Project Alpha tasks with overdue status"
}}

Query: "What are the best practices for project planning?"
{{
    "query_type": "DIRECT_ANSWER",
    "confidence": 0.90,
    "reasoning": "General project management knowledge question",
    "entities_detected": [],
    "requires_db_fetch": false,
    "suggested_response_approach": "Provide general project planning best practices"
}}

Query: "What's the weather like today?"
{{
    "query_type": "OUT_OF_SCOPE",
    "confidence": 0.95,
    "reasoning": "Weather query is not related to project management",
    "entities_detected": [],
    "requires_db_fetch": false,
    "suggested_response_approach": "Politely redirect to project management topics"
}}
"""

        try:
            response_text = self.response_generator.generate_response(analysis_prompt)
            if not response_text:
                logger.warning(f"No response generated for query analysis: {query}")
                raise ValueError("No response generated")

            # Clean the response
            response_text = re.sub(r'```json\n|```', '', response_text).strip()
            # Parse JSON response
            analysis = json.loads(response_text)
            
            # Validate required fields
            required_fields = ["query_type", "confidence", "reasoning", "requires_db_fetch"]
            if not all(field in analysis for field in required_fields):
                logger.warning(f"Missing required fields in analysis response")
                return self._get_fallback_analysis(query)
            
            # Validate query_type
            valid_types = ["DATABASE_REQUIRED", "DIRECT_ANSWER", "OUT_OF_SCOPE"]
            if analysis["query_type"] not in valid_types:
                logger.warning(f"Invalid query_type: {analysis['query_type']}")
                raise ValueError("Invalid query_type")

            logger.info(f"Query analysis: {analysis['query_type']} (confidence: {analysis['confidence']})")
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise
        except Exception as e:
            logger.error(f"Error analyzing query '{query}': {e}")
            raise

    def _get_fallback_analysis(self, query: str) -> Dict[str, Any]:
        """Provide fallback analysis if primary analysis fails"""
        return {
            "query_type": "DIRECT_ANSWER",
            "confidence": 0.5,
            "reasoning": "Fallback analysis due to processing error",
            "entities_detected": [],
            "requires_db_fetch": False,
            "suggested_response_approach": "Provide general response or ask for clarification"
        }

class QdrantSearcher:
    """Handles Qdrant vector database operations"""
    
    def __init__(self, embeddings: TransformersEmbeddings):
        self.embeddings = embeddings
        self.collection_name = QDRANT_COLLECTION_NAME
        self.client = self._initialize_client()
        self._ensure_collection()

    def _initialize_client(self) -> QdrantClient:
        """Initialize Qdrant client with environment variables"""
        try:
            qdrant_url = os.getenv('QDRANT_URL')
            qdrant_api_key = os.getenv('QDRANT_API_KEY')
            
            client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key
            )
            client.get_collections()
            logger.info("Successfully connected to Qdrant")
            return client
            
        except Exception as e:
            logger.error(f"Error initializing Qdrant client: {e}")
            raise
    
    def _ensure_collection(self) -> None:
        """Create or verify Qdrant collection with HNSW indexing"""
        try:
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=QDRANT_VECTOR_SIZE,
                        distance=models.Distance.COSINE,
                        hnsw_config=models.HnswConfigDiff(
                            m=16,
                            ef_construct=100,
                            full_scan_threshold=10000
                        )
                    )
                )
                logger.info(f"Created Qdrant collection {self.collection_name} with HNSW indexing")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection {self.collection_name}: {e}")
            raise
    
    def search_documents(self, query: str, k: int = SEARCH_DEFAULT_K) -> List[Document]:
        """Search documents in Qdrant"""
        try:
            query_vector = self.embeddings.embed_query(query)
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=k,
                with_payload=True,
            )
            
            docs = []
            for result in search_results:
                payload = result.payload
                # Process and clean the payload
                cleaned_payload = self._clean_payload(payload)
                page_content = self._format_document_content(cleaned_payload)
                metadata = {k: v for k, v in cleaned_payload.items()}
                metadata['similarity_score'] = result.score
                docs.append(Document(page_content=page_content, metadata=metadata))
            
            logger.info(f"Found {len(docs)} documents for query: {query}")
            return docs
            
        except Exception as e:
            logger.error(f"Error searching Qdrant for query '{query}': {e}")
            return []
    
    def _clean_payload(self, payload: Dict) -> Dict:
        """Clean and process payload data"""
        cleaned = {}
        for key, value in payload.items():
            if key.startswith('_'):
                continue
            
            if isinstance(value, str):
                cleaned_value = DocumentProcessor.clean_json_artifacts(value)
            else:
                cleaned_value = value
            
            cleaned[key] = cleaned_value
        
        return cleaned
    
    def _format_document_content(self, payload: Dict) -> str:
        """Format document content in a structured way"""
        content_sections = []
        
        # Group fields by category
        for category, fields in DATABASE_FIELDS.items():
            category_data = {}
            for field in fields:
                if field in payload and payload[field] is not None:
                    value = payload[field]
                    if 'date' in field.lower() or 'start' in field.lower() or 'end' in field.lower():
                        value = DocumentProcessor.format_date(str(value))
                    elif 'revenue' in field.lower() and isinstance(value, (int, float)):
                        value = f"${value:,.2f}"
                    category_data[field] = value
            
            if category_data:
                section = f"[{category.upper()}] " + " | ".join([
                    f"{key.replace('_', ' ').title()}: {value}" 
                    for key, value in category_data.items()
                ])
                content_sections.append(section)
        
        return "\n".join(content_sections)

class QueryRewriter:
    """Enhanced query rewriter for better semantic search"""
    
    def __init__(self, response_generator):
        self.response_generator = response_generator
    
    def rewrite_query(self, query: str) -> List[str]:
        """Generate multiple semantic variations of the query"""
        if not query or not query.strip():
            return [query]

        rewrite_prompt = f"""
You are an expert query rewriter for a project management system. Generate 4-5 diverse query variations that will improve semantic search results.

**Original Query:** "{query}"
**Current Date:** {datetime.today().strftime("%Y-%m-%d")}

**Database Context:**
- Clients: Have names, stages, industries, revenue, sales/CS owners
- Projects: Have names, descriptions, stages, priorities, dates, delivery managers
- Use Cases: Belong to projects, have stages, priorities, dates, managers
- Phases: Belong to use cases, have names
- Tasks: Belong to phases, have names, descriptions, stages, priorities, dates, owners

**Rewriting Strategy:**
1. **Intent Preservation**: Keep the core meaning and entities (names, dates)
2. **Semantic Expansion**: Add synonyms and related terms
   - "status" → "progress", "state", "condition", "stage"
   - "deadline" → "due date", "end date", "completion date"
   - "delay" → "behind schedule", "overdue", "late"
   - "team member" → "owner", "assignee", "responsible person"
3. **Business Context**: Include relevant business terms
   - "at risk" → "delayed", "behind schedule", "overdue"
   - "progress" → "completion status", "current state"
4. **Entity Variations**: Use different ways to reference entities
   - "Project X tasks" → "tasks in Project X", "Project X task list"
5. **Question Reformulation**: Convert between question types
   - "What is..." → "Show me...", "List...", "Find..."

**Output Format:**
Return ONLY a JSON array:
[
    {{"query": "variation1", "focus": "semantic expansion"}},
    {{"query": "variation2", "focus": "business context"}},
    {{"query": "variation3", "focus": "entity variation"}},
    {{"query": "variation4", "focus": "question reformulation"}},
    {{"query": "variation5", "focus": "comprehensive"}}
]

**Example:**
Original: "Show me overdue tasks for Project Alpha"
Output:
[
    {{"query": "Project Alpha tasks behind schedule", "focus": "semantic expansion"}},
    {{"query": "Project Alpha delayed activities", "focus": "business context"}},
    {{"query": "late tasks in Project Alpha", "focus": "entity variation"}},
    {{"query": "Project Alpha overdue assignments", "focus": "question reformulation"}},
    {{"query": "Project Alpha tasks past due date", "focus": "comprehensive"}}
]
"""

        try:
            response_text = self.response_generator.generate_response(rewrite_prompt)
            if not response_text:
                return [query]
            
            response_text = re.sub(r'```json\n|```', '', response_text).strip()
            result = json.loads(response_text)
            
            if isinstance(result, list) and result:
                queries = []
                for item in result[:SEARCH_MAX_QUERY_VARIATIONS]:
                    if isinstance(item, dict) and "query" in item and item["query"].strip():
                        queries.append(item["query"].strip())
                
                # Always include original query
                if query not in queries:
                    queries.insert(0, query)
                
                return queries[:SEARCH_MAX_QUERY_VARIATIONS]
            
            return [query]
            
        except Exception as e:
            logger.error(f"Error rewriting query '{query}': {e}")
            return [query]

class ResponseGenerator:
    """Enhanced response generator with conversational capabilities"""
    
    def __init__(self):
        self.model = self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Gemini API model"""
        try:
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                generation_config={
                    "temperature": GENERATION_TEMPERATURE,
                    "top_p": GENERATION_TOP_P,
                    "top_k": GENERATION_TOP_K
                }
            )
            logger.info(f"Successfully initialized Gemini model {GEMINI_MODEL}")
            return model
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {e}")
            raise
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using Gemini API"""
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""
    
    def generate_direct_response(self, query: str) -> str:
        """Generate direct response for general project management questions"""
        
        direct_response_prompt = f"""
You are a professional project management assistant. The user has asked a general question that doesn't require database lookup.

**User Query:** "{query}"
**Current Date:** {datetime.today().strftime("%B %d, %Y")}

**Response Guidelines:**
- Provide helpful, accurate project management guidance
- Be conversational and professional
- Keep responses concise but informative
- If it's a greeting, acknowledge warmly and offer to help with project management
- For general PM questions, provide best practices and actionable advice
- Do not make up specific data or mention specific projects/clients
- If the question is completely unrelated to project management, politely redirect

**Response Style:**
- Professional but friendly tone
- Use bullet points for lists when appropriate
- Include practical examples when helpful
- End with an offer to help with specific project data if needed

**Example Response Types:**

*For greeting: "Hello"*
"Hello! I'm your project management assistant. I can help you with information about your projects, tasks, deadlines, team assignments, and client details. What would you like to know about today?"

*For general PM question: "What are the best practices for project planning?"*
"Great question! Here are key project planning best practices:

• **Define clear objectives** - Establish specific, measurable goals
• **Break down work** - Create a detailed work breakdown structure  
• **Estimate resources** - Consider time, budget, and team capacity
• **Identify dependencies** - Map out task relationships and constraints
• **Plan for risks** - Anticipate potential issues and mitigation strategies
• **Set realistic timelines** - Allow buffer time for unexpected delays

Would you like me to look up information about any specific projects you're currently planning?"

Generate your response now:
"""
        
        return self.generate_response(direct_response_prompt)
    
    def generate_out_of_scope_response(self, query: str) -> str:
        """Generate response for out-of-scope questions"""
        
        return f"""I'm a project management assistant focused on helping you with your projects, tasks, clients, and team coordination. 

I can help you with:
• Project status and progress tracking
• Task assignments and deadlines  
• Client information and relationships
• Team workload and capacity planning
• Timeline and milestone management
• Project analytics and reporting

Is there anything related to your projects or team that I can help you with today?"""

class ProjectChatbot:
    """Enhanced chatbot with tool calling and intelligent query routing"""
    
    def __init__(self):
        self._validate_environment()
        self.embeddings = TransformersEmbeddings()
        self.searcher = QdrantSearcher(self.embeddings)
        self.response_generator = ResponseGenerator()
        self.query_analyzer = QueryAnalyzer(self.response_generator)
        self.query_rewriter = QueryRewriter(self.response_generator)
        self.document_processor = DocumentProcessor()
    
    def _validate_environment(self):
        """Validate required environment variables"""
        required_env_vars = ['QDRANT_URL', 'QDRANT_API_KEY', 'GEMINI_API_KEY']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")
    
    def process_query(self, query: str) -> str:
        """Enhanced query processing with intelligent routing"""
        if not query or not query.strip():
            return "I'd be happy to help! Please ask me about clients, projects, tasks, or team members."

        try:
            # Step 1: Analyze the query to determine processing approach
            logger.info(f"Analyzing query: {query}")
            analysis = self.query_analyzer.analyze_query(query)
            
            query_type = analysis.get("query_type", "DIRECT_ANSWER")
            confidence = analysis.get("confidence", 0.0)
            reasoning = analysis.get("reasoning", "")
            
            logger.info(f"Query analysis: {query_type} (confidence: {confidence}) - {reasoning}")
            
            # Step 2: Route based on analysis
            if query_type == QueryType.OUT_OF_SCOPE.value:
                return self.response_generator.generate_out_of_scope_response(query)
            
            elif query_type == QueryType.DIRECT_ANSWER.value:
                return self.response_generator.generate_direct_response(query)
            
            elif query_type == QueryType.DATABASE_REQUIRED.value:
                return self._process_database_query(query, analysis)
            
            else:
                # Fallback to direct answer
                logger.warning(f"Unknown query type: {query_type}, defaulting to direct answer")
                return self.response_generator.generate_direct_response(query)
                
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            return "I encountered an issue while processing your request. Could you please rephrase your question?"
    
    def _process_database_query(self, query: str, analysis: Dict[str, Any]) -> str:
        """Process queries that require database search"""
        try:
            # Step 1: Generate semantic query variations
            logger.info(f"Generating query variations for: {query}")
            rewritten_queries = self.query_rewriter.rewrite_query(query)
            logger.info(f"Generated {len(rewritten_queries)} query variations")
            
            # Step 2: Search with multiple queries
            all_docs = []
            for q in rewritten_queries:
                docs = self.searcher.search_documents(q, k=SEARCH_DEFAULT_K)
                all_docs.extend(docs)
            
            # Step 3: Deduplicate and rank documents
            unique_docs = self._deduplicate_and_rank_documents(all_docs)
            
            # Step 4: Generate contextual response
            if not unique_docs:
                return self._generate_no_results_response(query)
            
            context = self._prepare_context(unique_docs[:SEARCH_MAX_UNIQUE_DOCS])
            return self._generate_conversational_response(query, context, unique_docs, analysis)
            
        except Exception as e:
            logger.error(f"Error processing database query '{query}': {e}")
            return "I encountered an issue while searching for that information. Could you please rephrase your question?"
    
    def _deduplicate_and_rank_documents(self, docs: List[Document]) -> List[Document]:
        """Deduplicate and rank documents by relevance"""
        unique_docs = []
        seen_content = set()
        
        # Sort by similarity score (highest first)
        docs.sort(key=lambda x: x.metadata.get('similarity_score', 0), reverse=True)
        
        for doc in docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                unique_docs.append(doc)
                seen_content.add(content_hash)
        
        return unique_docs
    
    def _prepare_context(self, docs: List[Document]) -> str:
        """Prepare structured context for response generation"""
        context_parts = []
        
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"Record {i}:\n{doc.page_content}\n")
        
        return "\n".join(context_parts)
    
    def _generate_conversational_response(self, query: str, context: str, docs: List[Document], analysis: Dict[str, Any]) -> str:
        """Generate a conversational, intelligent response"""
        
        entities_detected = analysis.get("entities_detected", [])
        
        response_prompt = f"""
You are a professional project management assistant. Provide a helpful, conversational response based on the database information.

**User Query:** "{query}"
**Detected Entities:** {entities_detected}
**Current Date:** {datetime.today().strftime("%B %d, %Y")}
**Analysis Confidence:** {analysis.get("confidence", 0.0)}

**Database Information:**
{context}

**Response Guidelines:**

**Tone & Style:**
- Be conversational, professional, and helpful
- Use natural language, not database field names
- Address the user directly when appropriate
- Show empathy for project challenges

**Content Structure:**
1. **Direct Answer**: Start with a clear answer to the question
2. **Key Details**: Provide relevant specifics (names, dates, status)
3. **Insights**: Add helpful context or observations
4. **Next Steps**: Suggest follow-up actions when appropriate

**Formatting:**
- Use bullet points for lists of items
- **Bold** important names, dates, and statuses
- Use natural date formats (e.g., "January 15, 2024" instead of "2024-01-15")
- Include context like "as of today" for time-sensitive information

**Response Types:**

**For Status Inquiries:**
- Current status with timeline context
- Progress indicators and completion percentages where applicable
- Highlight any delays or issues
- Mention responsible team members

**For List Requests:**
- Organized lists with key details
- Prioritize by urgency, date, or importance
- Include brief context for each item

**For Analysis Questions:**
- Provide insights and patterns
- Highlight risks or opportunities
- Suggest actionable recommendations
- Use data to support conclusions

**For Team/Resource Questions:**
- Focus on people and their responsibilities
- Mention workload and availability implications
- Suggest alternatives or solutions for resource conflicts

**Example Responses:**

*For "What projects are behind schedule?"*
"I found 3 projects that are currently behind their planned timeline:

• **Project Alpha** - Originally due March 15, now estimated for March 28 (2 weeks delay)
  - Issue: Key developer on sick leave
  - Owner: Sarah Johnson
  
• **Website Redesign** - 30% complete, should be 60% by now
  - Behind due to client feedback delays
  - Owner: Mike Chen

Would you like me to look into the specific tasks causing these delays?"

**Important:**
- If information is incomplete, acknowledge limitations
- Don't make assumptions beyond the provided data
- Offer to help find additional information
- Use "I found" or "According to the data" to show information source
- If no relevant data found, suggest alternative queries or broader searches
"""

        return self.response_generator.generate_response(response_prompt)
    
    def _generate_no_results_response(self, query: str) -> str:
        """Generate helpful response when no results are found"""
        suggestions = [
            "Try using different keywords (e.g., 'project' instead of 'initiative')",
            "Check if the client or project name is spelled correctly",
            "Ask about broader topics (e.g., 'all projects' instead of a specific project)",
            "Try asking about recent activities or current status"
        ]
        
        return f"""I couldn't find specific information matching your query: "{query}"

This could be because:
• The information might not be in our database yet
• The names or terms might be different than expected
• The data might be stored under different categories

Here are some suggestions to help me find what you're looking for:
{chr(10).join(f'• {suggestion}' for suggestion in suggestions)}
"""

# Gradio Interface
def chatbot_interface(query, chat_history):
    """Function to handle Gradio chatbot interactions"""
    chatbot = ProjectChatbot()
    response = chatbot.process_query(query)
    chat_history.append((query, response))
    return "", chat_history

def clear_chat():
    """Clear the chat history"""
    return []

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Project Management AI Assistant
    🚀 Welcome to the Project Management Assistant demo! Ask about projects, tasks, clients, or team members.
    💡 I can provide general project management advice or search specific project data.
    🔍 Type your question below and see how I can help!
    """)
    
    chatbot = gr.Chatbot(label="Conversation", height=400)
    query_input = gr.Textbox(
        label="Your Question",
        placeholder="e.g., 'Show me overdue tasks for Project Alpha' or 'What are best practices for project planning?'",
        lines=2
    )
    
    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear Chat")
    
    submit_btn.click(
        fn=chatbot_interface,
        inputs=[query_input, chatbot],
        outputs=[query_input, chatbot]
    )
    
    clear_btn.click(
        fn=clear_chat,
        inputs=None,
        outputs=[chatbot]
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch()