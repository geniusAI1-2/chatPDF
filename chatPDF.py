import re
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()
app = FastAPI(title="PDF QA with Gemini")

DEFAULT_FAISS_INDEX = "faiss_index"

# Request Models
class QuestionRequest(BaseModel):
    question: str
    prev_question: str = ""
    use_existing_index: bool = True 

class UploadResponse(BaseModel):
    message: str
    index_path: str

# Initialize embeddings
hf_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Global vectorstore instance
vectorstore = None

def clean_response(text):
    return re.sub(r"[ﭼ-ﯿ]+", "", text)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def initialize_vectorstore(pdf_path: Optional[str] = None, index_path: str = DEFAULT_FAISS_INDEX):
    global vectorstore
    
    if os.path.exists(index_path) and pdf_path is None:
        print("Loading existing FAISS index...")
        vectorstore = FAISS.load_local(index_path, hf_embeddings, allow_dangerous_deserialization=True)
    elif pdf_path:
        print("Processing new PDF...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
        splits = text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(splits, hf_embeddings)
        vectorstore.save_local(index_path)
        print(f"Saved new index to {index_path}")
    else:
        raise HTTPException(status_code=400, detail="Neither PDF nor existing index provided")
    
    return vectorstore

#LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

prompt = ChatPromptTemplate.from_template("""
أنت مساعد ذكي يساعد الطلاب في جميع المواد الدراسية. إذا كان السؤال يحتوي على معادلات أو مسائل رياضية أو علمية، قم بحلها بدقة حتى لو لم تكن موجودة في النص. 
ركّز على الفهم والتحليل، ولا تعتمد فقط على السياق النصي.

السياق:
{context}

السؤال السابق: {prev_question}
السؤال الحالي: {question}

الإجابة:
""")


#endpoints
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), index_path: str = DEFAULT_FAISS_INDEX):
    try:
        # Save uploaded PDF 
        temp_pdf = "temp_uploaded.pdf"
        with open(temp_pdf, "wb") as f:
            f.write(await file.read())
        
        # Process PDF 
        initialize_vectorstore(pdf_path=temp_pdf, index_path=index_path)
        os.remove(temp_pdf)  
        
        return UploadResponse(
            message="تم معالجة الملف وحفظ الفهرس بنجاح",
            index_path=index_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QuestionRequest, index_path: str = DEFAULT_FAISS_INDEX):
    global vectorstore
    
    try:
        if vectorstore is None:
            initialize_vectorstore(index_path=index_path)
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 40})
        
        chain = (
            {
                "context": retriever | format_docs,
                "prev_question": lambda x: request.prev_question,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        response = chain.invoke(request.question)
        return {"answer": clean_response(response)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def check_status(index_path: str = DEFAULT_FAISS_INDEX):
    index_exists = os.path.exists(index_path)
    return {
        "index_exists": index_exists,
        "index_path": index_path,
        "ready_for_queries": index_exists or (vectorstore is not None)
    }