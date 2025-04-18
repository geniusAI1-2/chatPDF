=======================================
        PDF QA with Gemini API
=======================================

QUICK SETUP GUIDE

1. CLONE THE REPOSITORY
------------------------
git clone https://github.com/your-repo/pdf-qa-gemini.git  
cd pdf-qa-gemini  

2. CREATE VIRTUAL ENVIRONMENT (OPTIONAL)
----------------------------------------
python -m venv venv  

# On Linux/Mac:
source venv/bin/activate  

# On Windows:
venv\Scripts\activate  

3. INSTALL DEPENDENCIES
------------------------
pip install -r requirements.txt

4. ADD GEMINI API KEY
----------------------
GOOGLE_API_KEY=your_api_key_here(في ملف .env)

5. RUN THE SERVER
------------------
uvicorn main:app --reload

الواجهة التفاعلية للـ API متاحة على:
http://127.0.0.1:8000/docs

---------------------------------------
 API ENDPOINTS
---------------------------------------

1. UPLOAD PDF
--------------
curl -X POST -F "file=@document.pdf" http://localhost:8000/upload_pdf

2. ASK A QUESTION
-----------------
curl -X POST -H "Content-Type: application/json" ^
-d "{\"question\":\"What is this document about?\"}" ^
http://localhost:8000/ask

3. CHECK INDEX STATUS
----------------------
curl http://localhost:8000/status

---------------------------------------
NOTES
---------------------------------------

- يدعم فقط ملفات PDF النصية (لا يدعم صور).
- النموذج المستخدم افتراضياً للـ Embeddings هو: bge-small-en
  (يمكن تغييره من ملف main.py)
