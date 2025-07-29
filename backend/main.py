from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from classifier import ContractRiskClassifier
from contract_summarizer import EnhancedContractSummarizer
import os
import uuid
import json

app = FastAPI(title="LegalDecoder API", version="1.0.0")

# Initialize both components
print("Initializing AI models...")
try:
    classifier = ContractRiskClassifier("./legal-bert-risk-classifier")
    summarizer = EnhancedContractSummarizer()
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    classifier = None
    summarizer = None

# Directory setup
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production!
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Serve static files (for the frontend)
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def root():
    return {"message": "LegalDecoder Backend API running!", "models_loaded": classifier is not None}

@app.post("/analyze")
async def analyze_contract(file: UploadFile = File(...)):
    """
    Complete contract analysis pipeline:
    1. Upload PDF
    2. Classify contract clauses (ContractRiskClassifier)
    3. Generate comprehensive summary (EnhancedContractSummarizer)
    """
    
    # Check if models are loaded
    if classifier is None or summarizer is None:
        raise HTTPException(status_code=500, detail="AI models not loaded. Please check model files.")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    # Check file size (max 50MB)
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50MB).")
    
    # Generate unique identifiers
    unique_id = str(uuid.uuid4())
    filename = f"{unique_id}_{file.filename.replace(' ', '_')}"
    pdf_path = os.path.join(UPLOAD_DIR, filename)
    csv_path = os.path.join(OUTPUT_DIR, f"classification_{unique_id}.csv")
    json_path = os.path.join(OUTPUT_DIR, f"report_{unique_id}.json")
    
    # Save uploaded file
    try:
        with open(pdf_path, "wb") as f:
            f.write(content)
        print(f"📄 Saved uploaded file: {filename}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    try:
        # STEP 1: Classify contract clauses using Legal-BERT
        print(f"🔍 Step 1: Classifying contract clauses...")
        classification_results = classifier.process_pdf(
            pdf_path=pdf_path,
            output_csv=csv_path
        )
        
        if classification_results is None or len(classification_results) == 0:
            raise HTTPException(status_code=500, detail="Classification failed: No clauses could be processed.")
        
        print(f"✅ Classification complete: {len(classification_results)} clauses analyzed")
        
        # STEP 2: Generate comprehensive summary and analysis
        print(f"📊 Step 2: Generating comprehensive analysis...")
        report = summarizer.create_detailed_report(
            classification_csv=csv_path,
            contract_pdf_path=pdf_path,
            output_file=json_path
        )
        
        if report is None:
            raise HTTPException(status_code=500, detail="Summarization failed: Could not generate report.")
        
        print(f"✅ Analysis complete!")
        
        # Add processing metadata
        report['processing_info'] = {
            'file_name': file.filename,
            'file_size_mb': round(len(content) / (1024 * 1024), 2),
            'clauses_processed': len(classification_results),
            'risky_clauses_found': len(classification_results[classification_results['risk_classification'] == 'risky']),
            'processing_steps': [
                'PDF text extraction',
                'Clause segmentation', 
                'Legal-BERT risk classification',
                'Comprehensive analysis generation'
            ]
        }
        
    except Exception as e:
        # Cleanup on error
        cleanup_files([pdf_path, csv_path, json_path])
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    # Cleanup temporary files
    cleanup_files([pdf_path, csv_path, json_path])
    
    return JSONResponse(content=report)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "classifier_loaded": classifier is not None,
        "summarizer_loaded": summarizer is not None,
        "models_status": "ready" if (classifier and summarizer) else "not_ready"
    }

def cleanup_files(file_paths):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️ Cleaned up: {file_path}")
        except Exception as e:
            print(f"⚠️ Could not clean up {file_path}: {e}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting LegalDecoder API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
