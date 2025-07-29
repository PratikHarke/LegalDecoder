import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pypdf import PdfReader
import re
import os

class ContractRiskClassifier:
    def __init__(self, model_path="./legal-bert-risk-classifier"):
        """
        Initialize the contract risk classifier with your trained model
        """
        print("Loading trained model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()  # Set to evaluation mode
        
        # Labels mapping (based on your training)
        self.label_mapping = {0: 'non-risky', 1: 'risky'}
        print("Model loaded successfully!")
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract all text from PDF file
        """
        try:
            reader = PdfReader(pdf_path)
            all_text = ""
            
            print(f"Processing {len(reader.pages)} pages...")
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text:
                    all_text += text + "\n"
                print(f"Extracted text from page {page_num}")
            
            return all_text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return None
    
    def split_into_clauses(self, text):
        """
        Split contract text into individual clauses
        You can customize this based on your contract structure
        """
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split by common clause separators
        # Adjust these patterns based on your contract format
        clause_patterns = [
            r'\n\s*\d+\.\s+',  # Numbered clauses (1. 2. 3.)
            r'\n\s*\([a-z]\)\s+',  # Lettered subclauses (a) (b) (c))
            r'\n\s*\([0-9]+\)\s+',  # Numbered subclauses (1) (2) (3)
            r'\.\s*[A-Z][^.]*shall',  # Sentences with "shall"
            r'\.\s*[A-Z][^.]*agree',  # Sentences with "agree"
            r';\s*[A-Z]',  # Semi-colon separated clauses
        ]
        
        # Simple sentence-based splitting as fallback
        clauses = re.split(r'[.;]\s*(?=[A-Z])', text)
        
        # Clean and filter clauses
        cleaned_clauses = []
        for clause in clauses:
            clause = clause.strip()
            if len(clause) > 20 and len(clause) < 2000:  # Filter very short/long texts
                cleaned_clauses.append(clause)
        
        return cleaned_clauses
    
    def classify_clause(self, clause_text):
        """
        Classify a single clause using the trained model
        """
        # Tokenize the input
        inputs = self.tokenizer(
            clause_text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        return {
            'prediction': self.label_mapping[predicted_class],
            'confidence': round(confidence * 100, 2),
            'risk_score': predicted_class  # 0 = non-risky, 1 = risky
        }
    
    def process_pdf(self, pdf_path, output_csv=None):
        """
        Complete pipeline: PDF -> Text -> Clauses -> Risk Classification
        """
        print(f"Processing PDF: {pdf_path}")
        
        # Step 1: Extract text from PDF
        contract_text = self.extract_text_from_pdf(pdf_path)
        if not contract_text:
            return None
        
        # Step 2: Split into clauses
        print("Splitting text into clauses...")
        clauses = self.split_into_clauses(contract_text)
        print(f"Found {len(clauses)} clauses")
        
        # Step 3: Classify each clause
        print("Classifying clauses...")
        results = []
        
        for i, clause in enumerate(clauses, 1):
            classification = self.classify_clause(clause)
            
            result = {
                'clause_number': i,
                'clause_text': clause[:200] + "..." if len(clause) > 200 else clause,
                'full_clause_text': clause,
                'risk_classification': classification['prediction'],
                'confidence_percentage': classification['confidence'],
                'risk_score': classification['risk_score']
            }
            results.append(result)
            
            # Progress indicator
            if i % 10 == 0:
                print(f"Processed {i}/{len(clauses)} clauses")
        
        # Convert to DataFrame for easy handling
        df_results = pd.DataFrame(results)
        
        # Summary statistics
        risky_count = len(df_results[df_results['risk_classification'] == 'risky'])
        non_risky_count = len(df_results[df_results['risk_classification'] == 'non-risky'])
        
        print(f"\n{'='*50}")
        print(f"ANALYSIS COMPLETE!")
        print(f"Total Clauses: {len(clauses)}")
        print(f"Risky Clauses: {risky_count}")
        print(f"Non-Risky Clauses: {non_risky_count}")
        print(f"Risk Ratio: {risky_count/len(clauses)*100:.1f}%")
        print(f"{'='*50}\n")
        
        # Save to CSV if requested
        if output_csv:
            df_results.to_csv(output_csv, index=False)
            print(f"Results saved to: {output_csv}")
        
        return df_results

# Usage Example
def main():
    # Initialize the classifier with your trained model
    classifier = ContractRiskClassifier("./legal-bert-risk-classifier")
    
    # Process a PDF file (replace with your PDF path)
    pdf_file = "contract.pdf"  # Change this to your PDF file name
    
    if os.path.exists(pdf_file):
        # Process the PDF and get results
        results_df = classifier.process_pdf(
            pdf_path=pdf_file,
            output_csv="contract_risk_analysis.csv"
        )
        
        # Display high-risk clauses
        print("HIGH-RISK CLAUSES DETECTED:")
        print("-" * 50)
        risky_clauses = results_df[results_df['risk_classification'] == 'risky']
        
        for _, row in risky_clauses.head(5).iterrows():  # Show top 5 risky clauses
            print(f"Clause {row['clause_number']}: {row['confidence_percentage']}% confidence")
            print(f"Text: {row['clause_text']}")
            print("-" * 30)
    else:
        print(f"PDF file '{pdf_file}' not found. Please check the file path.")

if __name__ == "__main__":
    main()
