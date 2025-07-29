import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import json
import os
from datetime import datetime
import re

class EnhancedContractSummarizer:
    def __init__(self, summarization_model="google/flan-t5-base"):  # Using base model for faster loading
        """
        Enhanced contract summarizer with detailed layman explanations
        """
        print(f"Loading summarization model: {summarization_model}")
        
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(summarization_model)
            self.model = T5ForConditionalGeneration.from_pretrained(summarization_model)
            self.model.eval()
            print("✅ Summarization model loaded successfully!")
            self.model_loaded = True
        except Exception as e:
            print(f"❌ Error loading summarization model: {e}")
            print("📝 Falling back to enhanced rule-based summaries")
            self.model_loaded = False
    
    def load_classification_results(self, csv_path):
        """Load results from Legal-BERT classifier"""
        try:
            df = pd.read_csv(csv_path)
            print(f"✅ Loaded classification results: {len(df)} clauses")
            return df
        except Exception as e:
            print(f"❌ Error loading classification results: {e}")
            return None
    
    def extract_contract_content(self, pdf_path):
        """Extract and analyze contract content"""
        if not os.path.exists(pdf_path):
            return None
            
        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n"
            
            # Analyze contract type and parties
            contract_info = self.analyze_contract_basics(full_text)
            return {
                'full_text': full_text,
                'contract_info': contract_info
            }
        except Exception as e:
            print(f"Warning: Could not read PDF: {e}")
            return None
    
    def analyze_contract_basics(self, contract_text):
        """Extract basic contract information"""
        text_lower = contract_text.lower()
        
        # Determine contract type
        contract_type = "Unknown Contract"
        if 'service agreement' in text_lower:
            contract_type = "Service Agreement"
        elif 'employment' in text_lower:
            contract_type = "Employment Contract"
        elif 'lease' in text_lower or 'rental' in text_lower:
            contract_type = "Lease/Rental Agreement"
        elif 'purchase' in text_lower or 'sale' in text_lower:
            contract_type = "Purchase/Sale Agreement"
        elif 'license' in text_lower:
            contract_type = "License Agreement"
        elif 'partnership' in text_lower:
            contract_type = "Partnership Agreement"
        
        # Extract parties (simplified)
        parties = []
        lines = contract_text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            if 'party a' in line.lower() or 'party b' in line.lower():
                parties.append(line.strip())
            elif 'between' in line.lower() and 'and' in line.lower():
                parties.append(line.strip())
        
        return {
            'contract_type': contract_type,
            'parties': parties[:2] if parties else ["Party information not clearly identified"]
        }
    
    def calculate_detailed_risk_metrics(self, classification_df):
        """Calculate comprehensive risk metrics"""
        if classification_df is None or len(classification_df) == 0:
            return None
        
        total_clauses = len(classification_df)
        risky_clauses = len(classification_df[classification_df['risk_classification'] == 'risky'])
        risk_percentage = (risky_clauses / total_clauses) * 100 if total_clauses > 0 else 0
        
        # Risk level determination
        if risk_percentage >= 60:
            overall_risk = "CRITICAL"
            risk_description = "extremely dangerous"
        elif risk_percentage >= 40:
            overall_risk = "HIGH"
            risk_description = "quite risky"
        elif risk_percentage >= 25:
            overall_risk = "MEDIUM"
            risk_description = "moderately risky"
        elif risk_percentage >= 10:
            overall_risk = "LOW"
            risk_description = "relatively safe with some concerns"
        else:
            overall_risk = "MINIMAL"
            risk_description = "very safe with minimal concerns"
        
        return {
            'total_clauses': total_clauses,
            'risky_clauses': risky_clauses,
            'non_risky_clauses': total_clauses - risky_clauses,
            'risk_percentage': round(risk_percentage, 2),
            'overall_risk_level': overall_risk,
            'risk_description': risk_description,
            'is_contract_risky': risk_percentage >= 25
        }
    
    def analyze_specific_risks(self, classification_df):
        """Analyze specific risks in detail"""
        if classification_df is None:
            return []
        
        risky_clauses = classification_df[classification_df['risk_classification'] == 'risky']
        detailed_risks = []
        
        for _, clause in risky_clauses.iterrows():
            clause_text = clause['full_clause_text'].lower()
            
            # Detailed risk analysis
            risk_details = {
                'clause_number': clause['clause_number'],
                'clause_text': clause['clause_text'],
                'confidence': clause['confidence_percentage'],
                'risk_type': '',
                'what_it_means': '',
                'why_its_risky': '',
                'potential_impact': ''
            }
            
            # Analyze specific risk types
            if 'waive' in clause_text and ('claims' in clause_text or 'liability' in clause_text):
                risk_details.update({
                    'risk_type': 'Liability Waiver',
                    'what_it_means': 'You are giving up your right to hold the other party responsible if something goes wrong.',
                    'why_its_risky': 'Even if they cause damage through negligence or mistakes, you cannot seek compensation.',
                    'potential_impact': 'You could lose money or suffer damages with no legal recourse.'
                })
            
            elif 'terminate' in clause_text and 'without cause' in clause_text:
                risk_details.update({
                    'risk_type': 'Unfair Termination',
                    'what_it_means': 'The other party can end this contract at any time without giving you a good reason.',
                    'why_its_risky': 'You have no job security or contract security - they can terminate without justification.',
                    'potential_impact': 'Sudden loss of income or business relationship without warning.'
                })
            
            elif 'no refund' in clause_text:
                risk_details.update({
                    'risk_type': 'No Refund Policy',
                    'what_it_means': 'Once you pay, you cannot get your money back under any circumstances.',
                    'why_its_risky': 'Even if services are not delivered or are unsatisfactory, you lose your payment.',
                    'potential_impact': 'Financial loss with no recourse for poor performance.'
                })
            
            elif 'non-compete' in clause_text:
                risk_details.update({
                    'risk_type': 'Non-Compete Restriction',  
                    'what_it_means': 'You are prohibited from working with competitors or starting a competing business.',
                    'why_its_risky': 'Limits your future career opportunities and earning potential.',
                    'potential_impact': 'Restricted job market and potential legal action if violated.'
                })
            
            else:
                risk_details.update({
                    'risk_type': 'General Contract Risk',
                    'what_it_means': 'This clause contains terms that could disadvantage you.',
                    'why_its_risky': 'The wording may create obligations or restrictions that are unfavorable.',
                    'potential_impact': 'Potential legal or financial consequences depending on the specific terms.'
                })
            
            detailed_risks.append(risk_details)
        
        return detailed_risks
    
    def generate_comprehensive_layman_summary(self, contract_info, risk_metrics, detailed_risks):
        """Generate comprehensive layman-friendly summary"""
        
        summary_parts = []
        
        # Contract Overview
        if contract_info:
            summary_parts.append(f"📋 **CONTRACT TYPE**: This is a {contract_info['contract_info']['contract_type']}.")
            if contract_info['contract_info']['parties']:
                summary_parts.append(f"👥 **PARTIES**: {contract_info['contract_info']['parties'][0]}")
        
        # Overall Risk Assessment
        risk_emoji = "🚨" if risk_metrics['is_contract_risky'] else "✅"
        summary_parts.append(f"\n{risk_emoji} **OVERALL ASSESSMENT**: This contract is {risk_metrics['risk_description'].upper()}.")
        
        summary_parts.append(f"📊 **RISK BREAKDOWN**: Out of {risk_metrics['total_clauses']} total clauses, {risk_metrics['risky_clauses']} are risky ({risk_metrics['risk_percentage']}%) and {risk_metrics['non_risky_clauses']} are standard.")
        
        # Safety Level Explanation
        if risk_metrics['risk_percentage'] >= 40:
            summary_parts.append("⚠️ **SAFETY LEVEL**: This contract has many dangerous terms that could seriously harm your interests. We strongly recommend getting legal help before signing.")
        elif risk_metrics['risk_percentage'] >= 25:
            summary_parts.append("🔶 **SAFETY LEVEL**: This contract has some concerning terms that need careful consideration. Legal review is recommended.")
        elif risk_metrics['risk_percentage'] >= 10:
            summary_parts.append("🔸 **SAFETY LEVEL**: This contract is mostly safe but has a few terms to watch out for. Read carefully before signing.")
        else:
            summary_parts.append("✅ **SAFETY LEVEL**: This contract appears to be safe with standard terms. Basic review should be sufficient.")
        
        # Detailed Risk Explanations
        if detailed_risks:
            summary_parts.append(f"\n🚨 **SPECIFIC CONCERNS EXPLAINED**:")
            for i, risk in enumerate(detailed_risks, 1):
                summary_parts.extend([
                    f"\n**Risk #{i}: {risk['risk_type']}** (Clause {risk['clause_number']})",
                    f"🔍 **What it means**: {risk['what_it_means']}",
                    f"⚠️ **Why it's risky**: {risk['why_its_risky']}",
                    f"💥 **Potential impact**: {risk['potential_impact']}"
                ])
        
        # Recommendations
        summary_parts.append(f"\n💡 **WHAT YOU SHOULD DO**:")
        if risk_metrics['is_contract_risky']:
            summary_parts.extend([
                "1. 🏛️ **Get Legal Help**: Consult with a lawyer before signing",
                "2. 💬 **Negotiate**: Try to modify or remove the risky clauses",
                "3. 🛡️ **Protect Yourself**: Consider insurance or other protections",
                "4. 📝 **Document Everything**: Keep records of all discussions"
            ])
        else:
            summary_parts.extend([
                "1. 📖 **Read Carefully**: Make sure you understand all terms",
                "2. ❓ **Ask Questions**: Clarify anything unclear with the other party",
                "3. 📋 **Keep Records**: Save copies of all signed documents"
            ])
        
        return "\n".join(summary_parts)
    
    def create_detailed_report(self, classification_csv, contract_pdf_path=None, output_file=None):
        """Create comprehensive contract report with detailed layman summary"""
        print(f"🔍 Creating detailed contract analysis...")
        
        # Load data
        classification_df = self.load_classification_results(classification_csv)
        if classification_df is None:
            return None
        
        contract_info = None
        if contract_pdf_path:
            contract_info = self.extract_contract_content(contract_pdf_path)
        
        # Calculate metrics
        risk_metrics = self.calculate_detailed_risk_metrics(classification_df)
        detailed_risks = self.analyze_specific_risks(classification_df)
        
        # Generate comprehensive summary
        layman_summary = self.generate_comprehensive_layman_summary(contract_info, risk_metrics, detailed_risks)
        
        # Create report
        report = {
            'analysis_metadata': {
                'contract_file': contract_pdf_path or 'N/A',
                'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model_used': 'Legal-BERT + Enhanced Rule-Based Analysis'
            },
            'contract_overview': contract_info['contract_info'] if contract_info else None,
            'risk_assessment': risk_metrics,
            'detailed_layman_summary': layman_summary,
            'specific_risk_analysis': detailed_risks,
            'clause_breakdown': classification_df.to_dict('records')
        }
        
        # Save report
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"📝 Detailed report saved to: {output_file}")
        
        # Print summary
        self.print_detailed_summary(report)
        
        return report
    
    def print_detailed_summary(self, report):
        """Print comprehensive summary to console"""
        print("\n" + "="*80)
        print("COMPREHENSIVE CONTRACT ANALYSIS - LAYMAN'S SUMMARY")
        print("="*80)
        
        print(report['detailed_layman_summary'])
        
        print("\n" + "="*80)

# Usage
def main():
    # Initialize enhanced summarizer
    summarizer = EnhancedContractSummarizer()
    
    # Create detailed report
    report = summarizer.create_detailed_report(
        classification_csv="contract_risk_analysis.csv",
        contract_pdf_path="contract.pdf",
        output_file="detailed_contract_analysis.json"
    )

if __name__ == "__main__":
    main()
