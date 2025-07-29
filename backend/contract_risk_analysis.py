import pandas as pd

# Load the analysis CSV file
df = pd.read_csv('contract_risk_analysis.csv')

# Basic summary statistics
total_clauses = len(df)
risky_clauses = df[df['risk_classification'] == 'risky']
non_risky_clauses = df[df['risk_classification'] == 'non-risky']

num_risky = len(risky_clauses)
num_non_risky = len(non_risky_clauses)
risk_ratio = (num_risky / total_clauses) * 100

avg_conf_risky = risky_clauses['confidence_percentage'].mean()
avg_conf_non_risky = non_risky_clauses['confidence_percentage'].mean()

# Prepare the report text
report_lines = []

report_lines.append("CONTRACT CLAUSE RISK ANALYSIS REPORT\n")
report_lines.append("="*50 + "\n")

report_lines.append(f"Total Clauses Analyzed: {total_clauses}")
report_lines.append(f"Number of Risky Clauses: {num_risky} ({risk_ratio:.2f}%)")
report_lines.append(f"Number of Non-Risky Clauses: {num_non_risky} ({100 - risk_ratio:.2f}%)\n")

report_lines.append(f"Average Confidence (Risky Clauses): {avg_conf_risky:.2f}%")
report_lines.append(f"Average Confidence (Non-Risky Clauses): {avg_conf_non_risky:.2f}%\n")

# List top 10 risky clauses by confidence
report_lines.append("TOP 10 HIGH CONFIDENCE RISKY CLAUSES:\n")

top_risky = risky_clauses.sort_values(by='confidence_percentage', ascending=False).head(10)

for idx, row in top_risky.iterrows():
    report_lines.append(f"Clause #{row['clause_number']}: Confidence: {row['confidence_percentage']}%")
    report_lines.append(f"Text: {row['clause_text']}")
    report_lines.append("-"*40)

# Optional: List some non-risky clauses (top confident)
report_lines.append("\nSAMPLE NON-RISKY CLAUSES:\n")

top_non_risky = non_risky_clauses.sort_values(by='confidence_percentage', ascending=False).head(5)

for idx, row in top_non_risky.iterrows():
    report_lines.append(f"Clause #{row['clause_number']}: Confidence: {row['confidence_percentage']}%")
    report_lines.append(f"Text: {row['clause_text']}")
    report_lines.append("-"*40)

# Write the report to a text file
report_filename = "contract_risk_analysis_report.txt"
with open(report_filename, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"Report generated and saved as '{report_filename}'")
