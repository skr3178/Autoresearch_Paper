#!/usr/bin/env python3
"""
Phase 0: Paper Extraction
Extracts figures, equations, and metrics from the CarPlanner PDF
"""
import fitz  # PyMuPDF
import os
import re
from pathlib import Path

# Open the PDF
pdf_path = "paper/CarPlanner.pdf"
doc = fitz.open(pdf_path)

# Create output directories
os.makedirs("paper/images", exist_ok=True)

print(f"PDF has {len(doc)} pages")

# Extract all figures and their annotations
figure_annotations = []

for page_num in range(len(doc)):
    page = doc[page_num]
    text = page.get_text()
    
    # Look for figure captions
    figure_pattern = r'Figure\s+(\d+)[:\.]?\s*([^\n]+)'
    matches = re.finditer(figure_pattern, text, re.IGNORECASE)
    
    for match in matches:
        fig_num = match.group(1)
        caption = match.group(2).strip()
        print(f"\nPage {page_num + 1}, Figure {fig_num}:")
        print(f"  Caption: {caption}")
        figure_annotations.append({
            'page': page_num + 1,
            'figure_num': fig_num,
            'caption': caption,
            'text_context': text[max(0, match.start()-500):match.end()+500]
        })

# Extract tables
print("\n\n=== EXTRACTING TABLES ===\n")
table_pattern = r'Table\s+(\d+)[:\.]?\s*([^\n]+)'
for page_num in range(len(doc)):
    page = doc[page_num]
    text = page.get_text()
    matches = re.finditer(table_pattern, text, re.IGNORECASE)
    
    for match in matches:
        table_num = match.group(1)
        caption = match.group(2).strip()
        print(f"\nPage {page_num + 1}, Table {table_num}:")
        print(f"  Caption: {caption}")

# Extract equations
print("\n\n=== EXTRACTING EQUATIONS ===\n")
# Look for equation-like patterns
for page_num in range(len(doc)):
    page = doc[page_num]
    text = page.get_text()
    
    # Look for equation numbers
    eq_pattern = r'\((\d+)\)\s*$'
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if re.search(eq_pattern, line.strip()):
            print(f"\nPage {page_num + 1}, Line {i}: {line}")

doc.close()
print("\n\nExtraction complete!")
