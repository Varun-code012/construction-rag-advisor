import pdfplumber
import re

def clean_text(text):
    if not text:
        return ""
    
    # Remove standalone page numbers (a number alone on a line)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Remove common BBMP header/footer patterns
    text = re.sub(r'BBMP Building Bye.Laws.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Bruhat Bengaluru Mahanagara Palike.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Building Bye-Laws 2003.*?\n', '', text, flags=re.IGNORECASE)
    
    # Fix broken sentences — join lines that don't end with punctuation
    lines = text.split('\n')
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            fixed_lines.append('')
            i += 1
            continue
        # If line doesn't end with punctuation, join with next line
        while (i + 1 < len(lines) and
               line and
               not line[-1] in '.,:;)' and
               lines[i+1].strip() and
               not lines[i+1].strip()[0].isupper()):
            i += 1
            line = line + ' ' + lines[i].strip()
        fixed_lines.append(line)
        i += 1
    
    text = '\n'.join(fixed_lines)
    
    # Remove multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

# Extract and clean text page by page
print("Opening BBMP PDF...")
all_pages = []

with pdfplumber.open("bbmp.pdf") as pdf:
    print(f"Total pages: {len(pdf.pages)}")
    for i, page in enumerate(pdf.pages):
        raw_text = page.extract_text()
        cleaned = clean_text(raw_text)
        if cleaned:
            all_pages.append({
                "page": i + 1,
                "text": cleaned
            })
        print(f"Cleaned page {i+1}/{len(pdf.pages)}")

# Save cleaned text to a file
with open("bbmp_cleaned.txt", "w", encoding="utf-8") as f:
    for page_data in all_pages:
        f.write(f"\n--- PAGE {page_data['page']} ---\n")
        f.write(page_data["text"])
        f.write("\n")

print(f"\nDone! Cleaned {len(all_pages)} pages.")
print("Saved to bbmp_cleaned.txt")
print("\nSample from page 1:")
print(all_pages[0]["text"][:500])