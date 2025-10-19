"""Preprocess raw SEC documents"""
import os
import json
import glob
import re
from typing import Any, List, Dict, Optional
import unicodedata
from loguru import logger

def normalize_quotes(text: str) -> str:
    """Normalize all quote-like characters"""
    single_quotes = [
        '\u2018', '\u2019',  # Curly quotes
        '\u02bc', '\u2032',  # Apostrophe variants
        '\u00b4', '`',       # Accent, backtick
    ]
    for quote in single_quotes:
        text = text.replace(quote, "'")
    
    double_quotes = [
        '\u201c', '\u201d',  # Curly quotes
        '\u2033',            # Double prime
        # '"', '"'             # Literal curly quotes
    ]
    for quote in double_quotes:
        text = text.replace(quote, '"')
    
    return text

def clean_unicode(text: str) -> str:
    """Clean and normalize Unicode characters"""
    # Normalize to NFKD (compatibility decomposition)
    text = unicodedata.normalize('NFKD', text)

    text = normalize_quotes(text)
    
    # Replace common problematic characters
    replacements = {
        # '\u00a0': ' ',
        # '\u2018': "'",
        # '\u2019': "'",
        # '\u201c': '"',
        # '\u201d': '"',
        '\u2013': '-',
        '\u2014': '--',
        '\u2026': '...',
        '\xa0': ' ',
        '\u200b': '',
        '\ufeff': '',
        # '\u02bc': "'",
        # '\u2032': "'",
        # '\u00b4': "'",
        # '`': "'",
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove control characters except newlines/tabs
    text = ''.join(
        char for char in text 
        if unicodedata.category(char)[0] != 'C' or char in '\n\t'
    )
    
    # # Encode to ASCII with 'ignore' for remaining problematic chars
    # # then decode back (removes characters that can't be represented)
    # try:
    #     text = text.encode('ascii', 'ignore').decode('ascii')
    # except:
    #     pass
    
    return text

def clean_navigation_text(content: str) -> str:
    """Remove SEC.gov navigation and boilerplate"""
    
    nav_patterns = [
        r"SEC\.gov \|.*?(?=\n)",
        r"Skip to .*?(?=\n)",
        r"An official website.*?(?=\n)",
        r"Here's how you know.*?(?=\n)",
        r"Official websites use \.gov.*?(?=\n)",
        r"Secure \.gov websites.*?(?=\n)",
        r"SEC homepage.*?(?=\n)",
        r"Menu\s*Close",
        r"Search SEC\.gov.*?(?=\n)",
        r"Submit Filings.*?(?=\n)",
        r"Data & Research.*?(?=\n)",
        r"Rules, Enforcement.*?(?=\n)",
        r"More in this Section.*?(?=\n)",
        r"Return to top.*",
        r"Stay connected\..*",
        r"Sign Up\s*X\s*Facebook.*",
        r"Plain Writing.*Privacy.*Site Map.*"
    ]

    cleaned = content
    for pattern in nav_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    return cleaned

def extract_main_content(content: str) -> str:
    """Extract the main case content"""
    patterns = [
        r"(SEC Obtains.*?)(?:Resources|Return to top|$)",
        r"(On \w+ \d+, \d{4}.*?)(?:Resources|Return to top|$)",
        r"(The SEC's complaint.*?)(?:Resources|Return to top|$)"
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)
    
    return content

def normalize_whitespace(text: str) -> str:
    """Normalize whitespace"""
    # Replace multiple newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text

def extract_amounts(content: str) -> List[str]:
    """Extract dollar amounts"""
    # Pattern for amounts with million/billion
    pattern = r'\$\d+(?:[.,]\d+)*(?:\s*(?:million|billion|thousand))?'
    amounts = re.findall(pattern, content, re.IGNORECASE)
    
    # Remove duplicates, keep order
    return list(dict.fromkeys(amounts))[:10]

def categorize_crime(content: str) -> str:
    """Categorize the type of financial crime"""
    content_lower = content.lower()
    crime_types = []
    # Priority order (specific to general)
    # if 'ponzi' in content_lower:
    #     return 'Ponzi Scheme'
    # elif 'insider trading' in content_lower:
    #     return 'Insider Trading'
    # elif 'securities fraud' in content_lower:
    #     return 'Securities Fraud'
    # elif 'wire fraud' in content_lower:
    #     return 'Wire Fraud'
    # elif 'money laundering' in content_lower:
    #     return 'Money Laundering'
    # elif 'embezzlement' in content_lower:
    #     return 'Embezzlement'
    # elif 'market manipulation' in content_lower:
    #     return 'Market Manipulation'
    # elif 'fraud' in content_lower:
    #     return 'Fraud (General)'
    # elif 'economic crime' in content_lower or 'economic offence' in content_lower:
    #     return 'Economic Crime'
    # else:
    #     return 'Financial Crime (Other)'

    if 'ponzi' in content_lower:
        crime_types.append('Ponzi Scheme')
    
    if 'insider trading' in content_lower:
        crime_types.append('Insider Trading')
    
    if 'securities fraud' in content_lower:
        crime_types.append('Securities Fraud')
    
    if 'wire fraud' in content_lower:
        crime_types.append('Wire Fraud')
    
    if 'money laundering' in content_lower:
        crime_types.append('Money Laundering')
    
    if 'embezzlement' in content_lower:
        crime_types.append('Embezzlement')
    
    if 'market manipulation' in content_lower:
        crime_types.append('Market Manipulation')
    
    if 'bribery' in content_lower or 'foreign corrupt practices' in content_lower:
        crime_types.append('Bribery/FCPA')
    
    # Catch-all if no specific type found
    if not crime_types:
        if 'fraud' in content_lower:
            crime_types.append('Fraud (General)')
        else:
            crime_types.append('Financial Crime (Other)')
    
    return crime_types


def extract_people_simple(content: str) -> List[str]:
    """Extract person names from content"""
    # More strict pattern: First Middle? Last (2-3 words only)
    # Must have at least 2 capital letters
    name_pattern = r'\b([A-Z][a-z]{2,})\s+(?:([A-Z][a-z]?\.?\s+))?([A-Z][a-z]{2,})\b'
    matches = re.findall(name_pattern, content)
    
    # Build names from matches
    potential_names = []
    for match in matches:
        first, middle, last = match
        if middle.strip():
            name = f"{first} {middle.strip()} {last}"
        else:
            name = f"{first} {last}"
        potential_names.append(name)
    
    # Extensive exclude list
    exclude_words = {
        'securities', 'exchange', 'commission', 'worth', 'united', 'states',
        'district', 'court', 'civil', 'penalty', 'final', 'judgment',
        'litigation', 'release', 'act', 'scheme', 'september', 'december',
        'ponzi', 'cattle', 'federal', 'texas', 'against', 'company',
        'founders', 'obtains', 'complaint'
    }
    
    # Filter
    names = []
    for name in potential_names:
        name_lower = name.lower()
        # Skip if any word in name is in exclude list
        words = name_lower.split()
        if not any(word in exclude_words for word in words):
            names.append(name)
    
    return list(dict.fromkeys(names))[:10]

def preprocess_document(doc: Dict) -> Dict:
    """Preprocess a single SEC document"""
    content = doc.get('content', '')
    
    # Step 1: Clean Unicode
    content = clean_unicode(content)

    # Step 2: Clean navigation
    content = clean_navigation_text(content)
    
    # Step 3: Extract main content
    content = extract_main_content(content)
    
    # Step 4: Normalize whitespace
    content = normalize_whitespace(content)
    
    # Step 5: Extract metadata
    crime_type = categorize_crime(content)
    amounts = extract_amounts(content)
    people = extract_people_simple(content)

    processed = {
        'lr_no': doc.get('lr_no', '').replace('Release No.', '').strip(),
        'title': doc.get('title', ''),
        'date': doc.get('date', ''),
        'url': doc.get('url', ''),
        'content': content,
        'content_length': len(content),
        'crime_type': crime_type,
        'amounts': amounts,
        'people_mentioned': people,
        'see_also': doc.get('see_also', []),
        'source': 'SEC'
    }

    return processed

def preprocess_documents(documents: List[Dict]) -> List[Dict]:
    """Preprocess multiple documents"""
    processed = []
    
    for i, doc in enumerate(documents):
        if doc.get('success') is False:
            logger.warning(f"Skipping document {i} - no content")
            continue
        
        try:
            processed_doc = preprocess_document(doc)
            processed.append(processed_doc)
        except Exception as e:
            logger.error(f"Error processing document {i}: {e}")
            continue
    
    logger.info(f"Processed {len(processed)}/{len(documents)} documents")
    return processed

if __name__ == "__main__":

    os.makedirs("data/processed", exist_ok=True)

    raw_filepath = "data/raw/sec_releases_batch*.json"
    processed_filepath = "data/processed/sec_releases_batch_{i}_clean.json"

    batch_files = sorted(glob.glob(raw_filepath))

    for i, batch_file in enumerate(batch_files):
        logger.info(f"Processing batch file {batch_file}")
        with open(batch_file, 'r') as f:
            data = json.load(f)
            raw_docs = data["releases"]
        
        processed_docs = preprocess_documents(raw_docs)
        
        
        with open(processed_filepath.format(i=i+1), "w") as f:
            json.dump({"releases": processed_docs}, f, indent=2)
    
    # print("\n=== Sample Processed Document ===")
    # print(f"LR No: {processed_docs[0]['lr_no']}")
    # print(f"Crime Type: {processed_docs[0]['crime_type']}")
    # print(f"Amounts: {processed_docs[0]['amounts'][:3]}")
    # print(f"People: {processed_docs[0]['people_mentioned'][:5]}")
    # print(f"Content length: {processed_docs[0]['content_length']} chars")
    # print(f"\nFirst 500 chars:\n{processed_docs[0]['content'][:500]}")
