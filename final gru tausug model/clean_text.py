#!/usr/bin/env python3
"""
Advanced Text Cleaning Script
Process: Comprehensive Text Cleaning for Tausug Language Data

This script performs multiple cleaning operations on extracted text:
1. Removes OCR artifacts (like 'cid' sequences)
2. Filters out non-Tausug content (Arabic text, special characters)
3. Preserves sentence structure
4. Cleans encoding issues
5. Normalizes whitespace
6. Removes unwanted symbols and formatting artifacts
"""

import re
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_ocr_artifacts(text: str) -> str:
    """Remove OCR artifacts and scanning errors"""
    # Remove 'cid' sequences and similar OCR artifacts
    text = re.sub(r'\bcid+\b', '', text)
    text = re.sub(r'cid', '', text)
    
    # Remove common OCR error patterns
    text = re.sub(r'\b[a-z]{1,3}\d+[a-z]?\b', '', text)  # Short letter-number combinations
    text = re.sub(r'\d+[a-z]+\d*', '', text)  # Number-letter-number patterns
    
    return text

def remove_numbers_and_references(text: str) -> str:
    """Remove all numbers, digits, and reference patterns"""
    # Remove standalone numbers
    text = re.sub(r'\b\d+\b', '', text)
    
    # Remove digits within words
    text = re.sub(r'\d', '', text)
    
    # Remove page references and citations
    text = re.sub(r'page\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'pg\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'vol\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'chapter\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(\s*\d+\s*\)', '', text)  # (123) patterns
    text = re.sub(r'\[\s*\d+\s*\]', '', text)  # [123] patterns
    
    # Remove verse/ayat references
    text = re.sub(r'ayat\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'surah\s*[a-z-]+\s*\d+', '', text, flags=re.IGNORECASE)
    
    return text

def remove_non_tausug_languages(text: str) -> str:
    """Remove Arabic, English, and other non-Tausug language content"""
    # Remove Arabic script characters (Unicode range)
    text = re.sub(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', '', text)
    
    # Remove Greek letters often found in academic texts
    text = re.sub(r'[αβγδεζηθικλμνξοπρστυφχψω]', '', text)
    text = re.sub(r'[ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]', '', text)
    
    # Remove common English/Arabic religious terms that appear in religious texts
    english_arabic_terms = [
        r'\ballahu?\b', r'\ballah\b', r'\bmuhammad\b', r'\bislam\b', r'\bmuslim\b',
        r'\bqur\'?an\b', r'\bhadith\b', r'\bsunnah\b', r'\bshia\b', r'\bshi\'ah\b',
        r'\bsalafi\b', r'\bimam\b', r'\brafidah\b', r'\bahlus\b', r'\bal-\w+\b',
        r'\bibn\b', r'\babdul?\b', r'\babdu\b', r'\brahman\b', r'\braheem\b',
        r'\bbismillah\b', r'\bassalam\b', r'\balaykum\b', r'\bwarahmatullahi\b',
        r'\bwabarakatuhu\b', r'\bsubhanallah\b', r'\balhamdulillah\b', r'\ballahu\s+akbar\b',
        r'\bla\s+ilaha\s+illa\s+allah\b', r'\bmasha\'?allah\b', r'\binsha\'?allah\b'
    ]
    
    for term in english_arabic_terms:
        text = re.sub(term, '', text, flags=re.IGNORECASE)
    
    # Remove excessive punctuation and symbols
    text = re.sub(r'[ــــــــــــــــــــــــــ]+', '', text)  # Arabic tatweel
    text = re.sub(r'[\'\'\"\"]+', "'", text)  # Normalize quotes
    text = re.sub(r'[‚„""'']+', "'", text)  # More quote variations
    
    # Remove academic/reference symbols
    text = re.sub(r'[◦•▪▫■□▲△▼▽◆◇○●★☆]+', '', text)
    text = re.sub(r'[←→↑↓↔↕]+', '', text)
    text = re.sub(r'[§¶†‡•‰′″‴‹›«»]+', '', text)
    
    # Remove mathematical and technical symbols
    text = re.sub(r'[∀∂∃∅∇∈∉∋∌∏∑−∓∔∕∗∘∙√∛∜∝∞∟∠∡∢∣∤∥∦∧∨∩∪∫∬∭∮∯∰∱∲∳]+', '', text)
    
    return text

def clean_special_characters(text: str) -> str:
    """Remove special characters while preserving Tausug punctuation"""
    # Remove HTML/XML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove currency symbols
    text = re.sub(r'[¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿×÷]', '', text)
    
    # Remove excessive special characters but keep basic punctuation
    text = re.sub(r'[!@#$%^&*()_+=\[\]{}\\|;:"`~<>?/]+', ' ', text)
    
    # Keep only basic punctuation: . , ! ? ' - 
    # Remove other punctuation marks
    text = re.sub(r'[""''„‚‹›«»‰‱′″‴‵‶‷‸‹›⁄⁰⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎]', '', text)
    
    # Remove excessive dashes and underscores but keep single hyphens
    text = re.sub(r'[-_]{2,}', '', text)
    text = re.sub(r'[=]{2,}', '', text)
    
    # Remove code-like patterns
    text = re.sub(r'\b[A-Za-z]+\d+[A-Za-z]*\b', '', text)  # Mixed alphanumeric codes
    text = re.sub(r'\b[a-z]{1,2}\d+\b', '', text)  # Short code patterns
    
    return text

def remove_document_artifacts(text: str) -> str:
    """Remove document formatting and reference artifacts"""
    # Remove URL patterns
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email patterns
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # Remove academic references and citations
    text = re.sub(r'\([^)]*\d+[^)]*\)', '', text)  # Citations with numbers in parentheses
    text = re.sub(r'\[[^\]]*\d+[^\]]*\]', '', text)  # Citations with numbers in brackets
    
    # Remove common document headers/footers
    text = re.sub(r'copyright\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'all\s+rights\s+reserved', '', text, flags=re.IGNORECASE)
    text = re.sub(r'printed\s+in', '', text, flags=re.IGNORECASE)
    text = re.sub(r'published\s+by', '', text, flags=re.IGNORECASE)
    
    # Remove ISBN and similar patterns
    text = re.sub(r'isbn\s*:?\s*[\d-]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'issn\s*:?\s*[\d-]+', '', text, flags=re.IGNORECASE)
    
    return text

def normalize_whitespace(text: str) -> str:
    """Normalize whitespace and preserve sentence structure"""
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Replace multiple newlines with single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Remove spaces at the beginning and end of lines
    text = re.sub(r'^ +', '', text, flags=re.MULTILINE)
    text = re.sub(r' +$', '', text, flags=re.MULTILINE)
    
    # Remove empty lines
    text = re.sub(r'\n\s*\n', '\n', text)
    
    return text

def filter_tausug_content(text: str) -> str:
    """Filter to keep only Tausug-like content with stricter filtering"""
    lines = text.split('\n')
    filtered_lines = []
    
    # Enhanced Tausug word indicators
    tausug_common_words = [
        # Particles and connectors
        'in', 'sin', 'ha', 'pa', 'na', 'ku', 'mu', 'niya', 'namu', 'niyu', 'nila',
        'iban', 'unu', 'biya', 'dayng', 'kanu', 'bang', 'way', 'awn', 'manga',
        # Common Tausug words
        'sila', 'kita', 'kami', 'kamu', 'tau', 'manusiya', 'anak', 'babai', 'usug',
        'agama', 'tuhan', 'allah', 'hinang', 'pasal', 'tungud', 'sabab', 'puas',
        'mahuli', 'nakauna', 'bihaun', 'adlaw', 'waktu', 'masa', 'tahun', 
        'dunya', 'ginhawa', 'kamatay', 'kabuhi', 'surga', 'narka',
        # Verbs and actions
        'mag', 'nag', 'pag', 'hi', 'kan', 'kiyaridaan', 'mahinang', 'nahinang',
        'miyagad', 'timindug', 'namung', 'laung', 'bayta', 'sabunnal', 'tuud'
    ]
    
    # Words that indicate non-Tausug content
    non_tausug_indicators = [
        'the', 'and', 'or', 'but', 'not', 'for', 'with', 'from', 'this', 'that',
        'have', 'has', 'had', 'will', 'would', 'could', 'should', 'page', 'chapter',
        'vol', 'volume', 'edition', 'press', 'university', 'published', 'copyright',
        'arabic', 'english', 'translation', 'transliteration', 'www', 'http', 'com'
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip very short lines (likely artifacts)
        if len(line) < 15:
            continue
        
        # Skip lines with excessive special characters
        special_char_ratio = sum(1 for c in line if not (c.isalnum() or c.isspace() or c in ".,!?'-")) / len(line)
        if special_char_ratio > 0.3:
            continue
            
        # Skip lines that are mostly numbers or special characters
        if re.match(r'^[\d\s\W]+$', line):
            continue
        
        # Skip lines with non-Tausug indicators
        words_lower = [word.lower().strip('.,!?"-') for word in line.split()]
        if any(indicator in words_lower for indicator in non_tausug_indicators):
            continue
            
        # Check for Tausug content
        tausug_word_count = 0
        for word in words_lower:
            # Direct matches
            if word in tausug_common_words:
                tausug_word_count += 1
            # Partial matches for compound words
            elif any(indicator in word for indicator in tausug_common_words[:20]):  # Use core particles
                tausug_word_count += 0.5
        
        # Require higher threshold for Tausug content
        if len(words_lower) > 0 and (tausug_word_count / len(words_lower)) > 0.15:
            # Additional check: must contain at least one strong Tausug indicator
            if any(word in tausug_common_words[:20] for word in words_lower):
                filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

def clean_text_comprehensive(text: str) -> str:
    """Apply all cleaning operations in sequence"""
    logger.info("Starting comprehensive text cleaning...")
    
    # Step 1: Remove OCR artifacts
    logger.info("Removing OCR artifacts...")
    text = clean_ocr_artifacts(text)
    
    # Step 2: Remove numbers, digits, and references
    logger.info("Removing numbers, digits, and references...")
    text = remove_numbers_and_references(text)
    
    # Step 3: Remove non-Tausug languages (Arabic, English, etc.)
    logger.info("Removing non-Tausug languages...")
    text = remove_non_tausug_languages(text)
    
    # Step 4: Clean special characters
    logger.info("Cleaning special characters...")
    text = clean_special_characters(text)
    
    # Step 5: Remove document artifacts
    logger.info("Removing document artifacts...")
    text = remove_document_artifacts(text)
    
    # Step 6: Normalize whitespace
    logger.info("Normalizing whitespace...")
    text = normalize_whitespace(text)
    
    # Step 7: Filter Tausug content (final filter)
    logger.info("Filtering Tausug content...")
    text = filter_tausug_content(text)
    
    # Final cleanup
    text = text.strip()
    
    logger.info("Text cleaning completed!")
    return text

def process_file(input_file: str, output_file: str = None) -> None:
    """Process a text file with comprehensive cleaning"""
    if output_file is None:
        output_file = input_file.replace('.txt', '_cleaned.txt')
    
    try:
        logger.info(f"Reading file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logger.info(f"Original text length: {len(content)} characters")
        
        cleaned_content = clean_text_comprehensive(content)
        
        logger.info(f"Cleaned text length: {len(cleaned_content)} characters")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        logger.info(f"Cleaned text saved to: {output_file}")
        
        # Print statistics
        original_lines = len(content.split('\n'))
        cleaned_lines = len(cleaned_content.split('\n'))
        logger.info(f"Lines: {original_lines} → {cleaned_lines}")
        
    except Exception as e:
        logger.error(f"Error processing {input_file}: {e}")

def main():
    # Define the extracted_text folder path
    extracted_text_folder = "extracted_text"
    
    # Get all .txt files from the extracted_text folder
    txt_files = []
    if os.path.exists(extracted_text_folder):
        for filename in os.listdir(extracted_text_folder):
            if filename.endswith('.txt'):
                txt_files.append(os.path.join(extracted_text_folder, filename))
    
    # Also include files in the current directory
    for filename in os.listdir('.'):
        if filename.endswith('.txt') and not filename.endswith('_cleaned.txt'):
            txt_files.append(filename)
    
    if not txt_files:
        logger.error("No .txt files found")
        return
    
    # Display available files and let user choose
    print("Available text files:")
    for i, file_path in enumerate(txt_files, 1):
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        print(f"{i}. {filename} ({file_size:,} bytes)")
    
    try:
        choice = int(input("\nEnter the number of the file you want to clean: ")) - 1
        if 0 <= choice < len(txt_files):
            selected_file = txt_files[choice]
            base_name = os.path.splitext(os.path.basename(selected_file))[0]
            output_file = f"{base_name}_cleaned.txt"
            
            logger.info(f"Processing {selected_file}...")
            process_file(selected_file, output_file)
            logger.info(f"Cleaning complete! Output saved to: {output_file}")
            
            # Ask if user wants to preview the result
            preview = input("\nWould you like to see a preview of the cleaned text? (y/n): ").lower()
            if preview == 'y':
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        preview_content = f.read()[:1000]  # First 1000 characters
                    print("\n" + "="*50)
                    print("PREVIEW OF CLEANED TEXT:")
                    print("="*50)
                    print(preview_content)
                    if len(preview_content) >= 1000:
                        print("\n[... truncated ...]")
                    print("="*50)
                except Exception as e:
                    logger.error(f"Error reading preview: {e}")
        else:
            logger.error("Invalid selection")
    except (ValueError, KeyboardInterrupt):
        logger.error("Invalid input or operation cancelled")

if __name__ == "__main__":
    main()