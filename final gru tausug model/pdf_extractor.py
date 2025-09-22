#!/usr/bin/env python3
"""
PDF Text Extractor for Tausug Language Model Training
Process 1: Extraction & Preprocessing

This script extracts Tausug text from PDF files, applying strict filtering
to exclude non-Tausug content and genealogical data.
"""

import os
import re
import json
import PyPDF2
import pdfplumber
from collections import Counter, OrderedDict
from typing import List, Tuple, Dict, Set
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TausugTextExtractor:
    def __init__(self, min_sentence_length: int = 3):
        self.min_sentence_length = min_sentence_length
        self.tausug_sentences = []
        self.tausug_paragraphs = []
        self.stats = {
            'total_sentences': 0,
            'total_words': 0,
            'unique_words': 0,
            'word_frequency': {},
            'processed_files': []
        }
        
        # Common Tausug words/patterns for language identification
        self.tausug_indicators = {
            'ha', 'sin', 'bang', 'nag', 'mag', 'pag', 'tau', 'sug', 'allah',
            'muslim', 'islam', 'salah', 'sabab', 'dayn', 'kahit', 'biya',
            'awn', 'way', 'miyatup', 'magtuy', 'nah', 'dih', 'pila', 'pisan',
            'bakas', 'tuud', 'misan', 'hangkan', 'duun', 'ditu', 'adtu',
            'amu', 'ini', 'yan', 'siin', 'marayaw', 'mahinang', 'wayruun'
        }
        
        # Genealogical indicators to exclude
        self.genealogical_patterns = [
            r'\b(born|son of|daughter of|father|mother|married|died|husband|wife)\b',
            r'\b\d{4}\s*[-–]\s*\d{4}\b',  # Birth-death years
            r'\b(family tree|genealogy|lineage|ancestry)\b',
            r'\b[A-Z][a-z]+\s+bin\s+[A-Z][a-z]+\b',  # Arabic naming patterns
            r'\b[A-Z][a-z]+\s+bint\s+[A-Z][a-z]+\b',
            r'\b(mga anak|mga asawa|mga magulang)\b'
        ]
        
        # Non-Tausug language indicators
        self.non_tausug_indicators = {
            'english': {'the', 'and', 'or', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'will', 'would', 'should', 'could'},
            'tagalog': {'ang', 'ng', 'sa', 'mga', 'ay', 'na', 'at', 'para', 'kung', 'kapag', 'dahil'},
            'cebuano': {'ug', 'sa', 'nga', 'kay', 'naa', 'wala', 'adto', 'nimo', 'nako', 'ato'}
        }

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using pdfplumber (better for complex layouts)"""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            logger.info(f"Successfully extracted text from {pdf_path}")
        except Exception as e:
            logger.warning(f"Failed to extract with pdfplumber from {pdf_path}: {e}")
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                logger.info(f"Successfully extracted text with PyPDF2 from {pdf_path}")
            except Exception as e2:
                logger.error(f"Failed to extract text from {pdf_path}: {e2}")
                return ""
        return text

    def clean_text(self, text: str) -> str:
        """Initial text cleaning"""
        # Remove page numbers and headers/footers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Remove reference markers and numbers
        text = re.sub(r'\[\d+\]|\(\d+\)|\^\d+|\d+\.(?=\s|$)', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text.strip()

    def is_genealogical_content(self, text: str) -> bool:
        """Check if text contains genealogical information"""
        text_lower = text.lower()
        for pattern in self.genealogical_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        # Check for long lists of names (common in genealogies)
        words = text.split()
        capitalized_words = [w for w in words if w[0].isupper() and len(w) > 2]
        if len(capitalized_words) > len(words) * 0.6 and len(words) > 5:
            return True
            
        return False

    def is_likely_tausug(self, text: str) -> bool:
        """Determine if text is likely in Tausug language"""
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < self.min_sentence_length:
            return False
        
        # Check for non-Tausug language indicators
        for lang, indicators in self.non_tausug_indicators.items():
            if sum(1 for word in words if word in indicators) > len(words) * 0.3:
                return False
        
        # Check for Tausug indicators
        tausug_matches = sum(1 for word in words if word in self.tausug_indicators)
        
        # If we have clear Tausug indicators, it's likely Tausug
        if tausug_matches > 0:
            return True
        
        # If no clear indicators but also no non-Tausug indicators, be conservative
        # Check for patterns that suggest Tausug structure
        if any(pattern in text.lower() for pattern in ['ha ', ' sin ', ' bang ', ' nag', ' mag']):
            return True
            
        return False

    def process_sentence(self, sentence: str) -> str:
        """Process individual sentence according to requirements"""
        # Convert to lowercase
        sentence = sentence.lower()
        
        # Remove numbers and references
        sentence = re.sub(r'\d+', '', sentence)
        sentence = re.sub(r'\[\d*\]|\(\d*\)|\^\d*', '', sentence)
        
        # Remove non-essential special characters (keep . , ? ! and internal apostrophes/hyphens)
        sentence = re.sub(r'[^\w\s\.\,\?\!\'\-]', '', sentence)
        
        # Clean up extra spaces
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        
        return sentence

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Split on sentence endings
        sentences = re.split(r'[.!?]+', text)
        
        processed_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) == 0:
                continue
                
            # Skip if contains genealogical content
            if self.is_genealogical_content(sentence):
                continue
                
            # Process the sentence
            clean_sentence = self.process_sentence(sentence)
            
            # Check minimum length and Tausug likelihood
            words = clean_sentence.split()
            if len(words) >= self.min_sentence_length and self.is_likely_tausug(clean_sentence):
                processed_sentences.append(clean_sentence)
        
        return processed_sentences

    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        paragraphs = text.split('\n\n')
        
        processed_paragraphs = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if len(paragraph) == 0:
                continue
                
            # Skip if contains genealogical content
            if self.is_genealogical_content(paragraph):
                continue
                
            # Check if entire paragraph is likely Tausug
            if self.is_likely_tausug(paragraph):
                # Process the paragraph
                clean_paragraph = self.process_sentence(paragraph)  # Same processing as sentences
                if len(clean_paragraph.split()) >= self.min_sentence_length:
                    processed_paragraphs.append(clean_paragraph)
        
        return processed_paragraphs

    def process_pdf_files(self, pdf_directories: List[str]) -> None:
        """Process all PDF files in the given directories"""
        all_sentences = []
        all_paragraphs = []
        
        for directory in pdf_directories:
            if not os.path.exists(directory):
                logger.warning(f"Directory not found: {directory}")
                continue
                
            logger.info(f"Processing directory: {directory}")
            
            for filename in os.listdir(directory):
                if filename.lower().endswith('.pdf'):
                    pdf_path = os.path.join(directory, filename)
                    logger.info(f"Processing file: {filename}")
                    
                    # Extract text
                    raw_text = self.extract_text_from_pdf(pdf_path)
                    if not raw_text:
                        continue
                    
                    # Clean text
                    cleaned_text = self.clean_text(raw_text)
                    
                    # Extract sentences and paragraphs
                    sentences = self.split_into_sentences(cleaned_text)
                    paragraphs = self.split_into_paragraphs(cleaned_text)
                    
                    all_sentences.extend(sentences)
                    all_paragraphs.extend(paragraphs)
                    
                    self.stats['processed_files'].append({
                        'filename': filename,
                        'sentences_extracted': len(sentences),
                        'paragraphs_extracted': len(paragraphs)
                    })
                    
                    logger.info(f"Extracted {len(sentences)} sentences and {len(paragraphs)} paragraphs from {filename}")
        
        # Deduplicate sentences while preserving order
        seen = set()
        self.tausug_sentences = []
        for sentence in all_sentences:
            if sentence not in seen:
                seen.add(sentence)
                self.tausug_sentences.append(sentence)
        
        # Keep paragraphs (duplicates less likely but can deduplicate if needed)
        self.tausug_paragraphs = list(OrderedDict.fromkeys(all_paragraphs))
        
        logger.info(f"Total unique sentences: {len(self.tausug_sentences)}")
        logger.info(f"Total paragraphs: {len(self.tausug_paragraphs)}")

    def generate_statistics(self) -> None:
        """Generate statistics for the extracted text"""
        # Combine all text for word analysis
        all_text = ' '.join(self.tausug_sentences + self.tausug_paragraphs)
        words = re.findall(r'\b\w+\b', all_text.lower())
        
        self.stats['total_sentences'] = len(self.tausug_sentences)
        self.stats['total_words'] = len(words)
        self.stats['unique_words'] = len(set(words))
        
        # Word frequency (top 50)
        word_freq = Counter(words)
        self.stats['word_frequency'] = dict(word_freq.most_common(50))

    def save_outputs(self, output_dir: str = '.') -> None:
        """Save all outputs to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save sentences
        sentences_file = os.path.join(output_dir, 'tausug_sentences.txt')
        with open(sentences_file, 'w', encoding='utf-8') as f:
            for sentence in self.tausug_sentences:
                f.write(sentence + '\n')
        logger.info(f"Saved {len(self.tausug_sentences)} sentences to {sentences_file}")
        
        # Save paragraphs
        paragraphs_file = os.path.join(output_dir, 'tausug_paragraphs.txt')
        with open(paragraphs_file, 'w', encoding='utf-8') as f:
            for paragraph in self.tausug_paragraphs:
                f.write(paragraph + '\n')
        logger.info(f"Saved {len(self.tausug_paragraphs)} paragraphs to {paragraphs_file}")
        
        # Save statistics
        stats_file = os.path.join(output_dir, 'tausug_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved statistics to {stats_file}")

def main():
    """Main execution function"""
    # Configure paths
    base_dir = r"c:\Users\Tong\Desktop\final gru tausug model"
    pdf_directories = [
        os.path.join(base_dir, "pdf_data")  # Your PDFs are in the pdf_data folder
    ]
    output_dir = os.path.join(base_dir, "extracted_text")
    
    # Initialize extractor
    extractor = TausugTextExtractor(min_sentence_length=3)
    
    # Process PDFs
    logger.info("Starting PDF text extraction...")
    extractor.process_pdf_files(pdf_directories)
    
    # Generate statistics
    logger.info("Generating statistics...")
    extractor.generate_statistics()
    
    # Save outputs
    logger.info("Saving outputs...")
    extractor.save_outputs(output_dir)
    
    logger.info("Extraction complete!")
    print(f"\nExtraction Summary:")
    print(f"- Total sentences extracted: {extractor.stats['total_sentences']}")
    print(f"- Total paragraphs extracted: {len(extractor.tausug_paragraphs)}")
    print(f"- Total words: {extractor.stats['total_words']}")
    print(f"- Unique words: {extractor.stats['unique_words']}")
    print(f"- Files processed: {len(extractor.stats['processed_files'])}")
    print(f"\nOutput files saved to: {output_dir}")

if __name__ == "__main__":
    main()
