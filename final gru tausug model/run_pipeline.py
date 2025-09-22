#!/usr/bin/env python3
"""
Quick Setup and Execution Script
Automates the entire Tausug text extraction pipeline
"""

import subprocess
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_dependencies():
    """Install required packages"""
    logger.info("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2==3.0.1", "pdfplumber==0.10.3"])
        logger.info("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def run_extraction():
    """Run the main PDF extraction script"""
    logger.info("Starting PDF text extraction...")
    try:
        subprocess.check_call([sys.executable, "pdf_extractor.py"])
        logger.info("Extraction completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Extraction failed: {e}")
        return False

def check_outputs():
    """Check if output files were created"""
    output_dir = "extracted_text"
    expected_files = ["tausug_sentences.txt", "tausug_paragraphs.txt", "tausug_stats.json"]
    
    if not os.path.exists(output_dir):
        logger.warning(f"Output directory {output_dir} not found")
        return False
    
    missing_files = []
    for file in expected_files:
        file_path = os.path.join(output_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        logger.warning(f"Missing output files: {missing_files}")
        return False
    
    logger.info("All output files created successfully!")
    
    # Display summary
    try:
        import json
        stats_file = os.path.join(output_dir, "tausug_stats.json")
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        print("\n" + "="*50)
        print("EXTRACTION SUMMARY")
        print("="*50)
        print(f"Total sentences extracted: {stats['total_sentences']}")
        print(f"Total paragraphs extracted: {len(stats.get('paragraphs', []))}")
        print(f"Total words: {stats['total_words']}")
        print(f"Unique words: {stats['unique_words']}")
        print(f"Files processed: {len(stats['processed_files'])}")
        
        if stats['word_frequency']:
            print(f"\nTop 10 most common words:")
            for i, (word, freq) in enumerate(list(stats['word_frequency'].items())[:10], 1):
                print(f"  {i:2d}. {word}: {freq}")
        
        print(f"\nOutput files saved to: {output_dir}/")
        print("- tausug_sentences.txt (ready for model training)")
        print("- tausug_paragraphs.txt (ready for model training)")
        print("- tausug_stats.json (corpus statistics)")
        print("="*50)
        
    except Exception as e:
        logger.warning(f"Could not display summary: {e}")
    
    return True

def main():
    """Main execution function"""
    print("Tausug Text Extraction Pipeline")
    print("="*40)
    
    # Step 1: Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies. Please install manually:")
        print("   pip install PyPDF2==3.0.1 pdfplumber==0.10.3")
        return 1
    
    # Step 2: Run extraction
    if not run_extraction():
        print("❌ Extraction failed. Check the logs for details.")
        return 1
    
    # Step 3: Check outputs
    if not check_outputs():
        print("❌ Some output files are missing. Check the extraction process.")
        return 1
    
    print("\n✅ Pipeline completed successfully!")
    print("\nNext steps:")
    print("1. Review the extracted text files in the 'extracted_text/' directory")
    print("2. Use 'tausug_sentences.txt' for sentence-level model training")
    print("3. Use 'tausug_paragraphs.txt' for document-level model training")
    print("4. Reference 'tausug_stats.json' for corpus statistics")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
