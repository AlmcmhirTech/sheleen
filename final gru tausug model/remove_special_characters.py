#!/usr/bin/env python3
"""
Remove Special Characters Script
Process 2: Additional Cleaning Scripts

This script removes non-essential special characters while preserving
sentence punctuation and internal apostrophes/hyphens.
"""

import re
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_special_characters(text: str, preserve_chars: str = ".,?!'-") -> str:
    """
    Remove non-essential special characters
    
    Args:
        text: Input text to clean
        preserve_chars: Characters to preserve (default: ".,?!'-")
    
    Returns:
        str: Cleaned text
    """
    # Create pattern to keep only letters, numbers, whitespace, and specified characters
    pattern = f'[^\\w\\s{re.escape(preserve_chars)}]'
    
    # Remove unwanted special characters
    text = re.sub(pattern, '', text)
    
    # Clean up multiple consecutive punctuation marks (except ellipsis)
    text = re.sub(r'([.!?]){2,}', r'\1', text)
    
    # Remove isolated punctuation (punctuation not attached to words)
    text = re.sub(r'\s+([.,!?])\s+', r'\1 ', text)
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace from each line
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
    
    # Remove empty lines
    text = re.sub(r'\n\s*\n', '\n', text)
    
    return text.strip()

def process_file(input_file: str, output_file: str = None, preserve_chars: str = ".,?!'-") -> None:
    """Process a text file to remove special characters"""
    if output_file is None:
        output_file = input_file.replace('.txt', '_clean_chars.txt')
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_lines = len(content.splitlines())
        cleaned_content = clean_special_characters(content, preserve_chars)
        cleaned_lines = len(cleaned_content.splitlines())
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        logger.info(f"Processed {input_file} -> {output_file}")
        logger.info(f"Lines: {original_lines} -> {cleaned_lines}")
        
    except Exception as e:
        logger.error(f"Error processing {input_file}: {e}")

def batch_process_files(file_paths: list, preserve_chars: str = ".,?!'-") -> None:
    """Process multiple files in batch"""
    for file_path in file_paths:
        logger.info(f"Processing {file_path}...")
        process_file(file_path, preserve_chars=preserve_chars)

def main():
    parser = argparse.ArgumentParser(description='Remove non-essential special characters from text files')
    parser.add_argument('input_file', help='Input text file path')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    parser.add_argument('-p', '--preserve', default=".,?!'-",
                       help='Characters to preserve (default: ".,?!\'-")')
    parser.add_argument('--batch', nargs='+', help='Process multiple files')
    
    args = parser.parse_args()
    
    if args.batch:
        batch_process_files(args.batch, args.preserve)
    else:
        process_file(args.input_file, args.output, args.preserve)

if __name__ == "__main__":
    main()
