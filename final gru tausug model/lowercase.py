#!/usr/bin/env python3
"""
Lowercase Conversion Script
Process 2: Additional Cleaning Scripts

This script converts all text to lowercase for consistent processing.
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_lowercase(text: str) -> str:
    """Convert text to lowercase while preserving structure"""
    return text.lower()

def process_file(input_file: str, output_file: str = None) -> None:
    """Process a text file to convert to lowercase"""
    if output_file is None:
        output_file = input_file.replace('.txt', '_lowercase.txt')
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lowercase_content = convert_to_lowercase(content)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(lowercase_content)
        
        logger.info(f"Converted {input_file} to lowercase -> {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing {input_file}: {e}")

def batch_process_files(file_paths: list) -> None:
    """Process multiple files in batch"""
    for file_path in file_paths:
        logger.info(f"Converting {file_path} to lowercase...")
        process_file(file_path)

def main():
    parser = argparse.ArgumentParser(description='Convert text files to lowercase')
    parser.add_argument('input_file', help='Input text file path')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    parser.add_argument('--batch', nargs='+', help='Process multiple files')
    
    args = parser.parse_args()
    
    if args.batch:
        batch_process_files(args.batch)
    else:
        process_file(args.input_file, args.output)

if __name__ == "__main__":
    main()
