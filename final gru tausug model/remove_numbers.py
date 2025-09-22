#!/usr/bin/env python3
"""
Remove Numbers Script
Process 2: Additional Cleaning Scripts

This script removes all numerical digits from text files.
"""

import re
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def remove_numbers_from_text(text: str) -> str:
    """Remove all numerical digits from text"""
    # Remove standalone numbers
    text = re.sub(r'\b\d+\b', '', text)
    
    # Remove digits within words (but preserve letters)
    text = re.sub(r'\d', '', text)
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
    
    return text

def process_file(input_file: str, output_file: str = None) -> None:
    """Process a text file to remove numbers"""
    if output_file is None:
        output_file = input_file.replace('.txt', '_no_numbers.txt')
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        cleaned_content = remove_numbers_from_text(content)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        logger.info(f"Processed {input_file} -> {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing {input_file}: {e}")

def main():
    import os
    
    # Define the extracted_text folder path
    extracted_text_folder = "extracted_text"
    
    # Get all .txt files from the extracted_text folder
    txt_files = []
    if os.path.exists(extracted_text_folder):
        for filename in os.listdir(extracted_text_folder):
            if filename.endswith('.txt'):
                txt_files.append(os.path.join(extracted_text_folder, filename))
    
    if not txt_files:
        logger.error(f"No .txt files found in {extracted_text_folder} folder")
        return
    
    # Display available files and let user choose
    print("Available text files in extracted_text folder:")
    for i, file_path in enumerate(txt_files, 1):
        filename = os.path.basename(file_path)
        print(f"{i}. {filename}")
    
    try:
        choice = int(input("\nEnter the number of the file you want to process: ")) - 1
        if 0 <= choice < len(txt_files):
            selected_file = txt_files[choice]
            output_file = "remove_numbers.txt"
            
            logger.info(f"Processing {selected_file}...")
            process_file(selected_file, output_file)
            logger.info(f"Output saved to {output_file}")
        else:
            logger.error("Invalid selection")
    except (ValueError, KeyboardInterrupt):
        logger.error("Invalid input or operation cancelled")

if __name__ == "__main__":
    main()
