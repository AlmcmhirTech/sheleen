#!/usr/bin/env python3
"""
Filter Non-Tausug Script
Process 2: Additional Cleaning Scripts

This script filters out sentences that are not in Tausug language.
"""

import re
import argparse
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TausugLanguageFilter:
    def __init__(self):
        # Common Tausug words/morphemes for identification
        self.tausug_indicators = {
            'ha', 'sin', 'bang', 'nag', 'mag', 'pag', 'tau', 'sug', 'allah',
            'muslim', 'islam', 'salah', 'sabab', 'dayn', 'kahit', 'biya',
            'awn', 'way', 'miyatup', 'magtuy', 'nah', 'dih', 'pila', 'pisan',
            'bakas', 'tuud', 'misan', 'hangkan', 'duun', 'ditu', 'adtu',
            'amu', 'ini', 'yan', 'siin', 'marayaw', 'mahinang', 'wayruun',
            'naug', 'kayu', 'bata', 'tao', 'bagas', 'tubig', 'kaun', 'buh',
            'adlaw', 'gabii', 'ulu', 'lima', 'mata', 'baran', 'dughan',
            'pagkamatay', 'pagkabii', 'pagtutulun', 'pagtuyun', 'magsulat'
        }
        
        # Non-Tausug language indicators
        self.non_tausug_indicators = {
            'english': {
                'the', 'and', 'or', 'is', 'are', 'was', 'were', 'have', 'has', 
                'had', 'will', 'would', 'should', 'could', 'this', 'that', 
                'with', 'from', 'they', 'them', 'their', 'there', 'where',
                'when', 'what', 'who', 'how', 'why', 'because', 'if', 'then',
                'but', 'so', 'for', 'to', 'of', 'in', 'on', 'at', 'by'
            },
            'tagalog': {
                'ang', 'ng', 'sa', 'mga', 'ay', 'na', 'at', 'para', 'kung', 
                'kapag', 'dahil', 'kasi', 'pero', 'kaya', 'naman', 'din', 
                'rin', 'siya', 'ako', 'ikaw', 'kami', 'kayo', 'sila',
                'ito', 'iyan', 'iyon', 'dito', 'diyan', 'doon', 'may',
                'mayroon', 'wala', 'walang', 'maging', 'ginawa', 'gagawin'
            },
            'cebuano': {
                'ug', 'sa', 'nga', 'kay', 'naa', 'wala', 'adto', 'nimo', 
                'nako', 'ato', 'amo', 'sila', 'kita', 'dili', 'mao', 
                'ingon', 'basin', 'bisan', 'unsaon', 'ngano', 'asa',
                'kanus', 'kinsa', 'unsa', 'pila', 'daghan', 'gamay'
            }
        }

    def is_likely_tausug(self, text: str, min_confidence: float = 0.3) -> bool:
        """
        Determine if text is likely in Tausug language
        
        Args:
            text: Input text to analyze
            min_confidence: Minimum confidence threshold (0.0 to 1.0)
        
        Returns:
            bool: True if likely Tausug, False otherwise
        """
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < 3:  # Too short to determine
            return False
        
        # Check for strong non-Tausug indicators
        for lang, indicators in self.non_tausug_indicators.items():
            non_tausug_matches = sum(1 for word in words if word in indicators)
            if non_tausug_matches > len(words) * 0.25:  # More than 25% non-Tausug words
                logger.debug(f"Rejected due to {lang} indicators: {non_tausug_matches}/{len(words)}")
                return False
        
        # Check for Tausug indicators
        tausug_matches = sum(1 for word in words if word in self.tausug_indicators)
        tausug_ratio = tausug_matches / len(words)
        
        # If we have clear Tausug indicators above threshold
        if tausug_ratio >= min_confidence:
            logger.debug(f"Accepted with Tausug ratio: {tausug_ratio:.2f}")
            return True
        
        # Check for Tausug morphological patterns
        tausug_patterns = [
            r'\bmag\w+',     # mag- prefix
            r'\bnag\w+',     # nag- prefix  
            r'\bpag\w+',     # pag- prefix
            r'\bka\w+an\b',  # ka-...-an circumfix
            r'\bha\s+\w+',   # ha + word pattern
            r'\bsin\s+\w+',  # sin + word pattern
        ]
        
        pattern_matches = 0
        for pattern in tausug_patterns:
            if re.search(pattern, text.lower()):
                pattern_matches += 1
        
        # If we have morphological evidence and no strong counter-evidence
        if pattern_matches >= 2:
            logger.debug(f"Accepted due to morphological patterns: {pattern_matches}")
            return True
        
        # Conservative approach: if unclear, exclude
        logger.debug(f"Rejected - insufficient evidence. Tausug ratio: {tausug_ratio:.2f}, patterns: {pattern_matches}")
        return False

def filter_tausug_lines(input_file: str, output_file: str = None, min_confidence: float = 0.3) -> None:
    """Filter lines to keep only Tausug content"""
    if output_file is None:
        output_file = input_file.replace('.txt', '_filtered.txt')
    
    filter_obj = TausugLanguageFilter()
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        tausug_lines = []
        total_lines = len(lines)
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line and filter_obj.is_likely_tausug(line, min_confidence):
                tausug_lines.append(line)
            
            if i % 100 == 0:
                logger.info(f"Processed {i}/{total_lines} lines...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in tausug_lines:
                f.write(line + '\n')
        
        logger.info(f"Filtered {len(tausug_lines)}/{total_lines} lines as Tausug")
        logger.info(f"Output saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing {input_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Filter non-Tausug sentences from text files')
    parser.add_argument('input_file', nargs='?', help='Input text file path (optional - will process extracted_text folder if not provided)')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    parser.add_argument('-c', '--confidence', type=float, default=0.3, 
                       help='Minimum confidence threshold (0.0-1.0, default: 0.3)')
    parser.add_argument('--folder', default='extracted_text', 
                       help='Folder to process if no input file specified (default: extracted_text)')
    
    args = parser.parse_args()
    
    if not 0.0 <= args.confidence <= 1.0:
        logger.error("Confidence must be between 0.0 and 1.0")
        return
    
    if args.input_file:
        # Process single file
        filter_tausug_lines(args.input_file, args.output, args.confidence)
    else:
        # Process all text files in the specified folder
        import os
        folder_path = args.folder
        
        if not os.path.exists(folder_path):
            logger.error(f"Folder '{folder_path}' does not exist")
            return
        
        # Find all .txt files in the folder
        txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        
        if not txt_files:
            logger.error(f"No .txt files found in '{folder_path}'")
            return
        
        logger.info(f"Found {len(txt_files)} text files to process in '{folder_path}'")
        
        for txt_file in txt_files:
            input_path = os.path.join(folder_path, txt_file)
            output_path = os.path.join(folder_path, txt_file.replace('.txt', '_filtered.txt'))
            
            logger.info(f"Processing: {txt_file}")
            filter_tausug_lines(input_path, output_path, args.confidence)
            logger.info(f"Completed: {txt_file} -> {txt_file.replace('.txt', '_filtered.txt')}")
        
        logger.info("All files processed successfully!")

if __name__ == "__main__":
    main()
