import pandas as pd
from pathlib import Path
import logging
import glob
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LatinPostProcessor:
    def __init__(self):
        self.input_dir = Path("datapipeline/data/input")
        self.output_dir = Path("datapipeline/data/output")
        
    def extract_data_from_raw_content(self, row):
        """Extract IPA and translations from raw content"""
        content = str(row['raw_content'])
        data = {
            'ipa_phoneme': [],
            'english_translation': [],
            'part_of_speech': []
        }
        
        # Extract IPA
        ipa_patterns = [
            r'\{\{la-IPA\|([^}]+)\}\}',      # {{la-IPA|...}}
            r'\{\{IPA\|la\|([^}]+)\}\}',     # {{IPA|la|...}}
            r'\{\{IPAchar\|([^}]+)\}\}',     # {{IPAchar|...}}
            r'\/([^\/]+)\/',                 # /phoneme/
            r'\[([^\]]+)\]',                 # [phoneme]
            r'\{\{pron\|la\|([^}]+)\}\}',    # {{pron|la|...}}
            r'pronunciation: ([^\.]+)'        # Direct pronunciation
        ]
        
        for pattern in ipa_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                # Clean up IPA notation
                ipa = match.split('|')[0].strip()
                ipa = re.sub(r'ann=\d+\|', '', ipa)  # Remove annotations
                ipa = re.sub(r'[\'ː]', 'ː', ipa)    # Standardize length markers
                if ipa:
                    data['ipa_phoneme'].append(ipa)
        
        # Extract translations/definitions - add more patterns
        translation_sections = [
            # Existing patterns
            r'===Definitions?===\n([^=]+)',
            r'# ([^\.]+)(?=\.|$)',
            r'\* ([^\.]+)(?=\.|$)',
            r'===Etymology===\n[^"]*?"([^"]+)"',
            r'===Translation===\n([^=]+)',
            r'Literally "([^"]+)"',
            r'From [^,.]*, (?:meaning|lit\.) "([^"]+)"',
            r': ([^\.]+)(?=\.|$)',
            r'\{\{(?:lb|label)\|la\|[^}]*\}\}\s*([^.\n]+)',
            r'\{\{defn\|la\|([^}]+)\}\}',
            
            # New patterns
            r'\{\{(?:m|mention)\|la\|[^}|]+\|t=([^}]+)\}\}',  # Template with translation
            r'(?<=\n)\d+\.\s*([^.\n]+)',  # Numbered definitions
            r'meaning (?:is |of )?["\']([^"\']+)["\']',  # Direct meaning statements
            r'\{\{uxi\|la\|[^|]+\|([^}]+)\}\}',  # Usage examples with translations
            r'From [^.]+, (?:meaning|signifying) ["\']([^"\']+)["\']',  # Etymology meanings
            r'\{\{(?:l|link)\|la\|[^}|]+\|([^}]+)\}\}',  # Link templates with translations
        ]
        
        for pattern in translation_sections:
            matches = re.findall(pattern, content, re.MULTILINE)
            for match in matches:
                # Clean up translation
                trans = match.strip()
                trans = re.sub(r'\{\{[^}]+\}\}', '', trans)  # Remove templates
                trans = re.sub(r'\[\[[^\]]+\]\]', '', trans)  # Remove wiki links
                if trans:
                    data['english_translation'].append(trans)
        
        # Extract part of speech - add more patterns
        pos_patterns = [
            # Existing patterns
            r'===(\w+)===[^=]',
            r'\{\{la-(?:noun|verb|adj|adv|prep|conj|interj)[^}]*\}\}',
            r'\|\s*PoS\s*=\s*([^|\n}]+)',
            r'\|\s*pos\s*=\s*([^|\n}]+)',
            r'\|\s*type\s*=\s*([^|\n}]+)',
            
            # New patterns
            r'\{\{head\|la\|([^}|]+)',  # Head template
            r'\{\{la-(?:decl|conj)[^}]*\}\}',  # Declension/conjugation templates
            r'# \{\{(?:lb|label)\|la\|([^}|]+)',  # Label templates
            r'\{\{(?:la|latin)[- ]([^}|]+)',  # Latin-specific templates
        ]
        
        pos_mapping = {
            # Existing mappings
            'noun': 'noun',
            'verb': 'verb',
            'adjective': 'adjective', 
            'adverb': 'adverb',
            'preposition': 'preposition',
            'conjunction': 'conjunction',
            'interjection': 'interjection',
            'pronoun': 'pronoun',
            'proper noun': 'proper noun',
            'participle': 'participle',
            'numeral': 'numeral',
            
            # Add more mappings
            'n': 'noun',
            'v': 'verb',
            'adj': 'adjective',
            'adv': 'adverb',
            'prep': 'preposition',
            'conj': 'conjunction',
            'interj': 'interjection',
            'pron': 'pronoun',
            'prop n': 'proper noun',
            'part': 'participle',
            'num': 'numeral',
            'determiner': 'determiner',
            'article': 'article',
            'prefix': 'prefix',
            'suffix': 'suffix',
            'phrase': 'phrase',
            'proverb': 'proverb',
            'letter': 'letter'
        }
        
        for pattern in pos_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                pos = match.lower().strip()
                if pos in pos_mapping:
                    data['part_of_speech'].append(pos_mapping[pos])
        
        # Join multiple values with semicolons and remove duplicates
        for key in data:
            if data[key]:
                data[key] = '; '.join(sorted(set(data[key])))
            else:
                data[key] = None
        
        # Add existing field data if available
        for field in ['ipa_phoneme', 'english_translation', 'part_of_speech']:
            if pd.notna(row.get(field)) and row[field]:
                if data[field]:
                    data[field] = f"{data[field]}; {row[field]}"
                else:
                    data[field] = row[field]
        
        data['original_characters'] = row['original_characters']
        return data

    def load_and_combine_files(self):
        """Load and combine all Latin-related files from input and output directories"""
        all_dfs = []
        
        # First load the full files (which have clean data)
        full_files = list(self.input_dir.glob("[Ll]atin_terms_full_*.csv"))
        logger.info(f"Found {len(full_files)} full files")
        for file in full_files:
            try:
                df = pd.read_csv(file)
                logger.info(f"Loaded full file: {file.name} with {len(df)} entries")
                all_dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        # Then load batch files
        batch_files = list(self.input_dir.glob("latin_batch_*.csv"))
        logger.info(f"Found {len(batch_files)} batch files")
        
        # Process batch files
        batch_df = pd.concat([pd.read_csv(f) for f in batch_files], ignore_index=True)
        logger.info(f"Combined {len(batch_files)} batch files with {len(batch_df)} entries")
        
        # Extract data from raw content
        extracted_data = []
        for _, row in batch_df.iterrows():
            data = self.extract_data_from_raw_content(row)
            extracted_data.append(data)
        
        extracted_df = pd.DataFrame(extracted_data)
        all_dfs.append(extracted_df)
        
        # Combine all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Remove duplicate entries based on original_characters
        combined_df = combined_df.drop_duplicates(subset=['original_characters'], keep='first')
        
        logger.info(f"Combined {len(all_dfs)} dataframes with {len(combined_df)} unique entries")
        
        # Log column statistics
        logger.info("\nCombined data columns:")
        for col in combined_df.columns:
            non_null = combined_df[col].notna().sum()
            logger.info(f"{col}: {non_null} non-null values ({(non_null/len(combined_df))*100:.1f}%)")
        
        return combined_df
        
    def clean_data(self, df):
        """Clean and validate the combined data"""
        if df is None or len(df) == 0:
            return None
        
        initial_count = len(df)
        logger.info(f"Starting cleaning with {initial_count} entries")
        
        # First, let's see what we have
        logger.info("\nInitial column statistics:")
        for col in df.columns:
            non_null = df[col].notna().sum()
            logger.info(f"{col}: {non_null} non-null values ({(non_null/len(df))*100:.1f}%)")
        
        # Keep only the columns we need
        keep_columns = ['original_characters', 'ipa_phoneme', 'english_translation', 'part_of_speech']
        df = df[keep_columns] if all(col in df.columns for col in keep_columns) else df
        
        # Remove rows where all fields are missing
        df = df.dropna(how='all')
        logger.info(f"Removed {initial_count - len(df)} completely empty rows")
        
        # Require original_characters and at least one of IPA or translation
        required_fields = ['original_characters']
        optional_fields = ['ipa_phoneme', 'english_translation']
        
        clean_df = df.dropna(subset=required_fields)
        clean_df = clean_df[clean_df[optional_fields].notna().any(axis=1)]
        logger.info(f"Removed {len(df) - len(clean_df)} entries missing required fields")
        
        # Clean IPA format
        clean_df.loc[:, 'ipa_phoneme'] = clean_df['ipa_phoneme'].apply(lambda x: re.sub(r'[/\[\]]', '', str(x)) if pd.notna(x) else x)
        
        # Remove duplicates based on original word, keeping the entry with the most non-null fields
        clean_df = clean_df.sort_values(by=df.columns.tolist(), na_position='last')
        clean_df = clean_df.drop_duplicates(subset=['original_characters'], keep='first')
        logger.info(f"After removing duplicates: {len(clean_df)} entries")
        
        # Sort by original characters
        clean_df = clean_df.sort_values('original_characters')
        
        # Clean up multi-value fields
        for col in ['ipa_phoneme', 'english_translation', 'part_of_speech']:
            if col in clean_df.columns:
                # Remove duplicate values within each cell
                clean_df[col] = clean_df[col].apply(lambda x: '; '.join(sorted(set(str(x).split('; ')))) if pd.notna(x) else x)
                
                # Remove empty strings and None values
                clean_df[col] = clean_df[col].replace('', pd.NA).replace('None', pd.NA)
        
        # Final statistics
        logger.info("\nFinal column statistics:")
        for col in clean_df.columns:
            non_null = clean_df[col].notna().sum()
            logger.info(f"{col}: {non_null} non-null values ({(non_null/len(clean_df))*100:.1f}%)")
        
        return clean_df
        
    def save_final_output(self, df):
        """Save the final cleaned data"""
        if df is None or len(df) == 0:
            logger.error("No data to save")
            return
            
        output_file = self.output_dir / "latin_final.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved final output to {output_file} with {len(df)} entries")
        
        # Print summary statistics
        logger.info("\nFinal Dataset Summary:")
        logger.info(f"Total entries: {len(df)}")
        logger.info(f"Unique terms: {df['original_characters'].nunique()}")
        logger.info(f"IPA coverage: {(df['ipa_phoneme'].notna().sum() / len(df)) * 100:.1f}%")
        logger.info(f"Translation coverage: {(df['english_translation'].notna().sum() / len(df)) * 100:.1f}%")

def main():
    processor = LatinPostProcessor()
    
    # Load and combine files
    combined_df = processor.load_and_combine_files()
    
    # Clean and validate data
    cleaned_df = processor.clean_data(combined_df)
    
    # Save final output
    processor.save_final_output(cleaned_df)

if __name__ == "__main__":
    main()
