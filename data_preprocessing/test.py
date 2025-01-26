import pandas as pd
import os

# Define file paths relative to the Qhacks directory
base_path = "./data"  # Update this to your Qhacks directory path
old_english_file = os.path.join(base_path, "Old_English_terms_full_20250125_071247_raw.csv")
us_english_file = os.path.join(base_path, "en_US.txt")
uk_english_file = os.path.join(base_path, "en_UK.txt")
pie_file = os.path.join(base_path, "PIE_roots_parsed_20250124_215037_raw.csv")
# Function to process US/UK English files
def process_language_file(file_path, language):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            if '\t' in line:  # Assuming tab-separated format
                word, phonetic = line.strip().split('\t', 1)
                data.append({
                    "word": word.strip(),
                    "language": language,
                    "phonetic_representation": phonetic.strip(),
                    "english_translation": word.strip()  # US/UK English words are their own translations
                })
    return pd.DataFrame(data)

us_english_df = process_language_file(us_english_file, "US English")
uk_english_df = process_language_file(uk_english_file, "UK English")

# Rename columns for consistency
old_english_df = pd.read_csv(old_english_file)

if "raw_content" in old_english_df.columns:
    old_english_df.drop(columns=["raw_content"], inplace=True)

if "title" in old_english_df.columns:
    old_english_df.drop(columns=["title"], inplace=True)

old_english_df.rename(
    columns={
        "original_characters": "word",
        "english_translation": "english_translation",
        "ipa_phoneme": "phonetic_representation"
    },
    inplace=True
)
old_english_df["language"] = "Old English"

pie_df = pd.read_csv(pie_file)
if "description" in pie_df.columns:
     pie_df.drop(columns=["description"], inplace=True)
# Rename columns for consistency
pie_df.rename(
    columns={
        "original_characters": "word",
        "english_translation": "english_translation",
        "ipa_phoneme": "phonetic_representation"
    },
    inplace=True
)
pie_df["language"] = "PIE"
combined_df = pd.concat([pie_df, old_english_df, us_english_df, uk_english_df], ignore_index=True)
# Add a time_period column based on the language
time_period_mapping = {
    "US English": 0,
    "UK English": -300,
    "Old English": -1100,
    "PIE": -5000
}
combined_df["time_period"] = combined_df["language"].map(time_period_mapping)

# Reorganize columns
columns = ["word", "english_translation", "language", "phonetic_representation", "time_period"]
final_df = combined_df[columns]
# Remove rows where any value is null or blank
final_df = final_df.replace(r"^\s*$", None, regex=True)  # Replace blank values with NaN
final_df.dropna(inplace=True)  # Drop rows with any NaN values

# Remove duplicates where 'word' and 'language' are the same
final_df = final_df.drop_duplicates(subset=["word", "language"], keep="first")

# Save the cleaned DataFrame to a CSV file
output_path = os.path.join(base_path, "combined_language_data_with_translation_cleaned.csv")
final_df.to_csv(output_path, index=False)
