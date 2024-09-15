import pandas as pd
import re
import json

def load_contractions(filepath):
    """ Load contractions from a JSON file. """
    with open(filepath, 'r', encoding='utf-8') as file:
        contractions = json.load(file)
    return contractions

def load_stopwords(filepath):
    """ Load stopwords from a JSON file. """
    with open(filepath, 'r', encoding='utf-8') as file:
        stopwords_list = json.load(file)
    return stopwords_list

def normalize_turkish_chars(text, to_english=False):
    """ Normalize Turkish characters to English equivalents if to_english is True. """
    if to_english:
        turkish_char_map = str.maketrans(
            "çğıöşüÇĞİÖŞÜ",
            "cgiosuCGIOSU"
        )
    else:
        turkish_char_map = str.maketrans(
            "çğıöşü",
            "cgiosu"
        )
    return text.translate(turkish_char_map)

def preprocess_text(text, contractions, stopwords):
    """ Preprocess the input text. """
    if not isinstance(text, str):
        return text  # If text is not a string, return it as-is

    # Normalize Turkish characters
    text = normalize_turkish_chars(text)
    
    # Apply to lowercase and handle Turkish 'İ' correctly
    text = text.replace('İ', 'i').lower()
    
    # Remove URLs, mentions, hashtags, numeric values, special characters, single characters, extra spaces, stopwords, and expand contractions
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"[0-9]", "", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\b\w\b", "", text)
    text = re.sub('\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stopwords])
    text = ' '.join([contractions.get(word, word) for word in text.split()])

    return text

def main():
    """ Main function to preprocess and save cleaned data. """
    df = pd.read_csv('data/processed/merged_df.csv')
    contractions = load_contractions('C:/Users/Ali Riza Ercan/Documents/GitHub/PassoAssist/model_test/contractions.json')
    stopwords = load_stopwords('C:/Users/Ali Riza Ercan/Documents/GitHub/PassoAssist/model_test/stopwords.json')
    df['clean_text'] = df['text'].apply(lambda x: preprocess_text(x, contractions, stopwords))
    df.to_csv('data/processed/cleaned_df.csv', index=False)

if __name__ == "__main__":
    main()
