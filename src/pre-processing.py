import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk

# Download NLTK data silently with fallback
for resource in ['stopwords', 'punkt']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger('data_preprocessing')
        logger.error(f"Failed to download NLTK resource {resource}, proceeding with limited functionality")

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
    Returns empty string if any step fails.
    """
    ps = PorterStemmer()
    
    # Handle non-string inputs
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize with fallback
    try:
        tokens = nltk.word_tokenize(text)
    except Exception:
        return text  # Return original text if tokenization fails
    
    # Filter tokens
    filtered_tokens = []
    stop_words = set(stopwords.words('english')) if 'stopwords' in nltk.data.path else set()
    
    for word in tokens:
        if word.isalnum() and (not stop_words or word not in stop_words) and word not in string.punctuation:
            try:
                stemmed = ps.stem(word)
                filtered_tokens.append(stemmed)
            except Exception:
                filtered_tokens.append(word)  # Use unstemmed word if stemming fails
    
    return " ".join(filtered_tokens)

def preprocess_df(df, text_column='text', target_column='target'):
    """
    Preprocesses the DataFrame by encoding the target column, removing duplicates, and transforming the text column.
    """
    try:
        logger.debug('Starting preprocessing for DataFrame')
        
        # Check for required columns
        if text_column not in df.columns or target_column not in df.columns:
            raise KeyError(f"Missing required columns: {text_column} or {target_column}")

        # Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column].fillna(''))  # Handle NaN in target
        logger.debug('Target column encoded')

        # Remove duplicate rows
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')
        
        # Transform text column
        df[text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')
        return df
    
    except KeyError as e:
        logger.error(f'Column error: {e}')
        raise
    except Exception as e:
        logger.error(f'Preprocessing error: {e}')
        raise

def main(text_column='text', target_column='target'):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        # Define file paths
        raw_data_path = './data/raw'
        train_file = os.path.join(raw_data_path, 'train.csv')
        test_file = os.path.join(raw_data_path, 'test.csv')

        # Check if files exist
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            raise FileNotFoundError(f"Input files not found: {train_file} or {test_file}")

        # Load data with robust options
        train_data = pd.read_csv(train_file, encoding='utf-8', on_bad_lines='skip', dtype=str).fillna('')
        test_data = pd.read_csv(test_file, encoding='utf-8', on_bad_lines='skip', dtype=str).fillna('')
        logger.debug('Data loaded properly')

        # Preprocess data
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # Save processed data
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logger.debug(f'Processed data saved to {data_path}')
        
    except FileNotFoundError as e:
        logger.error(f'File error: {e}')
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f'Empty data error: {e}')
        raise
    except Exception as e:
        logger.error(f'Processing failed: {e}')
        raise

if __name__ == '__main__':
    main()