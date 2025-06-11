import pandas as pd
import re

def preprocess_tweets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess labeled tweet dataframe for sentiment classification.

    Parameters:
        df (pd.DataFrame): Input dataframe with columns ['text', 'sentiment', ...]

    Returns:
        pd.DataFrame: Preprocessed dataframe with cleaned tweets.
    """
    df = df.copy()
    print(df.columns)
    df['cleaned_tweets'] = df['text_en'].astype(str).str.lower()
    df['cleaned_tweets'] = df['cleaned_tweets'].apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x, flags=re.MULTILINE))
    df['cleaned_tweets'] = df['cleaned_tweets'].apply(lambda x: re.sub(r'@\w+', '', x))

    def remove_emoji(text):
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F700-\U0001F77F"  # alchemical symbols
            u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            u"\U0001FA00-\U0001FA6F"  # Chess Symbols
            u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            u"\U00002702-\U000027B0"  # Dingbats
            u"\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub(r'', text)
    df['cleaned_tweets'] = df['cleaned_tweets'].apply(remove_emoji)
    
    # Encode label
    if 'label' in df.columns:
        df = df[df['label'] != 'unknown'].reset_index(drop=True)
        df['labels'] = df['label'].map({
            "neutral": 0,
            "negative": 1,
            "positive": 2
        })

    # Reorder columns
    columns = [col for col in df.columns if col != 'label'] + ['label']
    df = df[columns]

    return df
