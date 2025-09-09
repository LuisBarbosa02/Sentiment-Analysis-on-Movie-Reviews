# Import libraries
import pandas as pd
from pathlib import Path
import os

# Load data
def load_data(path):
    """
    A function to load the data to be used.
    :param path: Relative path to the data.
    :return: A table with the data in it.
    :rtype: pandas.core.frame.DataFrame
    """
    data_dir = Path(path)

    if path == 'datasets/review_polarity':
        rows = []
        for label in ('pos', 'neg'):
            folder = data_dir / label
            for file in sorted(folder.glob('*.txt')):
                text = file.read_text(encoding='utf8', errors='ignore')

                tag = file.name.split("_", 1)[0]
                idx = int(tag.replace('cv', ''))
                fold = idx // 100+1

                rows.append({'text': text, 'label': 1 if label=='pos' else 0, 'fold': fold})

        return pd.DataFrame(rows)

    if path == 'datasets/rt-polaritydata':
        rows = []
        for file_name in os.listdir(path):
            with open(f"{path}/{file_name}", 'rb') as file:
                content = file.read().decode('utf-8', errors='ignore').split('\n')
                content = [{'text': text, 'label': 1 if 'pos' in file_name else 0} for text in content]
                del content[-1]
                rows += content
        return pd.DataFrame(rows)

if __name__ == '__main__':
    pass