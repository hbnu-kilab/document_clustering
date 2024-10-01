import tiktoken
import jsonlines
from tqdm import tqdm
import matplotlib.pyplot as plt

def read_and_tokenize_documents(file_path):
    documents = []
    token_counts = []
    tokenizer = tiktoken.get_encoding("cl100k_base")

    with jsonlines.open(file_path, mode='r') as reader:
        for obj in tqdm(reader, desc="Reading and Tokenizing Documents"):
            content = obj['contents']
            documents.append(content)
            token_counts.append(len(tokenizer.encode(content)))
    
    return token_counts

jsonl_file_path = 'D:/docs.jsonl'
token_counts = read_and_tokenize_documents(jsonl_file_path)

import pandas as pd
# Count occurrences of each token count
token_counts_series = pd.Series(token_counts)
top_token_counts = token_counts_series.value_counts()

# Plot the top token counts
plt.figure(figsize=(10, 5))
top_token_counts.sort_index().plot(kind='bar', color='blue', alpha=0.7)
plt.xlabel('Token Count')
plt.ylabel('Number of Documents')
plt.title('Top Token Counts Across Documents')

# Remove count labels from x-axis
plt.xticks([])

# 500 단위로 수직선 표시
for x in range(0, top_token_counts.index.max() + 1, 500):
    plt.axvline(x=x, color='red', linestyle='--', linewidth=0.5)

# 최대 토큰 개수로 x축 범위 설정
plt.xlim(0,1500)

plt.show()