import os
import threading
from tqdm import tqdm  # 추가
from dotenv import load_dotenv
from accelerate import Accelerator
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from raptor import BaseEmbeddingModel
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv(dotenv_path='apikey.env')
# 환경 변수에서 API 키 가져오기
api_key = os.getenv("OPENAI_API_KEY")

# API 키가 None이면 오류를 출력하고 프로그램 종료
if api_key is None:
    raise ValueError("API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
class CustomEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="BAAI/bge-multilingual-gemma2"):
        self.model = SentenceTransformer(model_name, model_kwargs={"torch_dtype": torch.float16})

    def create_embedding(self, text):
        return self.model.encode(text)   
        
print("Starting model initialization...")
custom_embedding = CustomEmbeddingModel()

custom_config = RetrievalAugmentationConfig(
    embedding_model=custom_embedding
)

RA = RetrievalAugmentation(config=custom_config)

def read_documents_from_jsonl(file_path):
    documents = []
    with jsonlines.open(file_path) as reader:
        for obj in tqdm(reader, desc="Reading Documents"):  # tqdm 사용
            documents.append(obj['contents'])
    return documents

jsonl_file_path = 'C:/Users/USER/Downloads/mrtydi-v1.1-korean/mrtydi-v1.1-korean/collection/docs.jsonl/docs1.jsonl'
documents = read_documents_from_jsonl(jsonl_file_path)

RA.add_documents(documents) 

SAVE_PATH = "C:/Users/USER/Desktop/clustering_doc/demo/doc"
RA.save(SAVE_PATH)
# RA = RetrievalAugmentation(tree=SAVE_PATH)

# question = "How did Cinderella reach her happy ending?"
# answer = RA.answer_question(question=question)
# print("Answer: ", answer)
