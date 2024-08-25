import os
import threading
from tqdm import tqdm  # 추가
from dotenv import load_dotenv

load_dotenv() 
api_token = os.getenv("api_token")
os.environ["OPENAI_API_KEY"] = str(os.getenv("OPENAI_API_KEY"))

from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from raptor import BaseEmbeddingModel

class CustomEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name='google/gemma-2-9b-it'):
        try:
            print("Starting model load...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=api_token)
            print("Tokenizer loaded successfully")
            
            self.model = AutoModelForCausalLM.from_pretrained(model_name, token=api_token)
            print("Model loaded successfully")
        
        except Exception as e:
            print(f"Error occurred: {e}")

    # def create_embedding(self, documents):
    #     # GPU가 사용 가능한 경우 GPU로 모델을 이동
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     self.model.to(device)

    #     embeddings = []
    #     for doc in documents:
    #     # 입력 데이터를 GPU로 이동
    #         inputs = self.tokenizer(doc, return_tensors='pt', truncation=True, padding=True, max_length=512)
    #         inputs = {key: value.to(device) for key, value in inputs.items()}

    #         with torch.no_grad():
    #             outputs = self.model(**inputs)
            
    #         # 출력을 CPU로 이동하고 numpy로 변환
    #         # GPT 모델인 경우 logits 사용
    #         if hasattr(outputs, 'logits'):
    #             embedding = outputs.logits.mean(dim=1).squeeze().cpu().numpy()
    #         # 다른 모델인 경우 last_hidden_state 사용
    #         elif hasattr(outputs, 'last_hidden_state'):
    #             embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    #         else:
    #             raise AttributeError("The model output does not have 'logits' or 'last_hidden_state' attributes.")
    #         embeddings.append(embedding)
    
    #     return embeddings

    def create_embedding(self, documents):
        # CPU에서 모델을 사용하도록 설정
        device = torch.device('cpu')
        self.model.to(device)  # 모델을 CPU로 이동

        embeddings = []
        for doc in documents:
            # 입력 데이터를 CPU로 이동 (이미 CPU에 있으므로 필요하지 않음)
            inputs = self.tokenizer(doc, return_tensors='pt', truncation=True, padding=True, max_length=512)
            # 입력 데이터를 명시적으로 CPU로 이동
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                 outputs = self.model(**inputs)

             # 출력을 CPU에서 처리하고 numpy로 변환
            if hasattr(outputs, 'logits'):
                embedding = outputs.logits.mean(dim=1).squeeze().numpy()
            elif hasattr(outputs, 'last_hidden_state'):
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            else:
                raise AttributeError("The model output does not have 'logits' or 'last_hidden_state' attributes.")
            embeddings.append(embedding)
        return embeddings
        
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

SAVE_PATH = "C:/cluster/raptor/demo"
RA.save(SAVE_PATH)
# RA = RetrievalAugmentation(tree=SAVE_PATH)

# question = "How did Cinderella reach her happy ending?"
# answer = RA.answer_question(question=question)
# print("Answer: ", answer)
