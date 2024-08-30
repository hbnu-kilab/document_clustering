# 필요한 모듈과 클래스 임포트
from raptor import RetrievalAugmentation
from openai import OpenAI
import os
from dotenv import load_dotenv

# 환경 변수에서 API 키 가져오기
load_dotenv(dotenv_path='apikey.env')
api_key = os.getenv("OPENAI_API_KEY")

# 트리 저장 경로
SAVE_PATH = 'C:/Users/USER/Desktop/clustering_doc/demo/doc'  # 실제 트리 파일 경로로 변경

# RetrievalAugmentation을 사용하여 트리 불러오기
RA = RetrievalAugmentation(tree=SAVE_PATH)

# 트리 객체 가져오기
tree = RA.tree

# 최상위 노드(root nodes) 확인 및 모든 속성 출력
root_nodes = tree.root_nodes  # 트리 객체의 root_nodes 속성 접근

# 최상위 노드의 모든 속성 출력
for root in root_nodes.values():
    print(f"Node ID: {root.index}")
    print(f"Text: {root.text}")
    print(f"Children: {root.children}")
    print(f"Embeddings: {root.embeddings}")
    print("-" * 40)

