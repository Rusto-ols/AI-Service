import os
from rag_lab3 import load_pages, split_pages, dense_index, rank, ollama_answer
pages=load_pages()[2:]
chunks=split_pages(pages,1000,200)
index=dense_index(chunks)
question='Что такое RAG?'
found=rank(question,chunks,idx=index,k=3)
print('Вопрос:',question)
print('Поисковый запрос:',question)
for item in found:
    print(f'Фрагмент {item["id"]}, стр. {item["page"]}: {item["text"][:300]}')
print('Ответ Ollama:',ollama_answer(question,found,model=os.getenv('OLLAMA_MODEL','qwen2.5:3b')))
