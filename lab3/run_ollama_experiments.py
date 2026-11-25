"""Запускает живые генерации для заданий 5, 6 и 7 и сохраняет ответы без выдуманных оценок."""
import json, csv, os, re
from pathlib import Path
from rag_lab3 import QA, NEGATIVE, load_pages, split_pages, dense_index, rank, ollama_answer
model=os.getenv('OLLAMA_MODEL','qwen2.5:3b')
chunks=split_pages(load_pages()[2:],1000,200)
idx=dense_index(chunks)
rows=[]
for qa in QA[:5]:
    q=qa['question']; found=rank(q,chunks,idx=idx)
    for prompt in ('Исходный','Строгий'):
        ans=ollama_answer(q,found,strict=prompt=='Строгий',model=model)
        citation_list=[{'id':int(a),'page':int(b)} for a,b in re.findall(r'\[Фрагмент (\d+), стр\. (\d+)\]',ans)]
        exists=all(any(x['id']==c['id'] and x['page']==c['page'] for x in found) for c in citation_list)
        rows.append({'task':5 if prompt=='Строгий' else 6,'question':q,'prompt':prompt,'answer_in_pdf':True,'search_query':q,'chunks':[(x['id'],x['page']) for x in found],'answer':ans,'citations_exist':bool(citation_list) and exists})
for q in NEGATIVE:
    found=rank(q,chunks,idx=idx)
    for prompt in ('Исходный','Строгий'):
        ans=ollama_answer(q,found,strict=prompt=='Строгий',model=model)
        rows.append({'task':6,'question':q,'prompt':prompt,'answer_in_pdf':False,'search_query':q,'chunks':[(x['id'],x['page']) for x in found],'answer':ans,'citations_exist':None})
followups=[
 ('RAG', [('Что такое RAG?','Что такое RAG?'),('Как этот подход работает?','Как работает подход RAG с поиском по документу?'),('Изменяет ли он веса модели?','Изменяет ли RAG веса языковой модели при индексации?')]),
 ('мешок слов',[('Что такое бинарный мешок слов?','Что такое бинарный мешок слов?'),('Что в нём означает единица?','Что означает значение 1 в бинарном мешке слов?'),('Учитывает ли он порядок слов?','Учитывает ли бинарный мешок слов порядок слов?')]),
 ('память',[('Что такое память диалога?','Что такое память диалога?'),('Сколько обменов в ней хранится?','Сколько обменов вопрос-ответ хранит память диалога?'),('Чем отделяют разные разговоры?','Чем разделяются сессии в памяти диалога?')]),
]
for session, turns in followups:
    history=[]
    for q,query in turns:
        found=rank(query,chunks,idx=idx)
        ans=ollama_answer(q,found,model=model)
        rows.append({'task':7,'session':session,'question':q,'search_query':query,'history_length':min(3,len(history)),'chunks':[(x['id'],x['page']) for x in found],'answer':ans})
        history.append((q,ans))
out=Path(__file__).with_name('results_lab3');out.mkdir(exist_ok=True)
(out/'ollama_experiments.json').write_text(json.dumps({'model':model,'results':rows},ensure_ascii=False,indent=2),encoding='utf8')
with (out/'ollama_experiments.csv').open('w',newline='',encoding='utf-8-sig') as f:
    fields = [
        'task', 'session', 'question', 'prompt',
        'answer_in_pdf', 'search_query', 'chunks',
        'answer', 'citations_exist', 'history_length'
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f'Сохранено {len(rows)} запусков модели: {out}')
