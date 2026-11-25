"""Реальные нейросетевые OllamaEmbeddings vs бинарный BoW (задание 4)."""
import os, json
from pathlib import Path
import numpy as np
import requests
from sklearn.preprocessing import normalize
from rag_lab3 import load_pages,split_pages,QA,lexical_index,rank,hit
MODEL=os.getenv('OLLAMA_EMBED_MODEL','nomic-embed-text')
URL=os.getenv('OLLAMA_URL','http://localhost:11434').rstrip('/')
chunks=split_pages(load_pages()[2:],1000,200)
def embed(texts):
    response=requests.post(URL+'/api/embed',json={'model':MODEL,'input':texts},timeout=600)
    response.raise_for_status()
    return normalize(np.array(response.json()['embeddings'],dtype=float))
mat=embed([c['text'] for c in chunks])
bow=lexical_index(chunks)
rows=[]
for qa in QA[:5]:
    for phrasing,query in [('точная',qa['question']),('перефразировка',qa['alt'])]:
        qv=embed([query])[0]
        similarities=mat@qv
        inds=np.argsort(-similarities)[:3]
        found=[chunks[int(i)] for i in inds if similarities[i]>0]
        lexical=rank(query,chunks,method='bow',idx=bow,k=3)
        for method,res in [(MODEL,found),('Бинарный мешок слов',lexical)]:
            rows.append({'method':method,'phrase':phrasing,'question':query,'question_id':qa['id'],'hit':hit(res,qa),'fragments':[{'id':x['id'],'page':x['page']} for x in res]})
out=Path(__file__).with_name('results_lab3')
out.mkdir(exist_ok=True)
(out/'ollama_embeddings_task4.json').write_text(json.dumps({'model':MODEL,'results':rows},ensure_ascii=False,indent=2),encoding='utf-8')
for method in (MODEL,'Бинарный мешок слов'):
    for phrase in ('точная','перефразировка'):
        matching=[r for r in rows if r['method']==method and r['phrase']==phrase]
        print(method,phrase,sum(r['hit'] for r in matching),'из',len(matching))
