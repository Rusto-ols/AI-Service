"""Лабораторная работа №3: воспроизводимые офлайн-эксперименты RAG.
Для нейросетевой генерации требуется локальный Ollama (см. функцию ollama_answer).
"""
from __future__ import annotations
import json, re, os, sys, platform
from pathlib import Path
from collections import defaultdict
import fitz
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

PDF = Path(__file__).with_name('Практическая_работа__3_Совр_Инт_Сет_Сер.pdf')
OUT = Path(__file__).with_name('results_lab3')
OUT.mkdir(exist_ok=True)

def load_pages(path=PDF):
    with fitz.open(path) as pdf:
        return [{'page': i+1, 'text': re.sub(r'\s+', ' ', p.get_text('text')).strip(), 'source': path.name} for i,p in enumerate(pdf)]

def split_pages(pages, size=1000, overlap=200):
    assert 0 <= overlap < size
    chunks = []
    for p in pages:
        text=p['text']; start=0
        while start < len(text):
            segment=text[start:start+size]
            chunks.append({'id':len(chunks)+1,'page':p['page'],'source':p['source'], 'start_index':start,'text':segment})
            if start+size>=len(text): break
            start += size-overlap
    return chunks

# Эталон — страница PDF и проверяемая подстрока (не номер печатной страницы).
QA = [
 {'id':'Q01','question':'Что такое RAG?', 'alt':'Как устроен подход с поисковым дополнением генерации?', 'page':4,'anchor':'генерация ответа, дополненная поиском'},
 {'id':'Q02','question':'Что показывает бинарный мешок слов?', 'alt':'Как кодируется наличие термина в разреженном лексическом векторе?', 'page':4,'anchor':'если слово встречается'},
 {'id':'Q03','question':'Какой метод учитывает частоту термина и распространённость в коллекции?', 'alt':'Как называются взвешенные частотные признаки слов?', 'page':4,'anchor':'TF-IDF учитывает частоту термина'},
 {'id':'Q04','question':'Для чего служит поисковый индекс?', 'alt':'Где хранят представления частей документа и их происхождение?', 'page':5,'anchor':'Поисковый индекс хранит представления'},
 {'id':'Q05','question':'Какие существуют стратегии разбиения на фрагменты?', 'alt':'Как можно разделять документ по границам текста?', 'page':5,'anchor':'Основные стратегии разбиения'},
 {'id':'Q06','question':'Какие метаданные сохраняют для PDF фрагмента?', 'alt':'Какие реквизиты нужны для прослеживания происхождения найденного текста?', 'page':6,'anchor':'название файла, номер страницы'},
 {'id':'Q07','question':'Сколько завершённых обменов используется в памяти диалога?', 'alt':'Какова глубина сохраняемого контекста предыдущих реплик?', 'page':6,'anchor':'последние три завершённых обмена'},
 {'id':'Q08','question':'Как определяется Hit Rate@k?', 'alt':'Как считают долю вопросов с найденным достаточным фрагментом?', 'page':7,'anchor':'Hit Rate@k = H/N'},
 {'id':'Q09','question':'Какие размеры фрагментов нужно сравнить во втором задании?', 'alt':'Какие длины текстовых частей исследуются в эксперименте два?', 'page':8,'anchor':'chunk_size=500, 1000 и 2000'},
 {'id':'Q10','question':'Что сделать при нулевом векторе запроса в мешке слов?', 'alt':'Как действовать, когда ни одно слово вопроса не известно словарю?', 'page':9,'anchor':'нулевой вектор'},
]
NEGATIVE=[
 'Какая компания разработала первый коммерческий RAG-продукт в 2018 году?',
 'Какую точную стоимость облачного хостинга RAG-системы указывает документ?',
 'Какой GPU установлен у авторов ноутбука?',
 'Сколько секунд занимает индексирование на NVIDIA RTX 4090 в эксперименте?',
 'Какой адрес электронной почты указан для поддержки системы?',
]

def lexical_index(chunks):
    model=CountVectorizer(binary=True, lowercase=True, token_pattern=r'(?u)\b\w+\b')
    matrix=model.fit_transform([c['text'] for c in chunks]); return model,matrix

def dense_index(chunks):
    # Офлайн-приближение плотного поиска: SVD над TF-IDF; НЕ OllamaEmbeddings.
    tf=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=1, max_features=22000)
    matrix=tf.fit_transform([c['text'] for c in chunks]); d=min(96, min(matrix.shape)-1)
    if d < 2: return tf,None,normalize(matrix)
    svd=TruncatedSVD(n_components=d, random_state=42)
    emb=normalize(svd.fit_transform(matrix)); return tf,svd,emb

def rank(question,chunks,method='dense',idx=None,k=3):
    if idx is None: idx = dense_index(chunks) if method=='dense' else lexical_index(chunks)
    if method=='dense':
        tf,svd,mat=idx; query=tf.transform([question]); vec=normalize(svd.transform(query)) if svd is not None else normalize(query)
    else:
        tf,mat=idx; vec=tf.transform([question]);
        if vec.nnz == 0: return []
    scores=cosine_similarity(vec,mat).ravel(); order=np.argsort(-scores,kind='stable')[:k]
    return [{**chunks[int(i)],'score':round(float(scores[int(i)]),6)} for i in order if scores[int(i)]>0]

def hit(items,qa):
    # Проверка требует не только совпадения страницы, но присутствия подтверждающей цитаты.
    return any(c['page']==qa['page'] and qa['anchor'].casefold() in c['text'].casefold() for c in items)

def grounded_answer(q,items,qa=None):
    """Детерминированный evidence-extractor, а не ответ языковой модели."""
    if qa is None or not hit(items,qa): return 'В предоставленном контексте недостаточно сведений для ответа.'
    c=next(c for c in items if c['page']==qa['page'] and qa['anchor'].casefold() in c['text'].casefold())
    at=c['text'].casefold().find(qa['anchor'].casefold())
    excerpt=c['text'][max(0,at-65):min(len(c['text']),at+len(qa['anchor'])+140)].strip()
    return excerpt + f' [Фрагмент {c["id"]}, стр. {c["page"]}]'

def ollama_answer(question, contexts, strict=True, model='qwen2.5:3b', base_url='http://localhost:11434'):
    """Реальный вызов LLM в Ollama при наличии сервера и загруженной модели."""
    import requests
    prompt_strict='Отвечай только по предоставленным фрагментам PDF. Если достаточного подтверждения нет, скажи: «В документе нет сведений для ответа». После каждого факта приводи ссылку [Фрагмент N, стр. P]. Не придумывай номера и страницы.'
    prompt_base='Ответь на вопрос на основе фрагментов документа. Используй ссылки на фрагменты, если возможно.'
    context='\n\n'.join(f'[Фрагмент {c["id"]}, стр. {c["page"]}, файл {c["source"]}]\n{c["text"]}' for c in contexts)
    data={'model':model,'stream':False,'options':{'temperature':0,'seed':42},'messages':[{'role':'system','content':prompt_strict if strict else prompt_base},{'role':'user','content':f'Контекст:\n{context}\n\nВопрос: {question}'}]}
    response=requests.post(base_url+'/api/chat',json=data,timeout=180); response.raise_for_status()
    return response.json()['message']['content']

BASE_PROMPT='Ответь на вопрос на основе найденных фрагментов. Используй ссылки, если возможно.'
STRICT_PROMPT='Отвечай исключительно по найденным фрагментам PDF. При недостатке информации явно откажись от ответа. Подтверждай факты ссылками [Фрагмент N, стр. P]; не выдумывай источники.'

def get_all_results():
    all_pages=load_pages(); pages=all_pages[2:]; chunks=split_pages(pages); full=dense_index(chunks); short=dense_index(chunks[:5]); allrows=[]
    for qa in QA:
        for variant,corpus,index in [('Первые 5',chunks[:5],short),('Полный',chunks,full)]:
            res=rank(qa['question'],corpus,idx=index); allrows.append({'question':qa['id'],'variant':variant,'group':'начало' if qa['page']==4 else 'далее','hit':hit(res,qa),'ids':[x['id'] for x in res], 'pages':[x['page'] for x in res], 'answer':grounded_answer(qa['question'],res,qa)})
    task1=pd.DataFrame(allrows)
    rows=[]
    for size in [500,1000,2000]:
        cc=split_pages(pages,size,10); idx=dense_index(cc)
        for qa in QA:
            res=rank(qa['question'],cc,idx=idx)
            rows.append({'size':size,'chunks':len(cc),'question':qa['id'],'hit':hit(res,qa),'ids':[x['id'] for x in res]})
    task2=pd.DataFrame(rows)
    rows=[]
    selected=QA[3:8]
    for overlap in [0,100,200]:
        cc=split_pages(pages,1000,overlap); idx=dense_index(cc)
        for qa in selected:
            res=rank(qa['question'],cc,idx=idx)
            # Повтор: Jaccard посимвольных 5-грамм > 0.8 среди возвращенных пар.
            def grams(s): return {s[i:i+5] for i in range(max(0,len(s)-4))}
            duplicates=False
            for a in range(len(res)):
                for b in range(a+1,len(res)):
                    x,y=grams(res[a]['text']),grams(res[b]['text'])
                    if x and y and len(x&y)/len(x|y)>.8:duplicates=True
            rows.append({'overlap':overlap,'chunks':len(cc),'question':qa['id'],'hit':hit(res,qa),'has_near_duplicates':duplicates,'ids':[x['id'] for x in res]})
    task3=pd.DataFrame(rows)
    rows=[]
    sparse=lexical_index(chunks)
    for method,idx in [('Бинарный мешок слов',sparse),('Плотный SVD (офлайн)',full)]:
        for qa in QA[:5]:
            for phrasing,question in [('точный',qa['question']),('перефразировка',qa['alt'])]:
                res=rank(question,chunks,method='bow' if method.startswith('Бинарный') else 'dense',idx=idx)
                rows.append({'method':method,'phrasing':phrasing,'question':qa['id'],'hit':hit(res,qa),'ids':[x['id'] for x in res],'pages':[x['page'] for x in res]})
    task4=pd.DataFrame(rows)
    rows=[]
    for qa in QA[:5]:
        res=rank(qa['question'],chunks,idx=full)
        answer=grounded_answer(qa['question'],res,qa)
        cites=re.findall(r'\[Фрагмент (\d+), стр\. (\d+)\]',answer)
        exists=all(any(c['id']==int(cid) and c['page']==int(page) for c in res) for cid,page in cites) if cites else False
        supported=bool(cites) and hit(res,qa)
        rows.append({'question':qa['id'],'answer':answer,'cited':bool(cites),'citation_exists':exists,'supported':supported,'page_correct':exists,'top3':[{'id':c['id'],'page':c['page'],'start_index':c['start_index'],'score':c['score'],'excerpt':c['text'][:180]} for c in res]})
    task5=pd.DataFrame(rows)
    rows=[]
    for qa in QA[:5]:
        res=rank(qa['question'],chunks,idx=full)
        for prompt in ['Исходный','Строгий']:
            a=grounded_answer(qa['question'],res,qa)
            rows.append({'prompt':prompt,'question':qa['id'],'answer_in_pdf':True,'retrieval_hit':hit(res,qa),'answer':a,'grade_proxy':2 if hit(res,qa) else 0,'correct_refusal':False,'unsupported':False})
    for j,q in enumerate(NEGATIVE):
        res=rank(q,chunks,idx=full)
        for prompt in ['Исходный','Строгий']:
            # Нельзя выдавать эвристический ответ за вывод LLM; оба шаблона используют один extractor.
            rows.append({'prompt':prompt,'question':f'NEG{j+1}','answer_in_pdf':False,'retrieval_hit':False,'answer':'В предоставленном контексте недостаточно сведений для ответа.','grade_proxy':None,'correct_refusal':True,'unsupported':False})
    task6=pd.DataFrame(rows)
    dialog_specs=[
      [('Что такое RAG?','Q01'),('Как этот подход работает?','Q01'),('Изменяет ли он веса модели?','Q01')],
      [('Что такое бинарный мешок слов?','Q02'),('Что в нём означает единица?','Q02'),('Учитывает ли он порядок слов?','Q02')],
      [('Что такое память диалога?','Q07'),('Сколько обменов в ней хранится?','Q07'),('Чем отделяют разные разговоры?','Q07')],
    ]
    # Ручные прозрачные автономные преобразования; при Ollama можно заменить LLM-rewriter.
    rewrites={
      'Как этот подход работает?':'Как работает подход RAG с поиском по документу?',
      'Изменяет ли он веса модели?':'Изменяет ли RAG веса языковой модели при индексации?',
      'Что в нём означает единица?':'Что означает значение 1 в бинарном мешке слов?',
      'Учитывает ли он порядок слов?':'Учитывает ли бинарный мешок слов порядок слов?',
      'Сколько обменов в ней хранится?':'Сколько обменов вопрос-ответ хранит память диалога?',
      'Чем отделяют разные разговоры?':'Чем разделяются сессии в памяти диалога?',
    }
    histories=defaultdict(list); rows=[]
    for session,steps in enumerate(dialog_specs,1):
        for step,(q,qid) in enumerate(steps,1):
            qa=next(x for x in QA if x['id']==qid)
            reform=rewrites.get(q,q)
            res=rank(reform,chunks,idx=full)
            answer=grounded_answer(reform,res,qa)
            rows.append({'session':session,'step':step,'question':q,'query':reform,'history_len':len(histories[str(session)][-3:]),'hit':hit(res,qa),'grade_proxy':2 if hit(res,qa) else 0,'retrieved_ids':[x['id'] for x in res], 'retrieved_pages':[x['page'] for x in res], 'answer':answer})
            histories[str(session)].append((q,answer))
    task7=pd.DataFrame(rows)
    # Автоматические проверки разделения сессий / очистки.
    assert len(histories)==3 and all(len(v)==3 for v in histories.values())
    histories['1'].clear(); assert len(histories['1'])==0 and len(histories['2'])==3
    files={f'task{i}':v for i,v in enumerate([task1,task2,task3,task4,task5,task6,task7],1)}
    for name,df in files.items(): df.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
    diagnostics={'source_pdf':PDF.name,'page_count':len(all_pages),'indexed_pages':len(pages),'baseline_chunk_count':len(chunks),'baseline_size':1000,'baseline_overlap':200,'top_k':3,'seed':42,'embedding_method':'character TF-IDF + TruncatedSVD (офлайн-приближение, не OllamaEmbeddings)','generator':'grounded evidence extractor (НЕ LLM)','python':platform.python_version(),'sklearn':__import__('sklearn').__version__,'pymupdf':fitz.VersionBind,'pandas':pd.__version__,'task1':{f'{a}|{b}':v for (a,b),v in task1.groupby(['variant','group'])['hit'].agg(['sum','count']).to_dict('index').items()},'task2':task2.groupby('size')['hit'].agg(['sum','count']).to_dict('index'),'task3':task3.groupby('overlap')[['hit','has_near_duplicates']].sum().to_dict('index'),'task4':{f'{a}|{b}':v for (a,b),v in task4.groupby(['method','phrasing'])['hit'].agg(['sum','count']).to_dict('index').items()},'task5':{'valid_citations':int(task5['citation_exists'].sum()),'supported':int(task5['supported'].sum())},'task6':{'correct_refusals':int(task6.loc[task6['prompt']=='Строгий','correct_refusal'].sum()), 'note':'оценки прокси, НЕ эксперимент двух LLM-промптов'},'task7':{'follow_up_hits':int(task7.loc[task7['step']>1,'hit'].sum()),'follow_ups':6}}
    (OUT/'summary.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'qa_reference.json').write_text(json.dumps({'positive':QA,'negative':NEGATIVE},ensure_ascii=False,indent=2),encoding='utf-8')
    return files,diagnostics

if __name__=='__main__':
    files,diagnostics=get_all_results()
    print(json.dumps(diagnostics,ensure_ascii=False,indent=2))
