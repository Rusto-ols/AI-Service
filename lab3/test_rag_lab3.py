import unittest
from rag_lab3 import load_pages,split_pages,QA,rank,dense_index,lexical_index,hit,grounded_answer,get_all_results

class TestRAGLab3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages=load_pages()[2:]
        cls.chunks=split_pages(cls.pages,1000,200)
        cls.dense=dense_index(cls.chunks)
    def test_pages(self):
        self.assertGreater(len(self.pages),5)
    def test_chunks(self):
        self.assertGreater(len(self.chunks),5)
    def test_metadata(self):
        self.assertTrue(all({'id','page','source','start_index','text'}<=set(x) for x in self.chunks))
    def test_overlap(self):
        x=split_pages([{'page':1,'source':'t.pdf','text':'a'*1500}],1000,200)
        self.assertEqual(x[1]['start_index'],800)
    def test_search(self):
        self.assertLessEqual(len(rank(QA[0]['question'],self.chunks,idx=self.dense)),3)
    def test_oov(self):
        self.assertEqual(rank('zzxyzqnonwordzz',self.chunks,method='bow',idx=lexical_index(self.chunks)),[])
    def test_citation(self):
        qa=QA[0]; items=[c for c in self.chunks if c['page']==qa['page'] and qa['anchor'].casefold() in c['text'].casefold()]
        self.assertTrue(items)
        self.assertIn('[Фрагмент',grounded_answer(qa['question'],items,qa))
    def test_outputs(self):
        tables,_=get_all_results()
        self.assertEqual(set(tables),{f'task{i}' for i in range(1,8)})
        self.assertEqual(len(tables['task2']),30)
        self.assertEqual(len(tables['task7']),9)
if __name__=='__main__': unittest.main()
