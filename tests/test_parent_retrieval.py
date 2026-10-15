import unittest
import numpy as np
from pycon_model2vec_demo.parent_retrieval import shared_slices,parent_top_k
from pycon_model2vec_demo.segmented_encoding import SegmentPolicy
from pycon_model2vec_demo.grade3 import Grade3Query,score_top10_query


class ParentRetrievalTests(unittest.TestCase):
    def test_shared_reconstruction_and_all_limits(self):
        policies=[SegmentPolicy(len,len,16,None,'p: '),SegmentPolicy(len,len,10,10)]
        text='hello world 日本語\n next sentence '*6
        pieces=shared_slices(text,policies)
        self.assertEqual(''.join(text[a:b] for a,b in pieces),text)
        self.assertTrue(all(p.fits(text[a:b]) for a,b in pieces for p in policies))

    def test_duplicate_segments_do_not_take_parent_slots(self):
        vectors=np.array([[1.,0.],[.99,.01],[.9,.1],[.8,.2]],dtype=np.float32)
        rows=parent_top_k(np.array([[1.,0.]]),vectors,[0,0,1,2],['a','b','c'],k=3,block_size=1)[0]
        self.assertEqual([r.corpus_id for r in rows],['a','b','c'])

    def test_max_not_average(self):
        rows=parent_top_k(np.array([[1.,0.]]),np.array([[1.,0.],[-1.,0.],[.8,.6]]),[0,0,1],['a','b'],k=2)[0]
        self.assertEqual(rows[0].corpus_id,'a')
        self.assertEqual(rows[0].similarity,1.)

    def test_bruteforce_and_block_boundaries(self):
        rng=np.random.default_rng(4)
        owners=np.repeat(np.arange(13),rng.integers(1,7,13))
        vectors=rng.normal(size=(len(owners),7)).astype('float32')
        queries=rng.normal(size=(5,7)).astype('float32')
        ids=[f'p{i:02d}' for i in range(13)]
        actual=parent_top_k(queries,vectors,owners,ids,k=10,block_size=3)
        for q,ranking in zip(queries,actual):
            scores=[float(np.max(vectors[owners==i]@q)) for i in range(13)]
            expected=sorted(range(13),key=lambda i:(-scores[i],ids[i]))[:10]
            self.assertEqual([r.corpus_id for r in ranking],[ids[i] for i in expected])

    def test_gold_parent_counted_once_at_ten(self):
        ids=[f'p{i}' for i in range(10)]
        owners=[0,0]+list(range(1,10))
        vectors=np.array([[1-i*.01,0] for i in range(11)],dtype=np.float32)
        ranking=parent_top_k(np.array([[1.,0.]]),vectors,owners,ids,k=10)[0]
        query=Grade3Query('q','question','category',1,frozenset({'p0'}),frozenset(ids))
        result=score_top10_query(query,ranking)
        self.assertEqual(result['recall_at_1'],1)
        self.assertEqual(result['recall_at_10'],1)

    def test_invalid_owners(self):
        with self.assertRaises(ValueError):
            parent_top_k(np.ones((1,2)),np.ones((2,2)),[1,0],['a','b'])
