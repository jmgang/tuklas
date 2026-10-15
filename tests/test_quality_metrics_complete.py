"""In-memory regressions; no cleanup or temporary directories."""
import unittest
import numpy as np
from quality_metrics_complete import aggregate,evaluate,METRICS,legacy_unweighted_encoder
from aws_model_shootout import encode_weighted_static


class CompleteQuality(unittest.TestCase):
    def fixture(self,blank=False):
        chunks=[{'parent_id':str(i),'text':'abcd','start':0,'end':4} for i in range(10)]
        query={'query_id':'q','language':'tl','partition':'evaluation','gold_parent_id':'2',
               'answer_spans':[] if blank else [{'start':0,'end':2}]}
        return chunks,query

    def test_complete_fields_and_replay(self):
        chunks,q=self.fixture()
        D=np.eye(10,dtype=np.float32); Q=np.arange(10,0,-1,dtype=np.float32)[None,:]
        summary,rows=evaluate(D,Q,[q],list(range(10)),[str(i) for i in range(10)],chunks)
        self.assertTrue(set(METRICS)<=summary.keys())
        self.assertEqual(len(rows[0]['ranking']),10)
        self.assertEqual(summary['Recall@1'],0)
        self.assertEqual(summary['Recall@3'],1)
        self.assertEqual(summary['MRR@10'],1/3)
        self.assertEqual(summary['nDCG@10'],0.5)
        self.assertEqual(summary['AnswerSpanCoverage@3'],1)
        self.assertEqual(summary,aggregate(rows))

    def test_blank_answer_denominator(self):
        chunks,q=self.fixture(True)
        s,_=evaluate(np.eye(10),np.ones((1,10)),[q],list(range(10)),[str(i) for i in range(10)],chunks)
        self.assertIsNone(s['AnswerSpanCoverage@10'])
        self.assertEqual(s['denominators']['AnswerSpanCoverage@10'],0)
        self.assertEqual(s['denominators']['Recall@10'],1)

    def test_missing_metric_rejected(self):
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            aggregate([{'query_id':'q','metrics':{'Recall@10':1}}])

    def test_duplicate_query_rejected(self):
        row={'query_id':'q','metrics':dict.fromkeys(METRICS,1.)}
        with self.assertRaisesRegex(ValueError,'duplicate'):aggregate([row,row])

    def test_alignment_rejected(self):
        chunks,q=self.fixture()
        with self.assertRaisesRegex(ValueError,'alignment'):
            evaluate(np.eye(10),np.ones((2,10)),[q],list(range(10)),[str(i) for i in range(10)],chunks)

    def test_legacy_pad_lookup_once_for_all_batches(self):
        class Encoding:
            ids=[0,1,2]
        class Tokenizer:
            def encode_batch_fast(self,values,add_special_tokens=False):
                return [Encoding() for _ in values]
        class Model:
            tokenizer=Tokenizer()
            unk_token_id=2
            embedding=np.asarray([[99,99],[3,4],[88,88]],dtype=np.float32)
        calls=[]
        def pad_lookup(tokenizer):
            calls.append(tokenizer);return 0
        encoder=legacy_unweighted_encoder(Model(),pad_lookup)
        result=encoder(['a']*2050)
        self.assertEqual(len(calls),1)
        self.assertEqual(result.shape,(2050,2))
        np.testing.assert_allclose(result[0],[0.6,0.8])

    def test_historical_shootout_honors_shipped_token_weights(self):
        from model2vec import StaticModel
        from tokenizers import Tokenizer, models, pre_tokenizers
        tokenizer=Tokenizer(models.WordLevel({'[UNK]':0,'a':1,'b':2},unk_token='[UNK]'))
        tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
        model=StaticModel(np.asarray([[0,0],[1,0],[0,1]],dtype=np.float32),
            tokenizer,weights=np.asarray([1,1,3],dtype=np.float32))
        actual=encode_weighted_static(model,['a b'])
        np.testing.assert_allclose(actual[0],[1/np.sqrt(10),3/np.sqrt(10)])
        self.assertGreater(actual[0,1],actual[0,0])


if __name__=='__main__':unittest.main()
