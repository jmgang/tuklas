"""In-memory leakage, batch-negative and official ranking-loss checks."""
import unittest
from pycon_model2vec_demo.multiwikiqa_ranking import prepare,batches,key


class RankingGuards(unittest.TestCase):
    def fixture(self):
        q=[];chunks=[]
        for i in range(100):
            parent=f'p{i}'
            q.append({'query_id':str(i),'partition':'development' if i<80 else 'evaluation',
                'component_id':str(i),'gold_parent_id':parent,'question':f'Question {i}',
                'language':'tl' if i%2 else 'en','answer_spans':[{'start':2,'end':4}]})
            chunks.append({'parent_id':parent,'chunk_id':parent+':0:10','start':0,'end':10,'text':'abcdefghij'})
        return q,chunks

    def test_heldout_is_never_training(self):
        q,c=self.fixture();train,val,_=prepare(q,c)
        self.assertFalse({x['query_id'] for x in train}&{str(i) for i in range(80,100)})
        self.assertFalse({x['component_id'] for x in train}&{x['component_id'] for x in val})

    def test_overlap_skipped(self):
        q,c=self.fixture();train,_,_=prepare(q,c)
        target=train[0]['query_id'];q[80]['question']=q[int(target)]['question']
        train,_,skipped=prepare(q,c)
        self.assertNotIn(target,{x['query_id'] for x in train})
        self.assertIn('evaluation question overlap',{x['reason'] for x in skipped})

    def test_empty_and_noncontained_answers_skip(self):
        q,c=self.fixture();train,_,_=prepare(q,c);target=int(train[0]['query_id'])
        q[target]['answer_spans']=[]
        train,_,skipped=prepare(q,c)
        self.assertNotIn(str(target),{x['query_id'] for x in train})
        self.assertIn('no fully answer-bearing chunk',{x['reason'] for x in skipped})

    def test_batch_components_unique(self):
        q,c=self.fixture();train,_,_=prepare(q,c)
        copy={**train[0],'query_id':'copy','question':'A sibling question'}
        groups=list(batches(train+[copy],2048))
        for batch in groups:
            self.assertEqual(len(batch),len({r['component_id'] for r in batch}))
            self.assertEqual(len(batch),len({key(r['question']) for r in batch}))

    def test_unicode_query_overlap(self):
        self.assertEqual(key(' ＡＢＣ   X '),key('abc x'))

    def test_real_official_losses_backward(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.sentence_transformer.modules import StaticEmbedding
        from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss,MatryoshkaLoss
        tok=Tokenizer(WordLevel({'[UNK]':0,'hello':1,'world':2,'bansa':3,'Pilipinas':4},unk_token='[UNK]'))
        tok.pre_tokenizer=Whitespace()
        module=StaticEmbedding(tok,embedding_weights=torch.randn(5,1024))
        model=SentenceTransformer(modules=[module],device='cpu')
        criterion=MatryoshkaLoss(model,MultipleNegativesRankingLoss(model),[32,64,128,256,512,1024])
        before=module.embedding.weight.detach().clone()
        opt=torch.optim.AdamW(model.parameters(),lr=.05)
        loss=criterion([model.tokenize(['hello','bansa']),model.tokenize(['hello world','bansa Pilipinas'])],None)
        self.assertTrue(torch.isfinite(loss))
        loss.backward();opt.step()
        self.assertFalse(torch.equal(before,module.embedding.weight))


if __name__=='__main__':unittest.main()
