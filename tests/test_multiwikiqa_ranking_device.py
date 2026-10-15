"""Real accelerator regression at the export-parity forward call."""
import unittest
import torch
from pycon_model2vec_demo.multiwikiqa_ranking_v2 import normalized_training_embeddings


class DeviceRegression(unittest.TestCase):
    def model(self,device):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.sentence_transformer.modules import StaticEmbedding
        tok=Tokenizer(WordLevel({'[UNK]':0,'hello':1,'world':2,'bansa':3,'Pilipinas':4},unk_token='[UNK]'))
        tok.pre_tokenizer=Whitespace()
        module=StaticEmbedding(tok,embedding_weights=torch.randn(5,1024))
        return SentenceTransformer(modules=[module],device=device)

    def test_cpu_forward(self):
        result=normalized_training_embeddings(self.model('cpu'),['hello world','bansa Pilipinas'])
        self.assertEqual(result.shape,(2,1024))

    @unittest.skipUnless(torch.cuda.is_available() or torch.backends.mps.is_available(),'No accelerator available')
    def test_real_accelerator_parity_forward(self):
        device='cuda' if torch.cuda.is_available() else 'mps'
        model=self.model(device)
        # Real tokenization returns CPU indices; the shared call must move them.
        self.assertEqual(str(model.tokenize(['hello'])['input_ids'].device),'cpu')
        result=normalized_training_embeddings(model,['hello world','bansa Pilipinas'])
        self.assertEqual(result.shape,(2,1024))
        self.assertTrue(torch.isfinite(torch.from_numpy(result)).all())


if __name__=='__main__':unittest.main()
