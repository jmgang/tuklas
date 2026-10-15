import unittest
import numpy as np
from pycon_model2vec_demo.segmented_encoding import SegmentPolicy, pool
from pycon_model2vec_demo.bedrock_embeddings import request_body


class FullContentTests(unittest.TestCase):
    def test_reconstruct_and_prefix_budget(self):
        policy = SegmentPolicy(len, len, 12, 12, 'p: ')
        text = 'hello  world\n日本語 another word ' * 4
        plan = policy.plan(text)
        self.assertEqual(''.join(text[a:b] for a,b,_ in plan), text)
        self.assertTrue(all(len('p: '+text[a:b]) <= 12 for a,b,_ in plan))

    def test_single_segment_identity(self):
        self.assertEqual(SegmentPolicy(len,len,10).plan('hello'), [[0,5,5]])
        np.testing.assert_allclose(pool(np.array([[.6,.8]]),[5]), [.6,.8])

    def test_weighted_pool(self):
        np.testing.assert_allclose(pool(np.eye(2),[3,4]), [.6,.8])

    def test_strict_provider_refuses_loss(self):
        settings = dict(model_id='cohere.embed-english-v3',dimension=1024,truncate='NONE',maximum_input_characters=2048,full_content=True)
        with self.assertRaises(ValueError):
            request_body(['x'*2049],'search_document',settings)
        self.assertEqual(request_body(['x'*2048],'search_document',settings)['texts'],['x'*2048])
        settings['truncate']='END'
        with self.assertRaises(ValueError):
            request_body(['hello'],'search_query',settings)

    def test_impossible_prefix(self):
        with self.assertRaises(ValueError):
            SegmentPolicy(len,len,2,2,'prefix').plan('a')

    def test_empty_and_nonfinite(self):
        with self.assertRaises(ValueError):
            SegmentPolicy(len,len,2).plan('')
        with self.assertRaises(ValueError):
            pool(np.array([[float('nan')]]),[1])
