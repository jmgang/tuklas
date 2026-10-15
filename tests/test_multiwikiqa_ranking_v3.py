"""Guards at the real ranking recipe and checkpoint-selection seams."""
import unittest

from pycon_model2vec_demo import multiwikiqa_ranking_v2 as old
from pycon_model2vec_demo import multiwikiqa_ranking_v3 as new
from pycon_model2vec_demo.multiwikiqa_ranking_v3 import select_checkpoint, validate_recipe


def row(epoch, overall, en, tl, en_recall, tl_recall):
    return {'epoch':epoch,'nDCG@10':overall,
            'by_language':{'en':{'nDCG@10':en,'Recall@10':en_recall},
                           'tl':{'nDCG@10':tl,'Recall@10':tl_recall}}}


class RankingRepairGuards(unittest.TestCase):
    def test_rejects_original_regressing_learning_rate(self):
        with self.assertRaisesRegex(ValueError,'learning rate'):
            validate_recipe({'learning_rate':0.05,'epochs':3,'matryoshka_dimensions':[32,64,128,256,512,1024]})

    def test_rejects_probe_rate_that_still_lost_quality(self):
        with self.assertRaisesRegex(ValueError,'learning rate'):
            validate_recipe({'learning_rate':0.005,'epochs':3,'matryoshka_dimensions':[32,64,128,256,512,1024]})

    def test_accepts_measured_noncollapsing_rate(self):
        validate_recipe({'learning_rate':0.0005,'epochs':3,'matryoshka_dimensions':[32,64,128,256,512,1024]})

    def test_selects_baseline_if_combined_gain_hides_tagalog_loss(self):
        history=[row(0,.5,.6,.4,.7,.5),row(1,.55,.7,.39,.8,.49)]
        self.assertEqual(select_checkpoint(history),0)

    def test_selects_later_epoch_only_when_both_languages_hold(self):
        history=[row(0,.5,.6,.4,.7,.5),row(1,.51,.61,.41,.71,.51),
                 row(2,.54,.64,.44,.72,.52)]
        self.assertEqual(select_checkpoint(history),2)

    def test_training_pair_and_batch_logic_unchanged(self):
        queries=[];chunks=[]
        for i in range(30):
            q={'query_id':str(i),'partition':'development' if i<24 else 'evaluation',
               'component_id':str(i//2),'gold_parent_id':'p'+str(i//2),
               'question':f'What is item {i}?','answer_spans':[{'start':0,'end':4}],
               'language':'en' if i%2 else 'tl'}
            queries.append(q)
            if i%2==0:
                chunks.append({'parent_id':'p'+str(i//2),'chunk_id':str(i),
                               'start':0,'end':20,'text':'item fact sentence'})
        self.assertEqual(old.prepare(queries,chunks),new.prepare(queries,chunks))
        train,_,_=old.prepare(queries,chunks)
        self.assertEqual(list(old.batches(train,20)),list(new.batches(train,20)))


if __name__=='__main__':
    unittest.main()
