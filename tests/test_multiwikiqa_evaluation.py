import unittest
import math
import numpy as np
from pycon_model2vec_demo.multiwikiqa_evaluation import rank_parents,score,paired_cluster_ci,summarize


class EvaluationTests(unittest.TestCase):
    def test_unique_parents_max_not_average_and_selected_span(self):
        ranked=rank_parents(np.array([[1,0]],dtype=np.float32),
            np.array([[1,0],[0,1],[.8,.6]],dtype=np.float32),[0,0,1],["a","b"])[0]
        self.assertEqual([x[0] for x in ranked],["a","b"])
        self.assertEqual(ranked[0][1],0)
        q={"gold_parent_id":"a","answer_spans":[{"start":4,"end":5}]}
        chunks=[{"parent_id":"a","start":0,"end":3},
                {"parent_id":"a","start":3,"end":6},
                {"parent_id":"b","start":0,"end":9}]
        m=score(q,ranked,chunks)
        self.assertEqual(m["Recall@1"],1)
        self.assertEqual(m["AnswerSpanCoverage@10"],0)

    def test_rank_metrics_and_empty_answer_not_dropped(self):
        chunks=[{"parent_id":x,"start":0,"end":10} for x in ["a","b"]]
        q={"gold_parent_id":"b","answer_spans":[]}
        m=score(q,[("a",0,1),("b",1,.5)],chunks)
        self.assertEqual(m["Recall@1"],0)
        self.assertEqual(m["Recall@10"],1)
        self.assertEqual(m["MRR@10"],.5)
        self.assertAlmostEqual(m["nDCG@10"],1/math.log2(3))
        self.assertIsNone(m["AnswerSpanCoverage@10"])

    def test_zero_vectors_ties_stable_and_duplicate_rejected(self):
        a=rank_parents(np.zeros((1,2),np.float32),np.eye(2,dtype=np.float32),[0,1],["b","a"])[0]
        self.assertEqual([r[0] for r in a],["a","b"])
        with self.assertRaises(ValueError):
            rank_parents(np.zeros((1,2),np.float32),np.eye(2,dtype=np.float32),[0,1],["a","a"])

    def test_paired_cluster_identical_is_zero_and_mismatch_rejected(self):
        rows=[{"query_id":str(i),"gold_parent_id":str(i//2),"metrics":{"Recall@10":float(i%2)}} for i in range(8)]
        self.assertEqual(paired_cluster_ci(rows,rows,"Recall@10",100)["ci95"],[0,0])
        with self.assertRaises(ValueError):
            paired_cluster_ci(rows,list(reversed(rows)),"Recall@10",100)

    def test_coverage_denominator_not_ten_empty_answers(self):
        rows=[{"query_id":str(i),"gold_parent_id":"a","language":"en","partition":"evaluation",
               "exclusion_reasons":[] if i else ["empty-answer-reference"],
               "metrics":{"Recall@10":1,"AnswerSpanCoverage@10":1 if i else None}} for i in range(2)]
        report=summarize(rows)["all_10000"]
        self.assertEqual(report["metrics"]["Recall@10"]["denominator"],2)
        self.assertEqual(report["metrics"]["AnswerSpanCoverage@10"]["denominator"],1)


if __name__=="__main__":
    unittest.main()
