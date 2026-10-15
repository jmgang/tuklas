import unittest
import numpy as np
from pycon_model2vec_demo.sealion_weight_audit import compare_block


class WeightAuditTests(unittest.TestCase):
    def test_export_scaling_alone_is_not_learning(self):
        initial=np.array([[2,1],[1,2],[4,3]],dtype=np.float32)
        student=initial*.5
        student[0]=0
        result=compare_block(initial,student,0,0)
        self.assertEqual(result["changed_beyond_fixed_export_scaling"],0)
        self.assertEqual(result["direction_changed_cosine_below_0_99999"],0)

    def test_real_direction_change_is_detected(self):
        initial=np.array([[1,2],[3,1]],dtype=np.float32)
        student=initial*.5
        student[1,0]+=1
        result=compare_block(initial,student,5,0)
        self.assertEqual(result["changed_beyond_fixed_export_scaling"],1)
        self.assertEqual(result["direction_changed_cosine_below_0_99999"],1)
