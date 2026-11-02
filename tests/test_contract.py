import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from enterprise_audit.study import load_data,valid_rows,project,process_pool,Generator
ROOT=Path(__file__).resolve().parents[1]
class ContractTests(unittest.TestCase):
 def test_projection_preserves_labels_and_other_fields(self):
  for name,dependent in [('adult','education_num'),('bank','previous'),('wine','total sulfur dioxide')]:
   df,cats,_=load_data(ROOT,name);pool=Generator(df.iloc[:400],cats,'marginal').sample(500,42)
   fixed=project(pool,name)
   self.assertTrue(valid_rows(fixed,name).all())
   pd.testing.assert_frame_equal(fixed.drop(columns=dependent),pool.drop(columns=dependent))
 def test_quota_budget_and_no_invalid_backfill(self):
  pool=pd.DataFrame({'pdays':[-1,10,-1,5,5,-1],'previous':[0,0,0,1,1,1],'target':[0,1,0,1,1,1]})
  chosen,n=process_pool(pool,'bank','quota',4,.5)
  self.assertEqual(n,5);self.assertEqual(len(chosen),4);self.assertEqual(chosen.target.sum(),2)
  depleted,n=process_pool(pool.iloc[:2],'bank','reject',4,.5)
  self.assertEqual(n,2);self.assertEqual(len(depleted),1);self.assertTrue(valid_rows(depleted,'bank').all())
 def test_seed_and_categories(self):
  df,cats,_=load_data(ROOT,'bank');g=Generator(df.iloc[:400],cats,'conditional_copula')
  a=g.sample(100,7);b=g.sample(100,7);pd.testing.assert_frame_equal(a,b)
  for c in cats:self.assertTrue(set(a[c])<=set(df.iloc[:400][c]))
if __name__=='__main__':unittest.main()
