#!/bin/bash

python -m eval.eval_consent 1
python -m eval.eval_consent 3
python -m eval.eval_sbert_zero 1
python -m eval.eval_sbert_zero 3

python -m eval.analysis.popularity_analysis_gen_pop_metric ../data/eval/consent_ndcg_3.txt >../data/eval/analysis/consent_ndcg_3_pop.txt
python -m eval.analysis.popularity_analysis_gen_pop_metric ../data/eval/sbert_zero_ndcg_3.txt >../data/eval/analysis/sbert_zero_ndcg_3_pop.txt

python -m eval.analysis.popularity_analysis_gen_pop_metric ../data/eval/consent_mrr_1.txt >../data/eval/analysis/consent_mrr_1_pop.txt
python -m eval.analysis.popularity_analysis_gen_pop_metric ../data/eval/sbert_zero_mrr_1.txt >../data/eval/analysis/sbert_zero_mrr_1_pop.txt

python -m eval.analysis.popularity_analysis_gen_pop_metric ../data/eval/consent_mrr_3.txt >../data/eval/analysis/consent_mrr_3_pop.txt
python -m eval.analysis.popularity_analysis_gen_pop_metric ../data/eval/sbert_zero_mrr_3.txt >../data/eval/analysis/sbert_zero_mrr_3_pop.txt

echo "Compare Our vs base (nDCG@3, P@1 and MRR@3) on all questions"
python eval/ttest.py ../data/eval/analysis/consent_ndcg_3_pop.txt ../data/eval/analysis/sbert_zero_ndcg_3_pop.txt all
python eval/ttest.py ../data/eval/analysis/consent_mrr_1_pop.txt ../data/eval/analysis/sbert_zero_mrr_1_pop.txt all
python eval/ttest.py ../data/eval/analysis/consent_mrr_3_pop.txt ../data/eval/analysis/sbert_zero_mrr_3_pop.txt all
echo "+++++"

echo "Compare Our vs base (nDCG@3, P@1 and MRR@3) on questions ookb"
python eval/ttest.py ../data/eval/analysis/consent_ndcg_3_pop.txt ../data/eval/analysis/sbert_zero_ndcg_3_pop.txt ookb
python eval/ttest.py ../data/eval/analysis/consent_mrr_1_pop.txt ../data/eval/analysis/sbert_zero_mrr_1_pop.txt ookb
python eval/ttest.py ../data/eval/analysis/consent_mrr_3_pop.txt ../data/eval/analysis/sbert_zero_mrr_3_pop.txt ookb
echo "+++++"

echo "Compare Our vs base (nDCG@3, P@1 and MRR@3) on questions rare"
python eval/ttest.py ../data/eval/analysis/consent_ndcg_3_pop.txt ../data/eval/analysis/sbert_zero_ndcg_3_pop.txt rare
python eval/ttest.py ../data/eval/analysis/consent_mrr_1_pop.txt ../data/eval/analysis/sbert_zero_mrr_1_pop.txt rare
python eval/ttest.py ../data/eval/analysis/consent_mrr_3_pop.txt ../data/eval/analysis/sbert_zero_mrr_3_pop.txt rare
echo "+++++"

echo "Compare Our vs base (nDCG@3, P@1 and MRR@3) on questions uncommon"
python eval/ttest.py ../data/eval/analysis/consent_ndcg_3_pop.txt ../data/eval/analysis/sbert_zero_ndcg_3_pop.txt uncommon
python eval/ttest.py ../data/eval/analysis/consent_mrr_1_pop.txt ../data/eval/analysis/sbert_zero_mrr_1_pop.txt uncommon
python eval/ttest.py ../data/eval/analysis/consent_mrr_3_pop.txt ../data/eval/analysis/sbert_zero_mrr_3_pop.txt uncommon
echo "+++++"

echo "Turn analysis of base"
python eval/analysis/turn_analysis.py ../data/eval/sbert_zero_ndcg_3.txt
echo "+++++"

echo "Turn analysis of Ours"
python eval/analysis/turn_analysis.py ../data/eval/consent_ndcg_3.txt
echo "+++++"
