import pandas as pd

from utils import plot_results

test_sel_ps = pd.read_pickle('results_uncertainty_rank_pseudo_labels.pkl')
test_sel_ps_sw = pd.read_pickle('results_uncertainty_rank_pseudo_labels_sample_weight.pkl')
test_sel = pd.read_pickle('results_uncertainty_rank.pkl')
test_rnd = pd.read_pickle('results_random_selection.pkl')

test_sel_ps = test_sel_ps.values
test_sel_ps_sw = test_sel_ps_sw.values
test_sel = test_sel.values
test_rnd = test_rnd.values

plot_results(test_sel, test_rnd, 'Uncertainty rank')

plot_results(test_sel_ps, test_rnd, 'Uncertainty rank\nwith pseudo-labeling')

plot_results(test_sel_ps_sw, test_rnd, 'Uncertainty rank\nwith pseudo-labeling\nand variable weighting')
