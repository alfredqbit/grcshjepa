# GR-CS-HJEPA Phase 1 Pilot Readiness Report

Pilot hardening only. These values estimate runtime, variance, failure modes, and metric sanity; they are not confirmatory dissertation results.

Rows analyzed: 184
Studies present: study1, study2, study3
Seeds present: 0, 1, 2, 3
Status counts: {'complete': 184}
Failure rate: 0.0

## Gate checks
- failure_rate_below_10_percent: PASS ({'check': 'failure_rate_below_10_percent', 'passed': True, 'value': 0.0})
- effective_rank_logged: PASS ({'check': 'effective_rank_logged', 'passed': True, 'rows': 8})
- val_prediction_loss_logged: PASS ({'check': 'val_prediction_loss_logged', 'passed': True, 'rows': 4})
- routing_surface_logged: PASS ({'check': 'routing_surface_logged', 'passed': True, 'rows': 40})

## Suggested confirmatory seed counts from pilot variance

- {'study': 'study1', 'task': 'maze', 'predictor': 'mlp', 'metric': 'val_pred_loss', 'pilot_sd': 2.9363266065497657e-05, 'target_half_width': 0.02, 'suggested_n_cap20': 2}
- {'study': 'study1', 'task': 'maze', 'predictor': 'spiking', 'metric': 'val_pred_loss', 'pilot_sd': 1.619347078386585e-05, 'target_half_width': 0.02, 'suggested_n_cap20': 2}
- {'study': 'study1', 'task': 'sorting', 'predictor': 'mlp', 'metric': 'val_pred_loss', 'pilot_sd': 0.0004674879416862808, 'target_half_width': 0.02, 'suggested_n_cap20': 2}
- {'study': 'study1', 'task': 'sorting', 'predictor': 'spiking', 'metric': 'val_pred_loss', 'pilot_sd': 0.0006001528596633972, 'target_half_width': 0.02, 'suggested_n_cap20': 2}
- {'study': 'study2', 'task': 'maze_action', 'head_type': 'classification', 'metric': 'test_acc', 'pilot_sd': 0.033979136329947625, 'target_half_width': 0.03, 'suggested_n_cap20': 5}
- {'study': 'study2', 'task': 'sorting_head', 'head_type': 'regression', 'metric': 'test_mse', 'pilot_sd': 0.02613741263267676, 'target_half_width': 0.02, 'suggested_n_cap20': 7}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'load_targeted', 'damage_level': 0.05, 'metric': 'normalized_surface', 'pilot_sd': 0.013084813691885534, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'load_targeted', 'damage_level': 0.05, 'metric': 'surface_degradation', 'pilot_sd': 0.00810977703140924, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'load_targeted', 'damage_level': 0.05, 'metric': 'traffic_degradation', 'pilot_sd': 3.041428809013131, 'target_half_width': 1.0, 'suggested_n_cap20': 20}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'load_targeted', 'damage_level': 0.1, 'metric': 'normalized_surface', 'pilot_sd': 0.019795109488163275, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'load_targeted', 'damage_level': 0.1, 'metric': 'surface_degradation', 'pilot_sd': 0.013374921916900777, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'load_targeted', 'damage_level': 0.1, 'metric': 'traffic_degradation', 'pilot_sd': 3.2951727376960616, 'target_half_width': 1.0, 'suggested_n_cap20': 20}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'load_targeted', 'damage_level': 0.15, 'metric': 'normalized_surface', 'pilot_sd': 0.024282983049775813, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'load_targeted', 'damage_level': 0.15, 'metric': 'surface_degradation', 'pilot_sd': 0.018084247595365623, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'load_targeted', 'damage_level': 0.15, 'metric': 'traffic_degradation', 'pilot_sd': 3.7381506218805955, 'target_half_width': 1.0, 'suggested_n_cap20': 20}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'none', 'damage_level': 0.0, 'metric': 'normalized_surface', 'pilot_sd': 0.013078825038054099, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'spatial', 'damage_level': 0.05, 'metric': 'normalized_surface', 'pilot_sd': 0.01495097733880006, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'spatial', 'damage_level': 0.05, 'metric': 'surface_degradation', 'pilot_sd': 0.001984403041944589, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'spatial', 'damage_level': 0.05, 'metric': 'traffic_degradation', 'pilot_sd': 0.20572034430834069, 'target_half_width': 1.0, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'spatial', 'damage_level': 0.1, 'metric': 'normalized_surface', 'pilot_sd': 0.010238142936722774, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'spatial', 'damage_level': 0.1, 'metric': 'surface_degradation', 'pilot_sd': 0.0049713244983912025, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'spatial', 'damage_level': 0.1, 'metric': 'traffic_degradation', 'pilot_sd': 2.595772474814195, 'target_half_width': 1.0, 'suggested_n_cap20': 20}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'spatial', 'damage_level': 0.15, 'metric': 'normalized_surface', 'pilot_sd': 0.017796339817096458, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'spatial', 'damage_level': 0.15, 'metric': 'surface_degradation', 'pilot_sd': 0.007686235961608342, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'spatial', 'damage_level': 0.15, 'metric': 'traffic_degradation', 'pilot_sd': 2.7772645840190875, 'target_half_width': 1.0, 'suggested_n_cap20': 20}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'uniform', 'damage_level': 0.05, 'metric': 'normalized_surface', 'pilot_sd': 0.013600299570632988, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'uniform', 'damage_level': 0.05, 'metric': 'surface_degradation', 'pilot_sd': 0.0005864025857408954, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'uniform', 'damage_level': 0.05, 'metric': 'traffic_degradation', 'pilot_sd': 0.1052544201627694, 'target_half_width': 1.0, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'uniform', 'damage_level': 0.1, 'metric': 'normalized_surface', 'pilot_sd': 0.01369324805192407, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'uniform', 'damage_level': 0.1, 'metric': 'surface_degradation', 'pilot_sd': 0.002826613669516796, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'uniform', 'damage_level': 0.1, 'metric': 'traffic_degradation', 'pilot_sd': 1.1763763421890927, 'target_half_width': 1.0, 'suggested_n_cap20': 6}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'uniform', 'damage_level': 0.15, 'metric': 'normalized_surface', 'pilot_sd': 0.014746904362718201, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'uniform', 'damage_level': 0.15, 'metric': 'surface_degradation', 'pilot_sd': 0.004983302250972911, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'euclidean_length', 'damage_type': 'uniform', 'damage_level': 0.15, 'metric': 'traffic_degradation', 'pilot_sd': 1.7283253727869907, 'target_half_width': 1.0, 'suggested_n_cap20': 12}
- {'study': 'study3', 'variant': 'full_surface', 'damage_type': 'load_targeted', 'damage_level': 0.05, 'metric': 'normalized_surface', 'pilot_sd': 0.004659810564086949, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'full_surface', 'damage_type': 'load_targeted', 'damage_level': 0.05, 'metric': 'surface_degradation', 'pilot_sd': 0.002648535471616007, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'full_surface', 'damage_type': 'load_targeted', 'damage_level': 0.05, 'metric': 'traffic_degradation', 'pilot_sd': 6.1814800773582865, 'target_half_width': 1.0, 'suggested_n_cap20': 20}
- {'study': 'study3', 'variant': 'full_surface', 'damage_type': 'load_targeted', 'damage_level': 0.1, 'metric': 'normalized_surface', 'pilot_sd': 0.005972094025460245, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'full_surface', 'damage_type': 'load_targeted', 'damage_level': 0.1, 'metric': 'surface_degradation', 'pilot_sd': 0.0031387130113305134, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'full_surface', 'damage_type': 'load_targeted', 'damage_level': 0.1, 'metric': 'traffic_degradation', 'pilot_sd': 8.23318917503843, 'target_half_width': 1.0, 'suggested_n_cap20': 20}
- {'study': 'study3', 'variant': 'full_surface', 'damage_type': 'load_targeted', 'damage_level': 0.15, 'metric': 'normalized_surface', 'pilot_sd': 0.0056689977463399484, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'full_surface', 'damage_type': 'load_targeted', 'damage_level': 0.15, 'metric': 'surface_degradation', 'pilot_sd': 0.003243902768574214, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'full_surface', 'damage_type': 'load_targeted', 'damage_level': 0.15, 'metric': 'traffic_degradation', 'pilot_sd': 8.424996839201976, 'target_half_width': 1.0, 'suggested_n_cap20': 20}
- {'study': 'study3', 'variant': 'full_surface', 'damage_type': 'none', 'damage_level': 0.0, 'metric': 'normalized_surface', 'pilot_sd': 0.0050137837696784425, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'full_surface', 'damage_type': 'spatial', 'damage_level': 0.05, 'metric': 'normalized_surface', 'pilot_sd': 0.005158279466340439, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'full_surface', 'damage_type': 'spatial', 'damage_level': 0.05, 'metric': 'surface_degradation', 'pilot_sd': 0.0002921151276274634, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'full_surface', 'damage_type': 'spatial', 'damage_level': 0.05, 'metric': 'traffic_degradation', 'pilot_sd': 0.48045041975659464, 'target_half_width': 1.0, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'full_surface', 'damage_type': 'spatial', 'damage_level': 0.1, 'metric': 'normalized_surface', 'pilot_sd': 0.005647469638851747, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'full_surface', 'damage_type': 'spatial', 'damage_level': 0.1, 'metric': 'surface_degradation', 'pilot_sd': 0.0014631575832404033, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'full_surface', 'damage_type': 'spatial', 'damage_level': 0.1, 'metric': 'traffic_degradation', 'pilot_sd': 13.52882506312665, 'target_half_width': 1.0, 'suggested_n_cap20': 20}
- {'study': 'study3', 'variant': 'full_surface', 'damage_type': 'spatial', 'damage_level': 0.15, 'metric': 'normalized_surface', 'pilot_sd': 0.004309985639751154, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'full_surface', 'damage_type': 'spatial', 'damage_level': 0.15, 'metric': 'surface_degradation', 'pilot_sd': 0.0007159622203212718, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'full_surface', 'damage_type': 'spatial', 'damage_level': 0.15, 'metric': 'traffic_degradation', 'pilot_sd': 15.822990958272387, 'target_half_width': 1.0, 'suggested_n_cap20': 20}
- {'study': 'study3', 'variant': 'full_surface', 'damage_type': 'uniform', 'damage_level': 0.05, 'metric': 'normalized_surface', 'pilot_sd': 0.005027691073317917, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'full_surface', 'damage_type': 'uniform', 'damage_level': 0.05, 'metric': 'surface_degradation', 'pilot_sd': 0.00020782543245177724, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'full_surface', 'damage_type': 'uniform', 'damage_level': 0.05, 'metric': 'traffic_degradation', 'pilot_sd': 1.9609699450844338, 'target_half_width': 1.0, 'suggested_n_cap20': 15}
- {'study': 'study3', 'variant': 'full_surface', 'damage_type': 'uniform', 'damage_level': 0.1, 'metric': 'normalized_surface', 'pilot_sd': 0.004924746679739257, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'full_surface', 'damage_type': 'uniform', 'damage_level': 0.1, 'metric': 'surface_degradation', 'pilot_sd': 0.000664075918011245, 'target_half_width': 0.05, 'suggested_n_cap20': 2}
- {'study': 'study3', 'variant': 'full_surface', 'damage_type': 'uniform', 'damage_level': 0.1, 'metric': 'traffic_degradation', 'pilot_sd': 1.5667438405204812, 'target_half_width': 1.0, 'suggested_n_cap20': 10}
- {'study': 'study3', 'variant': 'full_surface', 'damage_type': 'uniform', 'damage_level': 0.15, 'metric': 'normalized_surface', 'pilot_sd': 0.006342571427628849, 'target_half_width': 0.05, 'suggested_n_cap20': 2}

## Next actions

- Inspect failure manifests and rerun only administrative failures under the same seed.
- Replace remaining toy/smoke stand-ins with production architecture modules where needed.
- Freeze model variants, primary endpoints, seed list, hyperparameter budget, and analysis scripts before confirmatory runs.
- Use pilot variance to select 12 to 20 independent confirmatory seeds per primary variant.