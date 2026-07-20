# %% [markdown]
# # Causal Analysis

# %%
def default_params(): 
    return {
        'setting': 'balanced', #lenient, #strict
        'input_path': '/workspaces/mica/datax/causal_revision_out',
        'cache_dir': '/workspaces/CodeSmells/datax/hugging_face_cache',
        'output_path' : '/workspaces/mica/datax/causal_revision_out_results',
        'causal_analysis': {
            #'common_causes': ['t_length_token','t_token_counts', 't_ast_levels', 't_n_ast_nodes', 't_n_ast_errors', 't_n_identifiers', 't_complexity', 't_nloc'],
            'common_causes': ['t_token_counts', 't_n_ast_nodes', 't_complexity', 't_nloc'],
            #'effect_modifiers' : ['e_token_counts', 'e_ast_levels', 'e_n_ast_nodes', 'e_n_ast_errors', 'e_n_identifiers', 'e_complexity', 'e_nloc'],
            'effect_modifiers' : [],
            'instruments' : []
        }
    }
params = default_params()

# %%
models = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']

# %% [markdown]
# ### Imports

# %%
import pandas as pd
import numpy as np
from dowhy import CausalModel
import dowhy.datasets
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import zscore
import json
import os
import scipy.stats as stats

# %%
import matplotlib.pyplot as plt

# %% [markdown]
# ### Causal Modeling

# %%
class RefutationResult:
    def __init__(self, refutation_result=None, new_effect=None):
        self.refutation_result = refutation_result
        self.new_effect = new_effect

# %%
def compute_ATE_and_refute(causal_model, identified_estimand, method_name, method_params={}):
    try:
        estimate = causal_model.estimate_effect(identified_estimand, method_name, method_params = method_params, test_significance=True)
    except Exception as e:
        print(f"======= ERROR COMPUTING CE for {method_name}: {e} ============")
        return {'method_name': method_name, 
            'estimated_effect': None,
            'refutation_placebo_permute' : None, 
            'refutation_unobserved_confounder' : None, 
            'refutation_subset' : None}
    
    refutation_placebo_permute = RefutationResult()
    refutation_unobserved_confounder = RefutationResult()
    refutation_subset = RefutationResult()
    refutation_random_common_cause = RefutationResult()
    
    try:
        refutation_placebo_permute = causal_model.refute_estimate(
                identified_estimand, estimate, method_name="placebo_treatment_refuter", placebo_type="permute")
    except Exception as e:
        print(f"Error performing pacebo refutation - {e}")
    try: 
        refutation_unobserved_confounder = causal_model.refute_estimate(
                identified_estimand, estimate, method_name="add_unobserved_common_cause")
    except Exception as e:
        print(f"Error performing unobserved covariate refutation - {e}")
    try:
        refutation_subset = causal_model.refute_estimate(
                identified_estimand, estimate, method_name="data_subset_refuter")
    except Exception as e:
        print(f"Error performing subset refutation - {e}")
    try: 
        refutation_random_common_cause = causal_model.refute_estimate(
                identified_estimand, estimate, method_name="random_common_cause")
    except Exception as e:
        print(f"Error performing random covariate refutation - {e}")
    

    return {'method_name': method_name, 
            'estimated_effect': estimate.value,
            'refutation_random_common_cause' : {'result' : refutation_random_common_cause.refutation_result, 'new_effect': refutation_random_common_cause.new_effect},
            'refutation_placebo_permute' : {'result' : refutation_placebo_permute.refutation_result, 'new_effect': refutation_placebo_permute.new_effect}, 
            'refutation_unobserved_confounder' : {'result' : refutation_unobserved_confounder.refutation_result, 'new_effect': refutation_unobserved_confounder.new_effect}, 
            'refutation_subset' : {'result' : refutation_subset.refutation_result, 'new_effect': refutation_subset.new_effect},}

def compute_correlations(input_corr_data, outout_corr_data):
    pearson_corr =  stats.pearsonr(input_corr_data, outout_corr_data)
    spearman_corr = stats.spearmanr(input_corr_data, outout_corr_data)
    kendall_corr = stats.kendalltau(input_corr_data, outout_corr_data)
    return {'pearson_corr' : pearson_corr.statistic, 'spearman_corr' : spearman_corr.statistic, 'kendall_corr' : kendall_corr.statistic}

def compute_causal_effects(causal_model):
    causal_effects_df = pd.DataFrame(columns=['method_name', 'pearson_corr', 'spearman_corr', 'kendall_corr' ,'estimated_effect', 'refutation_placebo_permute', 'refutation_unobserved_confounder', 'refutation_subset'])
    ####### COMPUTE PEARSON
    #correlation_results = compute_correlations(list(causal_model._data[causal_model._common_causes].mean(axis=1)), list(causal_model._data[causal_model._outcome].mean(axis=1)))
    correlation_results = compute_correlations(causal_model._data['binary_treatment'].tolist(), list(causal_model._data[causal_model._outcome].mean(axis=1)))
    ####### COMPUTE ESTIMAND
    identified_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
    ###### COMPUTE CAUSAL EFFECT - BACKDOOR - propensity_score_matching
    causal_effects_df.loc[len(causal_effects_df)] = {**correlation_results, **compute_ATE_and_refute(causal_model, identified_estimand, "backdoor.propensity_score_matching")}
    ###### COMPUTE CAUSAL EFFECT - BACKDOOR - propensity_score_stratification
    #causal_effects_df.loc[len(causal_effects_df)] = {**correlation_results, **compute_ATE_and_refute(causal_model, identified_estimand, "backdoor.propensity_score_stratification")}
    ###### COMPUTE CAUSAL EFFECT - BACKDOOR - backdoor.propensity_score_weighting
    #causal_effects_df.loc[len(causal_effects_df)] = {**correlation_results, **compute_ATE_and_refute(causal_model, identified_estimand, "backdoor.propensity_score_weighting")}
    ###### COMPUTE CAUSAL EFFECT - BACKDOOR - backdoor.propensity_score_weighting
    #causal_effects_df.loc[len(causal_effects_df)] = {**correlation_results, **compute_ATE_and_refute(causal_model, identified_estimand, "backdoor.generalized_linear_model", method_params={"glm_family": sm.families.Gaussian()})}
    
    return causal_effects_df.dropna()


# %%
def compute_pii_causal_effects(causal_hypothesis_df, treatment, pii_type):
    print(f"=========== group: {treatment} FOR TYPE-{pii_type}")
    pii_type_df = causal_hypothesis_df[causal_hypothesis_df['piiType']==pii_type].copy()
    pii_type_df = pii_type_df[pii_type_df['group'].isin(['T0', treatment])].copy()
    pii_type_df['binary_treatment'] = pii_type_df['group'].map(lambda group: True if group == treatment else False)
    pii_type_df = pii_type_df.dropna()
    ##### DEFINE CAUSAL MODEL
    causal_model = CausalModel(
            data = pii_type_df,
            treatment = ['binary_treatment'],
            outcome = ['outcome'],
            common_causes = params['causal_analysis']['common_causes'],
            effect_modifiers = params['causal_analysis']['effect_modifiers'],
            instruments = params['causal_analysis']['instruments'])
    result_df = compute_causal_effects(causal_model)
    return result_df
    

# %%
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# %% [markdown]
# ### EXECUTE

# %%
for model_id in models:
    ### Data Loading
    pii_dataset_df = pd.read_json(f"{params['input_path']}/{params['setting']}/{model_id}.json")
    causal_hypothesis_df = pii_dataset_df.copy()
    type_values = causal_hypothesis_df['type'].unique()
    type_map = {'easy': 'T0', 'hard': 'T1', 'ambiguous': 'T2'}
    causal_hypothesis_df['group'] = causal_hypothesis_df['type'].map(type_map)

    ## iterate on treatments

    for treatment in causal_hypothesis_df['group'].unique():
        if treatment != 'T0':
            treatment_pii_causal_results = []
            print(f"============================= COMPUTING ANALYSIS FOR TREATMENT {treatment} ==================================== ")
            ##### iterate per pii type
            for pii_type in causal_hypothesis_df['piiType'].unique():
                #### run causal analysis
                causal_results_df = compute_pii_causal_effects(causal_hypothesis_df, treatment, pii_type)
                causal_results_df['pii_type'] = pii_type
                treatment_pii_causal_results.append(causal_results_df)
            #### save results
            print(f"============================= STORING ANALYSIS FOR TREATMENT-{treatment} ==================================== ")
            pii_causal_results_df = pd.concat(treatment_pii_causal_results, ignore_index=True)
            output_path = f"{params['output_path']}/{params['setting']}/{model_id}"
            create_folder(output_path)
            pii_causal_results_df.to_json(f"{output_path}/{treatment}.json")


