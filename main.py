#Code documentation and readability currently in a draft state but made open-source for potential collaborators and interested parties. 

from openai import OpenAI
import yaml
import pandas as pd
from functools import partial
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from transformers import AutoTokenizer
from transformers import AutoModel
import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F
from typing import List
from transformers import GPT2LMHeadModel
import time
from transformers import logging
from io import StringIO
import transformer_lens
from sae_lens import HookedSAETransformer, SAE
from itertools import product


#Cosine metric-base baseline
def baseline(topic, generated_prompts):
    topic_embedding = sentence_embeddings_maps([topic])[0]
    generated_prompts_embeddings = sentence_embeddings_maps(generated_prompts)
    distances = [1-(cosine_distances([x.numpy()], [topic_embedding.numpy()])[0])+1/2 for x in generated_prompts_embeddings]
    return distances


#Functions from SAE Steer (Copied for temporary access) 
def gather_residual_activations(model, target_layer, inputs):
    """More generic approach to gathering activations (for Gemma).
    See specs for gather_residual_activationsgpt method.
    Output is slightly different format."""
    target_act = None
    def gather_target_act_hook(outputs, hook):
        nonlocal target_act
        target_act = outputs[0]
        return outputs
    _ = model.add_hook(f"blocks.{target_layer}.hook_resid_post", gather_target_act_hook)
    _ = model(inputs)
    model.reset_hooks()
    return target_act

def summarized_latents(prompts: List[str], tokenizer, device, model,
                    layer_id: int, sae, summary_approach='avg'):
    all_outputs = []
    for prompt in prompts:
        tok = tokenizer.encode(prompt, return_tensors="pt").to(device)
        target_act = gather_residual_activations(model, layer_id, tok)
        lats=  sae.encode(target_act.to(device))
        lats = lats.cpu().detach().numpy()
        output = pd.DataFrame(lats)
        # remove BOS token
        if len(lats.shape) > 1:
            modify_lats = output[1:].sum(axis=0)/output[1:].sum(axis=0).sum()
        else:
            modify_lats = output
        all_outputs.append(modify_lats)
    all_summary_latents = pd.concat(all_outputs, axis=1).T
    return all_summary_latents

def sentence_embeddings_maps(sample_inputs):
    def mean_pooling(model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask = attention_mask.unsqueeze(-1)
        input_mask_expanded = input_mask.expand(token_embeddings.size()).float()
        multiplication_tensors = torch.sum(token_embeddings * input_mask_expanded, 1)
        return multiplication_tensors / torch.clamp(input_mask_expanded.sum(1),
                 min=1e-9)

    sentences = sample_inputs
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    encoded_input = tokenizer(sentences, padding=True,
                            truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


device = "cuda"
model = transformer_lens.HookedTransformer.from_pretrained("gemma-2-2b", center_writing_weights=False) #Need Gated HF Access
tokenizer =  AutoTokenizer.from_pretrained("google/gemma-2-2b")
layer_id = len(model.blocks)-1
autoencoder, cfg_dict, sparsity = SAE.from_pretrained(
      release="gemma-scope-2b-pt-mlp",  
      sae_id=f"layer_{layer_id}/width_65k/average_l0_215", #changable
      device=device,
  )


def SAE_baseline_method(contrastive_prompts_training, safe_test, unsafe_test, topic):
  #Generate distances and preactivation summaries 
  ref_embd = pd.DataFrame(sentence_embeddings_maps(contrastive_prompts_training))
  topic_embedding = sentence_embeddings_maps([topic])[0]
  dist_df = [(-1*cosine_distances([x], [topic_embedding.numpy()])[0]+1)/2 for _, x in ref_embd.iterrows()]
  acts_ref_embd = summarized_latents(contrastive_prompts_training, tokenizer,
                                      device, model, layer_id, autoencoder)
  feature_num = np.zeros((acts_ref_embd.shape[1], 1))
  feature_denom = np.zeros((acts_ref_embd.shape[1], 1))

  #Calculate presence scores and manipulations
  for i, act_summary in acts_ref_embd.iterrows():
      og = (dist_df[int(i)]*dist_df[int(i)]*act_summary).astype(float)
      feature_num += np.reshape(og, feature_denom.shape)
      feature_denom += np.reshape(act_summary, feature_denom.shape)
  scores = pd.Series((feature_num/(feature_denom)).T[0]).dropna()
  scores = scores - scores.min()
  scores = scores/scores.max()
  contam_list = []
  for test in [safe_test, unsafe_test]:
    final_test_latents = summarized_latents(test, tokenizer,
                                        device, model, layer_id, autoencoder)
    normed_final_test_latents = final_test_latents.apply(lambda x: x/x.sum(), axis=1)
    contam_scores = [np.dot(x.iloc[scores.index], scores) for _, x in normed_final_test_latents.iterrows()]
    contam_list.append(contam_score)
  return pd.DataFrame(contam_list), scores



topics = ["violent examples", "inappropriate themes"]
specificity = ["for toddlers", "for elders with a heart condition"]


#Datasets available upon request due to sensitivity of data. 
for topic in zip(final_dfs, [' '.join(x) for x in list(product(topics, specificity))]):
  df = pd.read_csv(f"Dataset/{topic}", index_col=0)
  df = df[df.apply(lambda x: False if 'unsafe' in x[0] else True, axis=1)]
  df = df[df.apply(lambda x: False if 'safe' in x[1] else True, axis=1)]
  df = df.reset_index(drop=True)
  scores_list = []
  config_list = []
  n_list  = [1, 10, 50, 100, 500]
  for n in n_list:
    training_prompts = df.iloc[:n, :]
    safe_test = df.iloc[-50:, 0].to_list()
    unsafe_test = df.iloc[-50:, 1].to_list()
    contrastive= list(training_prompts.unstack().values)
    config, scores = SAE_baseline_method(contrastive, safe_test, unsafe_test, topic)
    scores_list.append(scores)
    config_list.append(config)
    config_df = pd.DataFrame(config)
    config_df.to_csv(f'{topic}_{n}_config.csv')
    scores_df = pd.DataFrame(scores_list)
    scores_df.to_csv(f'{topic}_{n}_scores.csv')

