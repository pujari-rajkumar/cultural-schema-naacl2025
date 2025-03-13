 #!/usr/bin/env python
# coding: utf-8

# Import required utility packages
import os
import json
import re
import csv
import random
from tqdm.auto import tqdm
import pandas as pd

random.seed(4056)

# ### Utils

# Data paths and parameter save paths:
fpath = '/home/rpujari/gilbreth_scratch/scratch_ml/DARPA/'
mpdd_path = '/home/rpujari/gilbreth_scratch/scratch_ml/DARPA/mpdd/'

# Utility functions for text and other input processing
def get_points(sents):
    bulleted = False
    start_idx = 0
    c = 0
    for i, sent in enumerate(sents):
        if sent.strip():
            char0 = sent.strip().split('.')[0]
            if char0 == '1':
                bulleted = True
                start_idx = i
            c += 1
            if c == 3:
                break
    pnum = 1
    bullets = []
    if bulleted:
        while start_idx < len(sents):
            if sents[start_idx].strip().startswith(str(pnum) + '.'):
                bullets.append(sents[start_idx].strip())
                pnum += 1
            else:
                bullets[-1] += '\n' + sents[start_idx].strip()
            start_idx += 1
    return bulleted, bullets

def load_translations():    
    # Load translations of all dataset turns
    remaining_translations = {'input': {'prompts': [], 'ids': []}, 'output': []}
    fnames = os.listdir(fpath + '/translation_outputs/')
    for fname in tqdm(fnames, desc='Translations'):
        if fname.endswith('.json'):
            in_dict = json.load(open(fpath + '/translation_outputs/' + fname))
            for key in in_dict['input']:
                remaining_translations['input'][key] += in_dict['input'][key]
            remaining_translations['output'] += in_dict['output']
    
    print(len(remaining_translations['input']['ids']))
    
    # Create a dict of translations
    remaining_translation_dict = {}
    cc, mc, lc = 0, 0, 0
    for id_, out in zip(remaining_translations['input']['ids'], remaining_translations['output']):
        if '_' in id_:
            remaining_translation_dict[id_] = out
            cc += 1
        elif id_.count('-') == 3:
            fid, dnum, sc, ec = id_.split('-')
            if fid + '-' + dnum not in remaining_translation_dict:
                remaining_translation_dict[fid + '-' + dnum] = {}
            remaining_translation_dict[fid + '-' + dnum][sc + '-' + ec] = out
            lc += 1
        else:
            did, tnum = id_.split('-')
            if did not in remaining_translation_dict:
                remaining_translation_dict[did] = {}
            remaining_translation_dict[did][int(tnum)] = out
            mc += 1
    print(mc, cc, lc)

    return remaining_translation_dict


# ### MPDD
def load_mpdd_data():
    remaining_translation_dict = load_translations()
    
    # Load MPDD data
    mpdd_metadata = json.load(open(mpdd_path + 'metadata.json'))
    mpdd_dialogues = json.load(open(mpdd_path + 'dialogue.json'))
    llama_predicted_metadata = json.load(open(mpdd_path + 'llama_predicted_metadata.json'))
    
    # Load GPT-3.5 augmentations for social norms, violations, and effects
    mpdd_gpt_responses = json.load(open(mpdd_path + 'mpdd_gpt_resp.json'))
    
    # Create train, dev, test splits for MPDD data
    mpdd_train = []
    mpdd_valid = []
    mpdd_test = []
    assigned_data_types = {}
    random.seed(4056)
    for d_id in mpdd_dialogues:
        toss = random.random()
        if toss <= 0.7:
            mpdd_train.append(d_id)
            assigned_data_types[d_id] = 'train'
        elif toss <= 0.8:
            mpdd_valid.append(d_id)
            assigned_data_types[d_id] = 'valid'
        else:
            mpdd_test.append(d_id)
            assigned_data_types[d_id] = 'test'
    print(len(mpdd_train), len(mpdd_valid), len(mpdd_test))
    
    
    # Load T5-predicted symbols for MPDD data
    mpdd_norm_cats, mpdd_norm_ids = json.load(open(mpdd_path + 'predicted_norm_categories.json'))
    mpdd_violation_status, mpdd_violation_ids = json.load(open(mpdd_path + 'predicted_violation_status.json'))
    mpdd_changepoints, mpdd_changepoint_ids = json.load(open(mpdd_path + 'predicted_changepoints.json'))
    
    
    # Make a dictionary of gold annotated symbols
    rev_position = {}
    rev_field = {}
    
    for key in mpdd_metadata['field']:
        for reln in mpdd_metadata['field'][key]:
            rev_field[reln] = key
    
    for key in mpdd_metadata['position']:
        for reln in mpdd_metadata['position'][key]:
            rev_position[reln] = key
    
    
    # Load human-in-the-loop annotated cluster and theme data
    norms_data = pd.read_csv(fpath + '/human-in-loop-clustering/flask-gui/norms_dialogues_themes_relevance_symbols_data.csv')

    # ### Create Graphs
    # Make graph dictionaries for all the conversations in the dataset
    mpdd_graphs = {}
    for d_id in tqdm(mpdd_dialogues, desc='Dialogues'):
        if d_id in mpdd_gpt_responses:
            mpdd_graph = {
                'split': None,
                'dialogue': '',
                'summary': '',
                'metadata': '',
                'turns': [],
                'turns_en': [],
                'norms': [],
                'violations': [],
                'effects': [],
                
                'field': [],
                'norm_themes': [],
                'norm_relevance': [],
                'symbolic_data': [],
                # 'violation_themes': [],
                
                'emotions': [],
                'positions': [],
                'relationships': [],
                'norm_categories': [],
                'violation_status': [],
                'change_points': []
            }
            
            mpdd_graph['split'] = assigned_data_types[d_id]
            dialogue = mpdd_dialogues[d_id]
            norms, violations, ob_effects, po_effects = mpdd_gpt_responses[d_id]
    
    
            sel_norms_data = norms_data[norms_data.identifier == f'mpdd-{d_id}']
            sel_data = sel_norms_data[['text', 'name', 'description', 'violation_characteristic', 'activation_settings', 'actors', 'recepients', 'symbolic_annotation', 'relevance_judgment', 'symbolic_quality']]
            for index, row in sel_data.iterrows():
                mpdd_graph['norms'].append((row['text'].strip(), 'en-bprop'))
                mpdd_graph['norm_relevance'].append((row['relevance_judgment'].strip(), 'en'))
                
                tname = row['name'].strip()
                if not (tname.startswith('KMeans') or tname.lower() == 'unknown'):
                    # convert camel case to normal phrase
                    tname = re.sub(r'(?<!^)(?=[A-Z])', ' ', tname).lower()
                    mpdd_graph['norm_themes'].append((f"Theme: {tname}\nDescription: {row['description']}\nSettings: {row['activation_settings']}\nViolation Sketch: {row['violation_characteristic']}", 'en'))
                else:
                    mpdd_graph['norm_themes'].append(('Unknown', 'en'))
                if row['symbolic_quality']:
                    mpdd_graph['symbolic_data'].append((row['symbolic_annotation'], 'en'))
                else:
                    mpdd_graph['symbolic_data'].append(('', 'en'))
                    
            # if 'mpdd-' + str(d_id) in all_norm_data:
            #     mpdd_graph['norms'] = []
            #     for nid in all_norm_data['mpdd-' + str(d_id)]:
            #         n, tid = all_norm_data['mpdd-' + str(d_id)][nid]
            #         mpdd_graph['norms'].append((n, 'en-bprop'))
            #         t = themes_dict[tid]
            #         if not (t[0].strip().startswith('KMeans') or t[0].strip() == 'Unknown'):
            #             tname = re.sub(r'(?<!^)(?=[A-Z])', ' ', t[0]).lower()
            #             # print(tname)
            #             mpdd_graph['norm_themes'].append(('Theme: ' + tname + '\n' + 'Description: ' + t[1].strip(), 'en'))
            #         else:
            #             mpdd_graph['norm_themes'].append(('Unknown', 'en'))
            # else:
            #     norms = get_points(norms)
            #     if norms[0]:
            #         mpdd_graph['norms'] = [(n, 'en-bprop') for n in norms[1]]
            
            violations = get_points(violations)
            if violations[0]:
                mpdd_graph['violations'] = [(v, 'en') for v in violations[1]]
            
            ob_effects = get_points(ob_effects)
            if ob_effects[0]:
                mpdd_graph['effects'] = [(e, 'en') for e in ob_effects[1]]
                
            gpt_responses = json.load(open(mpdd_path + 'gpt_responses/response_' + d_id + '.json'))
            summary = None
            for ques in gpt_responses:
                if 'translat' in ques.lower():
                    tlines = gpt_responses[ques].strip().split('\n')
                elif 'summarize' in ques.lower():
                    summary = gpt_responses[ques].strip()
    
            dg_text = ''
            fields = set()
            for tnum, turn in enumerate(dialogue):
                positions = set()
                relationships = set()
                for listener in turn['listener']:
                    fields.add(rev_field[listener['relation']])
                    positions.add(rev_position[listener['relation']])
                    relationships.add(listener['relation'])
                mpdd_graph['relationships'].append([(r, 'en') for r in sorted(list(relationships))])
                mpdd_graph['positions'].append([(p, 'en') for p in sorted(list(positions))])
                mpdd_graph['emotions'].append(turn['emotion'])
                mpdd_graph['norm_categories'].append(mpdd_norm_cats[mpdd_norm_ids.index(d_id + '-' + str(tnum))])
                mpdd_graph['violation_status'].append(mpdd_violation_status[mpdd_violation_ids.index(d_id + '-' + str(tnum))])
                mpdd_graph['change_points'].append(mpdd_changepoints[mpdd_changepoint_ids.index(d_id + '-' + str(tnum))])
                if len(dialogue) == len(tlines):
                    dg_text += tlines[tnum].strip() + '\n'
                    mpdd_graph['turns_en'].append((tlines[tnum].strip(), 'en-bprop'))
                else:
                    dg_text += remaining_translation_dict[d_id][tnum].strip() + '\n'
                    mpdd_graph['turns_en'].append((remaining_translation_dict[d_id][tnum].strip(), 'en-bprop'))
                mpdd_graph['turns'].append((turn['utterance'].strip(), 'zh-bprop'))
            
            mpdd_graph['dialogue'] = (dg_text.strip(), 'en')
            mpdd_graph['field'] = [(f, 'en') for f in sorted(list(fields))]
            mpdd_graph['summary'] = (summary.strip(), 'en-bprop')
            mpdd_graph['metadata'] = (llama_predicted_metadata[d_id], 'en')
            mpdd_graphs[d_id] = mpdd_graph
            
    return mpdd_graphs #, assigned_data_types, (mpdd_train, mpdd_valid, mpdd_test)