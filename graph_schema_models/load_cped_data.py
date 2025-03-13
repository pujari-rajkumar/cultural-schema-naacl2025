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
cped_path = '/home/rpujari/gilbreth_scratch/scratch_ml/DARPA/CPED/data/CPED/'
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


# ### CPED
def load_cped_data():
    remaining_translation_dict = load_translations()
    
    cped_train = list(csv.reader(open(cped_path + 'train_split.csv')))
    cped_valid = list(csv.reader(open(cped_path + 'valid_split.csv')))
    cped_test = list(csv.reader(open(cped_path + 'test_split.csv')))
    cped_header = cped_train[0]
    
    cped_gpt_responses = json.load(open(cped_path + 'cped_gpt_resp.json'))
    cped_relationships = json.load(open(cped_path + 'llama_predicted_relationships.json'))
    cped_metadata = json.load(open(cped_path + 'llama_predicted_metadata.json'))
    
    mpdd_metadata = json.load(open(mpdd_path + 'metadata.json'))
    rev_position = {}
    rev_field = {}
    for reln in mpdd_metadata['relation']:
        for field in mpdd_metadata['field']:
            if reln in mpdd_metadata['field'][field]:
                if reln not in rev_field:
                    rev_field[reln] = field
    for reln in mpdd_metadata['relation']:
        for position in mpdd_metadata['position']:
            if reln in mpdd_metadata['position'][position]:
                if reln not in rev_position:
                    rev_position[reln] = position
    
    cped_dialogues = {}
    for row in cped_train[1:] + cped_valid[1:] + cped_test[1:]:
        tv_id = row[0]
        dia_id = row[1]
        ut_id = row[2]
        if dia_id not in cped_dialogues:
            cped_dialogues[dia_id] = []
        cped_dialogues[dia_id].append((ut_id, row[cped_header.index('Speaker')] + ': ' + row[cped_header.index('Utterance')]))

    for dia_id in cped_dialogues:
        cped_dialogues[dia_id] = sorted(cped_dialogues[dia_id], key=lambda x:int(x[0].split('_')[-1]))


    cped_translations = {}
    cped_summaries = {}
    for dia_id in cped_dialogues:
        summary = None
        gpt_response = json.load(open(cped_path + 'gpt_responses/response_' + dia_id + '.json'))
        for ques in gpt_response:
            if 'translat' in ques.lower():
                tlines = gpt_response[ques].strip().split('\n')
            if 'summarize' in ques.lower():
                cped_summaries[dia_id] = gpt_response[ques].strip()
        if len(tlines) == len(cped_dialogues[dia_id]):
            for trans, tup in zip(tlines, cped_dialogues[dia_id]):
                cped_translations[tup[0]] = trans
        else:
            for tup in cped_dialogues[dia_id]:
                cped_translations[tup[0]] = remaining_translation_dict[tup[0]]

    cped_emotions = {}
    cped_sentiments = {}
    cped_splits = {}
    cped_norm_categories = {}
    # cped_personalities = {}

    for row in cped_train[1:]:
        cped_splits[row[1]] = 'train'

    for row in cped_valid[1:]:
        cped_splits[row[1]] = 'valid'

    for row in cped_test[1:]:
        cped_splits[row[1]] = 'test'

    for row in cped_train[1:] + cped_valid[1:] + cped_test[1:]:
        cped_emotions[row[2]] = row[cped_header.index('Emotion')]
        cped_sentiments[row[2]] = row[cped_header.index('Sentiment')]
        cped_norm_categories[row[2]] = row[cped_header.index('DA')]

    # Load human-in-the-loop annotated cluster and theme data
    norms_data = pd.read_csv(fpath + '/human-in-loop-clustering/flask-gui/norms_dialogues_themes_relevance_symbols_data.csv')

    cped_graphs = {}

    for d_id in tqdm(cped_dialogues, desc='Graphs'):
        if d_id in cped_gpt_responses:
            cped_graph = {
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
                'sentiments': [],
                'positions': [],
                'relationships': [],
                'norm_categories': [],
                # 'violation_status': [],
                # 'change_points': []
            }

            dialogue = cped_dialogues[d_id]
            cped_graph['split'] = cped_splits[d_id]
            norms, violations, ob_effects, po_effects = cped_gpt_responses[d_id]
            
            sel_norms_data = norms_data[norms_data.identifier == f'cped-{d_id}']
            sel_data = sel_norms_data[['text', 'name', 'description', 'violation_characteristic', 'activation_settings', 'actors', 'recepients', 'symbolic_annotation', 'relevance_judgment', 'symbolic_quality']]
            for index, row in sel_data.iterrows():
                cped_graph['norms'].append((row['text'].strip(), 'en-bprop'))
                cped_graph['norm_relevance'].append((row['relevance_judgment'].strip(), 'en'))
                
                tname = row['name'].strip()
                if not (tname.startswith('KMeans') or tname.lower() == 'unknown'):
                    # convert camel case to normal phrase
                    tname = re.sub(r'(?<!^)(?=[A-Z])', ' ', tname).lower()
                    cped_graph['norm_themes'].append((f"Theme: {tname}\nDescription: {row['description']}\nSettings: {row['activation_settings']}\nViolation Sketch: {row['violation_characteristic']}", 'en'))
                else:
                    cped_graph['norm_themes'].append(('Unknown', 'en'))
                if row['symbolic_quality']:
                    cped_graph['symbolic_data'].append((row['symbolic_annotation'], 'en'))
                else:
                    cped_graph['symbolic_data'].append(('', 'en'))

            # if 'cped-' + str(d_id) in all_norm_data:
            #     cped_graph['norms'] = []
            #     for nid in all_norm_data['cped-' + str(d_id)]:
            #         n, tid = all_norm_data['cped-' + str(d_id)][nid]
            #         cped_graph['norms'].append((n, 'en-bprop'))
            #         t = themes_dict[tid]
            #         if not (t[0].strip().startswith('KMeans') or t[0].strip() == 'Unknown'):
            #             tname = re.sub(r'(?<!^)(?=[A-Z])', ' ', t[0]).lower()
            #             cped_graph['norm_themes'].append(('Theme: ' + tname + '\n' + 'Description: ' + t[1].strip(), 'en'))
            #         else:
            #             cped_graph['norm_themes'].append(('Unknown', 'en'))
            # else:
            #     norms = get_points(norms)
            #     if norms[0]:
            #         cped_graph['norms'] = [(n, 'en-bprop') for n in norms[1]]

            violations = get_points(violations)
            if violations[0]:
                cped_graph['violations'] = [(v, 'en') for v in violations[1]]

            ob_effects = get_points(ob_effects)
            if ob_effects[0]:
                cped_graph['effects'] = [(e, 'en') for e in ob_effects[1]]

            po_effects = get_points(po_effects)

            dg_text = ''
            fields = set()
            for ut_id, ut in dialogue:
                positions = set()
                relationships = set()
                for relation in cped_relationships[ut_id][:2]:
                    relation = relation.lower()
                    fields.add(rev_field[relation])
                    positions.add(rev_position[relation])
                    relationships.add(relation)
                cped_graph['relationships'].append([(r, 'en') for r in sorted(list(relationships))])
                cped_graph['positions'].append([(p, 'en') for p in sorted(list(positions))])
                
                cped_graph['emotions'].append(cped_emotions[ut_id])
                cped_graph['sentiments'].append(cped_sentiments[ut_id])
                cped_graph['norm_categories'].append(cped_norm_categories[ut_id])
                dg_text += cped_translations[ut_id].strip() + '\n'
                cped_graph['turns_en'].append((cped_translations[ut_id].strip(), 'en-bprop'))
                cped_graph['turns'].append((ut.strip(), 'zh-bprop'))
                
            cped_graph['dialogue'] = (dg_text.strip(), 'en')
            cped_graph['field'] = [(f, 'en') for f in sorted(list(fields))]
            cped_graph['summary'] = (cped_summaries[d_id], 'en-bprop')
            cped_graph['metadata'] = (cped_metadata[d_id].strip(), 'en')
            cped_graphs[d_id] = cped_graph

    return cped_graphs