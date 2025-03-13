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
ldc_path = '/home/rpujari/gilbreth_scratch/scratch0_ml/ldc_data/'
mpdd_path = '/home/rpujari/gilbreth_scratch/scratch_ml/DARPA/mpdd/'

plutchik =  ['anger', 'fear', 'sadness', 'disgust', 'surprise', 'anticipation', 'trust', 'joy', 'neutral']

source_data = ['ldc2022e11', 'ldc2022e19', 'ldc2022e20', 'ldc2022e22', 'ldc2023e03', 'ldc2023e06', 'ldc2023e07']
annotation_folders = ['ldc2022e18', 'ldc2023e01', 'ldc2023e20']
eval_data = ['ldc2022e22', 'ldc2023e07']

ldc_norm_ids = {
        101: 'apology',
        102: 'criticism',
        103: 'greeting',
        104: 'request',
        105: 'persuasion',
        106: 'thanking',
        107: 'leave',
        108: 'admiration',
        109: 'negotiation',
        110: 'refusal',
        150: 'other'
}

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


def is_overlapping(bo1, bo2):
    b1, e1 = bo1
    b2, e2 = bo2
    if e1 <= b2 or b1 >= e2:
        return False
    else:
        return True

def get_span(transcript, bounds):
    ret_spans = []
    ret_bounds = []
    tr_keys = sorted(list(transcript.keys()), key=lambda x:x[0])
    for bo in tr_keys:
        if is_overlapping(bounds, bo):
            ret_spans.append(transcript[bo][1]) 
            ret_bounds.append(bo)
    return ' '.join(ret_spans), ret_bounds

def get_transcript_data(data_path):
    transcript_data = {}
    fnames = os.listdir(data_path)
    for fname in fnames:
        if fname[:-4] not in transcript_data:
            transcript_data[fname[:-4]] = {}
        data = list(csv.reader(open(data_path + fname), delimiter='\t'))
        if len(data) > 0:
            if 'start_seconds' in data[0]:
                data = data[1:]
            for row in data:
                if len(row) == 4:
                    sc, ec, sp, text = row
                    seg_bounds = (float(sc.strip()), float(ec.strip()))
                    transcript_data[fname[:-4]][seg_bounds] = (sp.strip(), text.strip())
                else:
                    print(row)
    return transcript_data

def get_ldc_task_data(data_version, task, add_neutral=False, annotation_folders=annotation_folders,
                      get_text=True, get_audio=True, get_video=True, verbose=False):
    if verbose:
        print(data_version)
    if os.path.exists(ldc_path + data_version + '/data/video_transcripts/azure/') or os.path.exists(ldc_path + data_version + '/data/transcripts/azure/') or\
    os.path.exists(ldc_path + data_version + '/data/text/txt/'):
        tx_tfnames = []
        vi_tfnames = []
        au_tfnames = []
        
        tx_fc = 0
        vi_fc = 0
        au_fc = 0
        
        if get_video and os.path.exists(ldc_path + data_version + '/data/video_transcripts/azure/'):
            vi_tfnames += os.listdir(ldc_path + data_version + '/data/video_transcripts/azure/')
            vi_fc = len(os.listdir(ldc_path + data_version + '/data/video_transcripts/azure/'))
            vi_tfnames = [fn[:-4] for fn in vi_tfnames]
            
        if get_audio and os.path.exists(ldc_path + data_version + '/data/audio_transcripts/azure/'):
            au_tfnames += os.listdir(ldc_path + data_version + '/data/audio_transcripts/azure/')
            au_fc = len(os.listdir(ldc_path + data_version + '/data/audio_transcripts/azure/'))
            au_tfnames = [fn[:-4] for fn in au_tfnames]
        
        if get_text and os.path.exists(ldc_path + data_version + '/data/text/txt/'):
            tx_tfnames += os.listdir(ldc_path + data_version + '/data/text/txt/')
            tx_fc = len(os.listdir(ldc_path + data_version + '/data/text/txt/'))
            tx_tfnames = [fn[:-4] for fn in tx_tfnames]
        
        #file count for each modality and overall
        if verbose:
            print('File Counts:', tx_fc, au_fc, vi_fc, tx_fc + au_fc + vi_fc)
        tfnames = tx_tfnames + au_tfnames + vi_tfnames

        
        #read and save all segments from segments.tab 
        #(all_segments: file_ID -> segment_name -> seg_bounds, rev_segments: -> file_ID -> seg_bounds -> segment_name)
        as_c = 0
        all_segments = {}
        rev_segments = {}
        seg_data = []
        for ann_version in annotation_folders:
            seg_data += list(csv.reader(open(ldc_path + ann_version + '/docs/segments.tab'), delimiter='\t'))[1:]
        #drop header
        for row in seg_data:
            #each row have file_ID, segment_name, start_char/start_secs, end_char/end_secs
            ann_fname, seg_name, sc, ec = row
            seg_bounds = (float(sc), float(ec))
            if ann_fname in tfnames:
                if ann_fname not in all_segments:
                    all_segments[ann_fname] = {}
                all_segments[ann_fname][seg_name] = seg_bounds

                if ann_fname not in rev_segments:
                    rev_segments[ann_fname] = {}
                rev_segments[ann_fname][seg_bounds] = seg_name

        for tfname in all_segments:
            as_c += len(all_segments[tfname])

        #read and save annotations from perfect_submission for ED (annotations: file_ID -> seg_bounds -> emotion)
        an_c = 0 #annotated-segment-count
        annotations = {}
        label_dist = {}
        zf_c = 0 #files-with-annotated-segments-count
        zf_names = set() #files-with-annotated-segments
        ann_fc = 0 #annotated-file-count
        
        for tfname in tfnames:
            tf_anns = []
            for ann_version in annotation_folders:
                if os.path.exists(ldc_path + ann_version + '/perfect_submissions/' + task + '/' + tfname + '.tab'):
                    ann_fdata = list(csv.reader(open(ldc_path + ann_version + '/perfect_submissions/' + task + '/' + tfname + '.tab'), delimiter='\t'))
                    if tfname not in annotations:
                        if task == 'CD':
                            annotations[tfname] = []
                        else:
                            annotations[tfname] = {}
                        
                    tf_anns += ann_fdata[1:]
                    # if len(ann_fdata) > 1:
                    ann_fc += 1       
                    for ann in tf_anns:
                        #read annotation data from each row
                        if task == 'ED':
                            ann_fname, em, sc, ec, llr = ann
                        elif task == 'CD':
                            ann_fname, timestamp, llr = ann
                        elif task == 'ND':
                            ann_fname, norm, sc, ec, status, llr = ann
                        elif task in ['AD', 'VD']:
                            ann_fname, sc, ec, val = ann
                        
                        # write data to annotations dictionary (CD -> file-level annotations)
                        if task == 'CD':
                            annotations[tfname].append(float(timestamp))
                        else:
                            seg_bounds = (float(sc), float(ec))
                            if seg_bounds not in annotations[tfname]:
                                annotations[tfname][seg_bounds] = []
                        #other tasks -> segment level annotations
                        if task == 'ED':
                            annotations[tfname][seg_bounds].append(em)
                            if em not in label_dist:
                                label_dist[em] = 0
                            label_dist[em] += 1
                        elif task == 'ND':
                            annotations[tfname][seg_bounds].append((norm, status))
                            if (norm, status) not in label_dist:
                                label_dist[(norm, status)] = 0
                            label_dist[(norm, status)] += 1
                        elif task in ['AD', 'VD']:
                            annotations[tfname][seg_bounds].append(float(val))
                            
                        
                    an_c += len(annotations[tfname])
                    if len(annotations[tfname]) > 0:
                        zf_c += 1
                        zf_names.add(tfname)
        
        #files present in perfect submissions
        if verbose:
            print('File-count in perfect submissions:', ann_fc)
        #all_segment_count, annotated_segment_count, non_zero_annotation_file_count, file-count
        if verbose:
            print(f"All segment count: {as_c}, Annotated segments: {an_c}, Non-zero annotation file-count: {zf_c}, File count: {len(tfnames)}\n")
        
        if add_neutral and an_c > 0:
            random.seed(4056)
            #randomly sample neutral segments
            num_neutral_samples = int(np.mean([v for k, v in label_dist.items()]))
            neutral_segments = []
            for tfname in zf_names:
                annotated_segments = list(annotations[tfname].keys())
                for seg_bounds in rev_segments[tfname]:
                    sel = True
                    for bo in annotated_segments:
                        if is_overlapping(bo, seg_bounds):
                            sel = False
                            break
                    if sel:
                        neutral_segments.append((tfname, seg_bounds))
            sel_neutral_segments = random.sample(neutral_segments, min(num_neutral_samples, len(neutral_segments)))
            label_dist['neutral'] = len(sel_neutral_segments)

            #add sampled segments to annotations
            for tfname, seg_bounds in sel_neutral_segments:
                if tfname not in annotations:
                    annotations[tfname] = {}
                annotations[tfname][seg_bounds] = ['neutral']

        return all_segments, rev_segments, annotations, label_dist
    
def get_all_ldc_task_data(task, annotation_folders, source_data=source_data,
                          get_text=True, get_audio=True, get_video=True, verbose=False):
    all_ldc_data = [{}, {}, {}, {}, {}]
    for dv in source_data:
        out = get_ldc_task_data(dv, task, add_neutral=False, annotation_folders=annotation_folders,
                               get_text=get_text, get_audio=get_audio, get_video=get_video, verbose=verbose)
        if out:
            segs, rsegs, anns, ld = out
            all_ldc_data[0] |= segs
            all_ldc_data[1] |= rsegs
            all_ldc_data[2] |= anns
            for key in ld:
                if key in all_ldc_data[3]:
                    all_ldc_data[3][key] += ld[key]
                else:
                    all_ldc_data[3][key] = ld[key]
            for tfname in segs:
                all_ldc_data[4][tfname] = dv
    return all_ldc_data


def split_t5_response(response):
    state = 1
    points = []
    while str(state) + '.' in response:
        b = response.index(str(state) + '.')
        if str(state+1) + '.' in response:
            e = response.index(str(state+1) + '.')
        else:
            e = len(response)
        points.append(response[b:e])
        state += 1
    return points


def get_seg_anns(d_id, sb, ldc_data_train, ldc_data_valid, ldc_data_test, ldc_gpt_responses, all_in=False):
    segs_tr, rsegs_tr, anns_tr, ld_tr, dv_tr = ldc_data_train
    segs_va, rsegs_va, anns_va, ld_va, dv_va = ldc_data_valid
    segs_te, rsegs_te, anns_te, ld_te, dv_te = ldc_data_test
    
    seg_anns = []
    if all_in or d_id in ldc_gpt_responses:
        fname = d_id.split('-')[0]
        if fname in anns_tr:
            s, e = [float(x) for x in sb.split('-')]
            for sb_ann in anns_tr[fname]:
                if is_overlapping((s, e), sb_ann):
                    seg_anns += anns_tr[fname][sb_ann]
            return (seg_anns, 'train')
        if fname in anns_va:
            s, e = [float(x) for x in sb.split('-')]
            for sb_ann in anns_va[fname]:
                if is_overlapping((s, e), sb_ann):
                    seg_anns += anns_va[fname][sb_ann]
            return (seg_anns, 'valid')
        if fname in anns_te:
            s, e = [float(x) for x in sb.split('-')]
            for sb_ann in anns_te[fname]:
                if is_overlapping((s, e), sb_ann):
                    seg_anns += anns_te[fname][sb_ann]
            return (seg_anns, 'test')
    return seg_anns


def get_cp_anns(d_id, sb, ldc_data_train, ldc_data_valid, ldc_data_test, ldc_gpt_responses, all_in=False):
    segs_tr, rsegs_tr, anns_tr, ld_tr, dv_tr = ldc_data_train
    segs_va, rsegs_va, anns_va, ld_va, dv_va = ldc_data_valid
    segs_te, rsegs_te, anns_te, ld_te, dv_te = ldc_data_test
    
    seg_anns = []
    if all_in or d_id in ldc_gpt_responses:
        fname = d_id.split('-')[0]
        if fname in anns_tr:
            s, e = [float(x) for x in sb.split('-')]
            for sb_ann in anns_tr[fname]:
                if s <= float(sb_ann) and e >= float(sb_ann):
                    seg_anns += [True]
            return (seg_anns, 'train')
        if fname in anns_va:
            s, e = [float(x) for x in sb.split('-')]
            for sb_ann in anns_va[fname]:
                if s <= float(sb_ann) and e >= float(sb_ann):
                    seg_anns += [True]
            return (seg_anns, 'valid')
        if fname in anns_te:
            s, e = [float(x) for x in sb.split('-')]
            for sb_ann in anns_te[fname]:
                if s <= float(sb_ann) and e >= float(sb_ann):
                    seg_anns += [True]
            return (seg_anns, 'test')
    return seg_anns

# ### LDC Chinese
def load_ldc_chinese_data(get_text=True, get_audio=False, get_video=False):
    remaining_translation_dict = load_translations()
    
    ldc_emotion_train = get_all_ldc_task_data('ED', ['ldc2022e18/v6/'], get_text=get_text, get_audio=get_audio, get_video=get_video)
    ldc_emotion_valid = get_all_ldc_task_data('ED', ['ldc2023e01/v1/'], get_text=get_text, get_audio=get_audio, get_video=get_video)
    ldc_emotion_test = get_all_ldc_task_data('ED', ['ldc2023e20/v1/'], get_text=get_text, get_audio=get_audio, get_video=get_video)
    
    ldc_norm_train = get_all_ldc_task_data('ND', ['ldc2022e18/v6/'], get_text=get_text, get_audio=get_audio, get_video=get_video)
    ldc_norm_valid = get_all_ldc_task_data('ND', ['ldc2023e01/v1/'], get_text=get_text, get_audio=get_audio, get_video=get_video)
    ldc_norm_test = get_all_ldc_task_data('ND', ['ldc2023e20/v1/'], get_text=get_text, get_audio=get_audio, get_video=get_video)

    ldc_changepoint_train = get_all_ldc_task_data('CD', ['ldc2022e18/v6/'], get_text=get_text, get_audio=get_audio, get_video=get_video)
    ldc_changepoint_valid = get_all_ldc_task_data('CD', ['ldc2023e01/v1/'], get_text=get_text, get_audio=get_audio, get_video=get_video)
    ldc_changepoint_test = get_all_ldc_task_data('CD', ['ldc2023e20/v1/'], get_text=get_text, get_audio=get_audio, get_video=get_video)
    
    ldc_gpt_responses = json.load(open(ldc_path + 'gpt_responses/ldc_mandarin_gpt_resp.json'))
    selected_ldc_dialogues = json.load(open(ldc_path + 'saved_files/mandarin_text_all_dialogues_v3.json'))
    all_dialogue_metadata = json.load(open(ldc_path + 'saved_files/mandarin_text_all_dialogues_v3_metadata.json'))
    all_dialogue_relationships = json.load(open(ldc_path + 'saved_files/mandarin_text_all_dialogues_v3_relationships.json'))
    
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
    
    # Load human-in-the-loop annotated cluster and theme data
    norms_data = pd.read_csv(fpath + '/human-in-loop-clustering/flask-gui/norms_dialogues_themes_relevance_symbols_data.csv')
            
    ldc_graphs = {}

    for d_id in tqdm(selected_ldc_dialogues, desc='Graphs'):
        # print(ldc_path + 'gpt_responses/mandarin/response_' + d_id + '.json')
        if d_id in ldc_gpt_responses and os.path.exists(ldc_path + 'gpt_responses/mandarin/' + d_id + '.json'):
            # print('here')
            ldc_graph = {
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

                'positions': [],
                'relationships': [],
                'emotions': [],
                'norm_categories': [],
                'violation_status': [],
                'change_points': []
            }

            dialogue = selected_ldc_dialogues[d_id]
            norms, violations, ob_effects, po_effects = ldc_gpt_responses[d_id]

            sel_norms_data = norms_data[norms_data.identifier == f'ldc-{d_id}']
            sel_data = sel_norms_data[['text', 'name', 'description', 'violation_characteristic', 'activation_settings', 'actors', 'recepients', 'symbolic_annotation', 'relevance_judgment', 'symbolic_quality']]
            for index, row in sel_data.iterrows():
                ldc_graph['norms'].append((row['text'].strip(), 'en-bprop'))
                ldc_graph['norm_relevance'].append((row['relevance_judgment'].strip(), 'en'))
                
                tname = row['name'].strip()
                if not (tname.startswith('KMeans') or tname.lower() == 'unknown'):
                    # convert camel case to normal phrase
                    tname = re.sub(r'(?<!^)(?=[A-Z])', ' ', tname).lower()
                    ldc_graph['norm_themes'].append((f"Theme: {tname}\nDescription: {row['description']}\nSettings: {row['activation_settings']}\nViolation Sketch: {row['violation_characteristic']}", 'en'))
                else:
                    ldc_graph['norm_themes'].append(('Unknown', 'en'))
                if row['symbolic_quality']:
                    ldc_graph['symbolic_data'].append((row['symbolic_annotation'], 'en'))
                else:
                    ldc_graph['symbolic_data'].append(('', 'en'))
            
            # if 'ldc-' + d_id in all_norm_data:
            #     ldc_graph['norms'] = []
            #     for nid in all_norm_data['ldc-' + str(d_id)]:
            #         n, tid = all_norm_data['ldc-' + str(d_id)][nid]
            #         ldc_graph['norms'].append((n, 'en-bprop'))
            #         t = themes_dict[tid]
            #         if not (t[0].strip().startswith('KMeans') or t[0].strip() == 'Unknown'):
            #             tname = re.sub(r'(?<!^)(?=[A-Z])', ' ', t[0]).lower()
            #             # print(tname)
            #             ldc_graph['norm_themes'].append(('Theme: ' + tname + '\n' + 'Description: ' + t[1].strip(), 'en'))
            #         else:
            #             ldc_graph['norm_themes'].append(('Unknown', 'en'))
            # else:
            #     norms = get_points(norms)
            #     if norms[0]:
            #         ldc_graph['norms'] = [(n, 'en-bprop') for n in norms[1]]

            violations = get_points(violations)
            if violations[0]:
                ldc_graph['violations'] = [(v, 'en') for v in violations[1]]

            ob_effects = get_points(ob_effects)
            if ob_effects[0]:
                ldc_graph['effects'] = [(e, 'en') for e in ob_effects[1]]

            # po_effects = get_points(po_effects)
            gpt_responses = json.load(open(ldc_path + 'gpt_responses/mandarin/' + d_id + '.json'))
            summary = None
            for ques in gpt_responses:
                if 'translat' in ques.lower():
                    tlines = gpt_responses[ques].strip().split('\n')
                elif 'summarize' in ques.lower():
                    summary = gpt_responses[ques].strip()
            dg_text = ''
            fields = set()
            tnum = 0
            sorted_sbs = sorted(list(dialogue.keys()), key=lambda x:int(x.split('-')[0]))
            for sb in sorted_sbs:
                positions = set()
                relationships = set()
                for relation in all_dialogue_relationships[d_id][sb][:2]:
                    relation = relation.lower()
                    fields.add(rev_field[relation])
                    positions.add(rev_position[relation])
                    relationships.add(relation)
                ldc_graph['relationships'].append([(r, 'en') for r in sorted(list(relationships))])
                ldc_graph['positions'].append([(p, 'en') for p in sorted(list(positions))])
                
                emotions = get_seg_anns(d_id, sb, ldc_emotion_train, ldc_emotion_valid, ldc_emotion_test, ldc_gpt_responses)
                norms = get_seg_anns(d_id, sb, ldc_norm_train, ldc_norm_valid, ldc_norm_test, ldc_gpt_responses)
                cp = get_cp_anns(d_id, sb, ldc_changepoint_train, ldc_changepoint_valid, ldc_changepoint_test, ldc_gpt_responses)
                ldc_graph['split'] = emotions[1]
                # print(norms)
                if emotions and len(emotions[0]) > 0: 
                    ldc_graph['emotions'].append(emotions[0][0])
                else:
                    ldc_graph['emotions'].append('neutral')
                if norms and len(norms[0]) > 0:
                    ns = [ldc_norm_ids[int(t[0])] for t in norms[0]]
                    st = [t[1] for t in norms[0]]
                    ldc_graph['norm_categories'].append(ns[0])
                    ldc_graph['violation_status'].append('violate' if 'violate' in st else 'adhere')
                else:
                    ldc_graph['norm_categories'].append('other')
                    ldc_graph['violation_status'].append('adhere')
                if cp and len(cp[0]) > 0:
                    ldc_graph['change_points'].append('yes')
                else:
                    ldc_graph['change_points'].append('no')
                if len(dialogue) == len(tlines):
                    dg_text += tlines[tnum].strip() + '\n'
                    ldc_graph['turns_en'].append((tlines[tnum].strip(), 'en-bprop'))
                else:
                    dg_text += remaining_translation_dict[d_id][sb].strip() + '\n'
                    ldc_graph['turns_en'].append((remaining_translation_dict[d_id][sb].strip(), 'en-bprop'))
                ldc_graph['turns'].append((dialogue[sb].strip(), 'zh-bprop'))
                tnum += 1
            ldc_graph['dialogue'] = (dg_text.strip(), 'en')
            ldc_graph['field'] = [(f, 'en') for f in sorted(list(fields))]
            ldc_graph['summary'] = (summary.strip(), 'en-bprop')
            ldc_graph['metadata'] = (all_dialogue_metadata[d_id].strip(), 'en')
            ldc_graphs[d_id] = ldc_graph
            
    return ldc_graphs
                            