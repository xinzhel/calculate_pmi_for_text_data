import numpy as np
import os
from allennlp.data import Vocabulary
from data_utils import load_data
import argparse

def observed_over_expected(df):
    col_totals = df.sum(axis=0)
    total = col_totals.sum()
    row_totals = df.sum(axis=1)
    expected = np.outer(row_totals, col_totals) / total # Pr(word) in row * Pr(class) in cloumn 
    oe = df / expected 
    return oe


def pmi(df, positive=True):
    df = observed_over_expected(df)
    # Silence distracting warnings about log(0):
    with np.errstate(divide='ignore'):
        df = np.log(df)
    df[np.isinf(df)] = 0.0  # log(0) = 0
    if positive:
        df[df < 0] = 0.0
    return df
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='sst2')
    parser.add_argument('--label', type=str, default=1)
    parser.add_argument('--vocab_path', type=str, default='../models/sst2/lstm/vocabulary/')
    parser.add_argument(
        "--exclude_opinion_lexicon",
        action="store_true",
    )
    parser.add_argument('--num_display', type=int, default=20)
    parser.add_argument(
                    "--evaluate_words",
                    type=str,
                    action="append",
                    default=[],
                )
    
    args = parser.parse_args()
    
    # construct vocabulary
    vocab_dir = args.vocab_path
    dataloader = load_data(args.dataset, 'test', )
    try:
        print('Load vocabulary from ', vocab_dir)
        vocab = Vocabulary.from_files(vocab_dir)
        print('Load vocabulary with size: ', len(vocab._index_to_token['tokens']))
        print('The number of classes is ', len(vocab._index_to_token['labels']))
    except:
        print('No constructed vocabulary.')
        instances = dataloader.iter_instances()
        vocab = Vocabulary.from_instances(instances)


    # words x classes counting matrix
    file_name = f'{args.dataset}_count.npy'

    if not os.path.isfile(file_name):
        print('Construct words x classes counting matrix.')
        count_matrix = np.zeros(( len(vocab._index_to_token['tokens']),  len(vocab._index_to_token['labels'])))

        dataloader.index_with(vocab)
        instances = dataloader.iter_instances() # iter after applying index
        for instance in instances:
            label_id = instance.fields['label']._label_id
            for indexed_token in instance.fields['tokens']._indexed_tokens['tokens']['tokens']:
                count_matrix[indexed_token][label_id] += 1
                # print(indexed_token)
        print('Saving Count Matrix into ', file_name)
        np.save(file_name, count_matrix)
    else:
        print('Load words x classes counting matrix.')
        count_matrix = np.load(file_name)
    try:
        label_id = vocab._token_to_index['labels'][str(args.label)]
    except:
        label_id = vocab._token_to_index['labels'][int(args.label)]
    print(f'Top {args.num_display} frequent words for the targeted class')
    for i in np.sort(count_matrix[:, label_id])[-args.num_display:]:
        print(vocab._index_to_token['tokens'][i])

    # pmi
    pmi_df = pmi(count_matrix, positive=False)
    pmi_df = np.nan_to_num(pmi_df)
    
    # pmi rank
    if args.exclude_opinion_lexicon:
        print("exclude opinion lexicons.")
        from nltk.corpus import opinion_lexicon
        blacklist = list(opinion_lexicon.negative() )
        blacklist.extend( list(opinion_lexicon.positive()))
        for token in blacklist:
            try:
                indexed_token = vocab._token_to_index['tokens'][token]
                pmi_df[indexed_token] = -np.inf
            except:
                pass

    # descending order
    
    words_sorted_by_pmi = []
    for i in pmi_df[:,label_id].argsort()[::-1]: # 1 for negative class; 0 for positive class
        words_sorted_by_pmi.append(vocab._index_to_token['tokens'][i])
    

    print(f'Top {args.num_display} PMI words for the targeted class')
    for i in pmi_df[:, label_id].argsort()[-args.num_display:]:
        print(vocab._index_to_token['tokens'][i])

    pmi_rank = words_sorted_by_pmi
    # get pmi rank for triggers
    for word in args.evaluate_words:
        try:
            token_rank = pmi_rank.index(word)
            freq_in_class = count_matrix[vocab._token_to_index['tokens'][word]][label_id] 
            total_freq = count_matrix[vocab._token_to_index['tokens'][word]][0] +  count_matrix[vocab._token_to_index['tokens'][word]][1]
            print(word, freq_in_class, '/', total_freq, token_rank)
        except:
            print(word + ' not in vocab')


    
    
