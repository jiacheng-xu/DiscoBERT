#!/usr/bin/env bash

data_name='dailymail'
data_name='cnn'
#data_name='nyt'

#home_dir='/datadrive'
home_dir='/scratch/cluster/jcxu'

segs='segs'
tokenized='tokenized'
chunk='chunk'


#discobert='/datadrive/DiscoBERT'
#neueduseg='/datadrive/NeuralEDUSeg/src'

discobert='/scratch/cluster/jcxu/DiscoBERT'
neueduseg='/scratch/cluster/jcxu/NeuralEDUSeg/src'

url_path="$discobert/data_preparation/urls_$data_name" # if you are using the nyt
# or
url_path="$discobert/data_preparation/urls_cnndm"    # if you are using the joint cnndm

tokenizer_name="roberta-large"   # for tokenizer, roberta-large = roberta-base, bert–large-uncased = bert-base-uncased. So it does not make a diff
#tokenizer_name="bert-large-uncased"

#parser.add_argument("-map_path", default='/datadrive/DiscoBERT/data_preparation/urls_cnndm')
cd $discobert

PYTHONPATH=./ python3 data_preparation/run_nlpyang_prepo.py -mode split -data_dir "$home_dir/data_intern/$data_name" -rel_split_doc_path raw_doc -rel_split_sum_path sum


PYTHONPATH=./ python3 data_preparation/run_nlpyang_prepo.py -mode tokenize -data_dir "$home_dir/data_intern/$data_name" -rel_split_doc_path raw_doc -rel_tok_path $tokenized -snlp_path  "$home_dir/stanford-corenlp-full-2018-10-05"

# DPLP convert to CONLL format
PYTHONPATH=./ python3 data_preparation/run_nlpyang_prepo.py -mode dplp -data_dir "$home_dir/data_intern/$data_name"  -dplp_path "$home_dir/DPLP" -rel_rst_seg_path $segs -rel_tok_path $tokenized

cd $neueduseg
CUDA_VISIBLE_DEVICES=3  python run.py --segment --input_conll_path "$home_dir/data_intern/$data_name/$tokenized"  --output_merge_conll_path "$home_dir/data/$data_name/$segs"  --gpu 0
CUDA_VISIBLE_DEVICES=5  python3 run.py --segment --input_conll_path "$home_dir/data_intern/$data_name/$tokenized"  --output_merge_conll_path "$home_dir/data/$data_name/$segs"  --gpu 0
CUDA_VISIBLE_DEVICES=6  python3 run.py --segment --input_conll_path "$home_dir/data_intern/$data_name/$tokenized"  --output_merge_conll_path "$home_dir/data/$data_name/$segs"  --gpu 0
CUDA_VISIBLE_DEVICES=7  python3 run.py --segment --input_conll_path "$home_dir/data_intern/$data_name/$tokenized"  --output_merge_conll_path "$home_dir/data/$data_name/$segs"  --gpu 0


cd $discobert
# RUN DPLP for RST parse
PYTHONPATH=./  python3 "data_preparation/run_nlpyang_prepo.py" -mode rst -data_dir "$home_dir/data_intern/$data_name"  -dplp_path "$home_dir/DPLP" -rel_rst_seg_path segs

# format to lines

cd $discobert
PYTHONPATH=./  python3 "data_preparation/run_nlpyang_prepo.py" \
-mode format_to_lines -data_dir "$home_dir/data_intern/$data_name"  \
-rel_rst_seg_path $segs -rel_tok_path $tokenized -rel_save_path $chunk -rel_split_sum_path sum -data_name $data_name -map_path $url_path

# format to bert

PYTHONPATH=./  python3 "data_preparation/run_nlpyang_prepo.py" \
-mode format_to_bert -data_dir "$home_dir/data_intern/$data_name" -bert_model_name $tokenizer_name -rel_save_path $chunk
#-rel_rst_seg_path $segs -rel_tok_path $tokenized  -rel_split_sum_path sum -data_name $data_name -map_path $url_path

