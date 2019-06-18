/** You could basically use this config to train your own BERT classifier,
    with the following changes:
    1. change the `bert_model` variable to "bert-base-uncased" (or whichever you prefer)
    2. swap in your own DatasetReader. It should generate instances
       that look like {"tokens": TextField(...), "label": LabelField(...)}.
       You don't need to add the "[CLS]" or "[SEP]" tokens to your instances,
       that's handled automatically by the token indexer.
    3. replace train_data_path and validation_data_path with real paths
    4. any other parameters you want to change (e.g. dropout)
 */
local root ='/datadrive/GETSum/bert_data';
local cuda_device = 0;
local lr=1e-5;
local train_data_path =root+'/train/';

local valid_data_path =root+'/test/';
local test_data_path =root+'/test/';

local bert_config='/datadrive/GETSum/configs/BertSumConfig.json';

local BATCH_SIZE=13;
//local ser_dir=root+'/tmp/';
####
//local train_data_path = test_data_path;
###

local base_iterator={
//    type: 'bucket',
    type: 'basic',
//    track_epoch: true,
//    "sorting_keys": [["doc_text", "num_tokens"]],
    batch_size: BATCH_SIZE,
    instances_per_epoch:1000*BATCH_SIZE,
     max_instances_in_memory: BATCH_SIZE * 30,
  };

local base_iterator_unlimit={
    type: 'basic',
//    track_epoch: true,
//    "sorting_keys": [["doc_text", "num_tokens"]],
    batch_size: BATCH_SIZE+10,
  };
# For a real model you'd want to use "bert-base-uncased" or similar.
//local bert_model = "allennlp/tests/fixtures/bert/vocab.txt";
local bert_model = "bert-base-uncased";
local bert_vocab = "/datadrive/bert_vocab/vocabulary/bert-base-uncased-vocab.txt";
//local bert_vocab = "/datadrive/bert_vocab/";

{
    "dataset_reader": {
        "lazy": true,
        "type": "cnndm",
        "bert_model_name": "bert-base-uncased",
//        "tokenizer": {
//            "word_splitter": "bert-basic"
//        },
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": bert_model
            }
        }
    },
    "datasets_for_vocab_creation": [],
     vocabulary: {
    directory_path: '/datadrive/bert_vocab',
    extend: false,
  },
//
//    "vocabulary":bert_vocab,
    "train_data_path": train_data_path,
    "validation_data_path": valid_data_path,

    "model": {
        "type": "tensor_bert",
        "bert_model": bert_model,
        "bert_config_file":bert_config,
        "dropout": 0.1
    },
     iterator:
     base_iterator,
//     {type: 'multiprocess',
//    base_iterator: base_iterator,
//    num_workers: 1,
//    output_queue_size: 130},

    validation_iterator:{
    type: 'multiprocess',
    base_iterator: base_iterator_unlimit,
    num_workers: 1,
    output_queue_size: 130},
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": lr
        },
        "summary_interval":200,
        "keep_serialized_model_every_num_seconds":30*60,
        "validation_metric": "+R_1",
        "num_serialized_models_to_keep": 3,
        "num_epochs": 40,
//        "grad_norm": 10.0,
        "patience": 10,
        "cuda_device": cuda_device,
        "grad_clipping":5,
        "learning_rate_scheduler":{
        "type":"noam",
        'model_size':768,
        "warmup_steps":8000
        },
        "should_log_learning_rate":true,
    }
}