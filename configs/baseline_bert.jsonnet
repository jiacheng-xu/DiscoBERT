local util = import "utils.libsonnet";

//local debug=false;
local debug=true;

local max_bpe=768;
//local max_bpe=512;

//local cuda_device = 0;
//local cuda_device = 1;
//local cuda_device = 2;
local cuda_device = 3;

local bertsum_oracle=false;
//local bertsum_oracle=true;

local multi_orac=false;
//local multi_orac=true;

local BATCH_SIZE=6;

//local use_disco=true;
local use_disco=false;

local trigram_block=true;
//local trigram_block=false;

local dropout=0.2;
local num_of_batch_per_train_epo= if debug then 22 else  2088;


//local global_root = '/scratch/cluster/jcxu/GETSum';
//local root = '/scratch/cluster/jcxu/dailymail';

local global_root = '/datadrive/GETSum';
local root = '/datadrive/data/cnndm';


local min_pred_word=40;
local max_pred_word=130;

//local pred_len_min=5;
//local pred_len_max=9;
//local use_disco=true;


local use_disco_graph = false;
local use_coref=false;

//local use_disco_graph = false;
//local use_coref=true;


//local use_disco_graph = true;
//local use_coref=false;

//local agg_func=util.easy_graph_encoder;
local agg_func=util.gcn;

local train_data_path =root+'/train/';
local valid_data_path =root+'/test/';
local test_data_path =root+'/test/';

local bert_config=global_root+'/configs/BertSumConfig.json';


####
//local train_data_path = if debug then test_data_path else train_data_path;
###

local base_iterator={
//    type: 'bucket',
    type: 'basic',
//    track_epoch: true,
//    "sorting_keys": [["doc_text", "num_tokens"]],
    batch_size: BATCH_SIZE,
    instances_per_epoch:num_of_batch_per_train_epo*BATCH_SIZE,
     max_instances_in_memory: BATCH_SIZE * 30,
  };

local base_iterator_unlimit={
    type: 'basic',
//    track_epoch: true,
//    "sorting_keys": [["doc_text", "num_tokens"]],
    batch_size: BATCH_SIZE*12,
  };
local bert_model = "bert-base-uncased";
local bert_vocab = global_root+"/bert_vocab";

//local model_archive = "/datadrive/GETSum/tmp_expsx0o2m4hl";
//local model_archive = "/datadrive/GETSum/tmp_expscqpap81m";

//local model_archive = null;
{
//    "model-archive":model_archive,
    "dataset_reader": {
        "lazy": true,
        "type": "cnndm",
        "debug":debug,
        "bertsum_oracle":bertsum_oracle,
        "max_bpe":max_bpe,
        "bert_model_name": "bert-base-uncased",
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": bert_model
            }
        }
    },
//    "datasets_for_vocab_creation": [],
     vocabulary: {
    directory_path: bert_vocab,
    extend: false,
  },
//
    "train_data_path": train_data_path,
    "validation_data_path": valid_data_path,

    "model": {
        "type": "tensor_bert",
        "bert_model": bert_model,
        "bert_config_file":bert_config,
        "bert_max_length":max_bpe,
        "multi_orac":multi_orac,
        "trainable":util.bert_trainable,
        "dropout":dropout,
        "graph_encoder":agg_func,
//        "pred_length":pred_len,
        "use_disco":use_disco,
        "use_disco_graph":use_disco_graph,
        "use_coref":use_coref,
        "span_extractor":util.SelfAttnSpan,
        "trigram_block":trigram_block,
        "min_pred_word":min_pred_word,
        "max_pred_word":max_pred_word
//         "min_pred_length":pred_len_min,       # 4 for cnn
//        "max_pred_length":pred_len_max,        # 6 for cnn
    },
     "iterator":base_iterator,
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
//        "optimizer": {
//            "type":optimizer,
//            "lr": lr,
//            "warmup":warmup,
//        },
         "optimizer": {
            "type": util.optimizer,
            "lr": util.lr,
            "warmup":util.warmup,
            "t_total": 8000,
            "max_grad_norm": 1.0,
            "weight_decay": 0.01,
            "parameter_groups": [
              [["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0.0}],
            ],
        },
        "summary_interval":1000,
        "keep_serialized_model_every_num_seconds":30*60,
        "validation_metric": "+R_1",
        "num_serialized_models_to_keep": 3,
        "num_epochs": 50,
//        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": cuda_device,
        "grad_clipping":5,
//        "learning_rate_scheduler":{
//        "type":"noam",
//        'model_size':768,
//        "warmup_steps":8000
//        },
//    "learning_rate_scheduler": {
//            "type": "slanted_triangular",
//            "num_epochs": 15,
//            "num_steps_per_epoch": 8829,
//        },
        "should_log_learning_rate":true,
    }
}