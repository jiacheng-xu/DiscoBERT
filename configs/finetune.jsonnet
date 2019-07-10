
local debug=false;
local lr=1e-5;
local dropout=0.2;
local num_of_batch_per_train_epo=500;
local global_root = '/datadrive/GETSum';

//local root =global_root+'/bert_data';
local root = '/datadrive/data/cnn';
local cuda_device = 1;
local bert_trainable=true;
local optimizer="bert_adam";
local pred_len=6;
local warmup=0.1;
local iden = {type:"identity"};
local gcn ={type:"gcn", input_dims:[768], num_layers:1,hidden_dims:[768]};
local lstm={type:"seq2seq",
    "seq2seq_encoder":{
     "type":"lstm",
     "input_size":768,
     "hidden_size":384,
     "batch_first":true,
     "bidirectional":true
     }};
local stacked_self_attention={
     type:"seq2seq",
     seq2seq_encoder:{
     type:"stacked_self_attention",
        input_dim:768,
        hidden_dim:768,
        projection_dim:768,
        feedforward_hidden_dim:768,
        num_layers:2,
        num_attention_heads:4
     }
};
local multi_head_self_attention={
    type:"seq2seq",
    "seq2seq_encoder":{
    type:"multi_head_self_attention",
    num_heads:4,
    input_dim:768,
    attention_dim:128,
    values_dim:128,
    output_projection_dim:768
}};

local SelfAttnSpan={
    type:'self_attentive',
    input_dim:768

};

local agg_func=iden;


local train_data_path =root+'/train/';
local valid_data_path =root+'/test/';
local test_data_path =root+'/test/';

local bert_config=global_root+'/configs/BertSumConfig.json';

local BATCH_SIZE=11;
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
    instances_per_epoch:num_of_batch_per_train_epo*BATCH_SIZE,
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
local bert_vocab = global_root+"/bert_vocab";

local model_archive = "/datadrive/GETSum/tmp_expsx0o2m4hl";
//local model_archive = null;
{
//    "model-archive":model_archive,
    "dataset_reader": {
        "lazy": true,
        "type": "cnndm",
        "debug":debug,
        "bert_model_name": "bert-base-uncased",
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": bert_model
            }
        }
    },
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
        "trainable":bert_trainable,
        "dropout":dropout,
        "graph_encoder":agg_func,
//        "pred_length":pred_len,
        "use_disco":true,
        "use_coref":false,
        "span_extractor":SelfAttnSpan,
        "min_pred_length":4,
        "max_pred_length":6,
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
            "type": optimizer,
            "lr": lr,
            "warmup":warmup,
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
        "num_epochs": 40,
//        "grad_norm": 10.0,
        "patience": 10,
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