{
matrix_attn:{
        type:"linear",
        combination:'x,y,x*y,x+y,x-y',
//        activation:"sigmoid",
        tensor_1_dim:1024,
        tensor_2_dim:1024,
        },
fusion_feedforward:{
                'input_dim': 1024*2,
                'hidden_dims': 1024,
                'activations': ['linear'],
                'dropout':0.2,
                'num_layers': 1},
                gcn:{type:"gcn",
        hdim:1024,
        nlayers:2},
SelfAttnSpan:{
    type:'self_attentive',
    input_dim:1024
}
}