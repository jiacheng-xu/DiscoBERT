{

matrix_attn:{
        type:"linear",
        combination:'x,y,x*y,x+y,x-y',
//        activation:"sigmoid",
        tensor_1_dim:768,
        tensor_2_dim:768,
        },

fusion_feedforward:{
                'input_dim': 768*2,
                'hidden_dims': 768,
                'activations': ['linear'],
                'dropout':0.2,
                'num_layers': 1},
         gcn:{type:"gcn",
        hdim:768,
        nlayers:2},
   SelfAttnSpan:{
    type:'self_attentive',
    input_dim:768
}
}