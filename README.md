# :dancer:DiscoBERT: Discourse-Aware Neural Extractive Model for Text Summarization
Code repository for an ACL 2020 paper [Discourse-Aware Neural Extractive Model for Text Summarization](https://arxiv.org/abs/1910.14142). 

Authors: [Jiacheng Xu](http://www.cs.utexas.edu/~jcxu/) (University of Texas at Austin), [Zhe Gan](https://zhegan27.github.io), [Yu Cheng](https://sites.google.com/site/chengyu05/home), and Jingjing Liu (Microsoft Dynamics 365 AI Research).

Contact: jcxu at cs dot utexas dot edu

## Illustration

![Stage1](http://www.cs.utexas.edu/~jcxu/material/ACL20/gif1.gif=512*384)



## Prerequisites

The code is based on `AllenNLP`, The code is developed with `python 3`, `allennlp` and `pytorch>=1.0`. For more requierments, please check `requirements.txt`.

## Citing
```
@inproceedings{xu-etal-2020-discourse,
    title = {Discourse-Aware Neural Extractive Model for Text Summarization},
    author = {Xu, Jiacheng and Gan, Zhe and Cheng, Yu and Liu, Jingjing},
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    year = {2020},
    publisher = "Association for Computational Linguistics"
}
```

## Acknowledgements
* The data preprocessing (dataset handler, oracle creation, etc.) is partially based on [PreSumm](https://github.com/nlpyang/PreSumm) by Yang Liu and Mirella Lapata.
* Data preprocessing (tokenization, sentence split, coreference resolution etc.) used [CoreNLP](https://stanfordnlp.github.io/CoreNLP/). 
* RST Discourse Segmentation is generated from [NeuEDUSeg](https://github.com/PKU-TANGENT/NeuralEDUSeg). I slightly modified the code to run with GPU. Please check my modification [here](https://github.com/jiacheng-xu/NeuralEDUSeg).
* RST Discourse Parsing is generated from [DPLP](https://github.com/jiyfeng/DPLP). My customized version is [here](https://github.com/jiacheng-xu/DPLP) featuring batch implementation and remaining file detection. 
Empirically I found that `NeuEDUSeg` provided better segmentation output than `DPLP` so we use `NeuEDUSeg` for segmentation and `DPLP` for parsing.  