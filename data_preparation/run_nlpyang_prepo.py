import argparse

from data_preparation.nlpyang_data_builder import tokenize
import os
import time
import subprocess


def dplp_interface(dplp_dir, xml_path):
    # first convert files in xml_path to files.conll in xml_path
    # then segmenter xml_path to segment
    # convert XML file to DPLP readable CONLL
    # cd DPLP
    command_convert = "python2 {}/convert.py {}".format(dplp_dir, xml_path)
    print("Calling {}".format(command_convert))
    print("in tokenzied dir, you will have .conll format from .xml")
    subprocess.call(command_convert, shell=True)

    # command_seg = "python2 {}/segmenter.py {} {}".format(dplp_dir, xml_path, seg_path)
    # print("Calling {}".format(command_seg))
    # subprocess.call(command_seg, shell=True)
    #
    # command_rst = "python2 {}/rstparser.py {} False".format(dplp_dir, seg_path)
    # print("Calling {}".format(command_rst))
    # subprocess.call(command_rst, shell=True)
    # python2 convert.py /datadrive/data/cnn/tokenized
    # python2 segmenter.py /datadrive/data/cnn/tokenized /datadrive/data/cnn/segs
    # python2 rstparser.py /datadrive/data/cnn/segs False


def dplp_rst_parser(dplp_dir, seg_path):
    os.chdir(dplp_dir)
    command_rst = "python2 {}/rstparser.py {} False".format(dplp_dir, seg_path)
    print("Calling {}".format(command_rst))
    subprocess.call(command_rst, shell=True)
    print("finish calling {}".format(command_rst))


def split_article_summary(path_of_stories, path_tgt_article, path_tgt_summary):
    print("Files must end with story")
    files = os.listdir(path_of_stories)
    files = [f for f in files if f.endswith('story')]

    for f in files:
        with open(os.path.join(path_of_stories, f), 'r') as fd:
            lines = fd.read().splitlines()
            lines = [l for l in lines if len(l) > 1]
            source = []
            tgt = []
            flag = False
            for l in lines:
                if '@highlight' in l:
                    flag = True
                    continue
                if (flag):
                    if len(l) < 2:
                        continue
                    tgt.append(l)
                    flag = False
                else:
                    if len(l) > 1:
                        source.append(l)
            if len("\n".join(source)) < 10 or len("\n".join(tgt)) < 10:
                print(f)
                continue
            with open(os.path.join(path_tgt_article, f + '.doc'), 'w') as farticle:
                source = source[:50]
                farticle.write("\n".join(source))
            with open(os.path.join(path_tgt_summary, f + '.sum'), 'w') as fsum:
                fsum.write("\n".join(tgt))


from data_preparation.nlpyang_data_builder import format_to_lines

if __name__ == '__main__':
    # tokenization
    # from data_preparation.nlpyang_prepo import tokenize

    parser = argparse.ArgumentParser()
    parser.add_argument("-mode",
                        default='format_to_bert',
                        # default='format_to_lines',
                        type=str,
                        help='tokenize or format_to_lines or format_to_bert')
    parser.add_argument("-oracle_mode",
                        default='greedy',
                        # default='combination',
                        type=str,
                        help='how to generate oracle summaries, greedy or combination, combination will generate more accurate oracles but take much longer time.')
    parser.add_argument("-map_path", default='/datadrive/DiscoBERT/data_preparation/urls_cnndm')
    parser.add_argument("-data_dir", default='/datadrive/data/cnn')
    parser.add_argument("-data_name", default='cnn')
    # parser.add_argument("-data_dir", default='/datadrive/data/dailymail')
    # parser.add_argument("-data_name", default='dailymail')

    parser.add_argument("-rel_raw_story_path", default='stories')
    parser.add_argument("-rel_split_doc_path", default='raw_doc')
    parser.add_argument("-rel_split_sum_path", default='sum')
    parser.add_argument("-rel_tok_path", default='tokenized')
    parser.add_argument('-rel_rst_seg_path', default='segs')
    parser.add_argument("-rel_save_path", default='chunk')
    parser.add_argument("-bert_model_name", default='roberta-base')
    parser.add_argument("-dplp_path", default="/datadrive/DPLP")
    parser.add_argument("-shard_size", default=1000, type=int)
    parser.add_argument('-min_nsents', default=3, type=int)
    parser.add_argument('-max_nsents', default=100, type=int)
    parser.add_argument('-min_src_ntokens', default=5, type=int)
    parser.add_argument('-max_src_ntokens', default=200, type=int)
    parser.add_argument('-oracle_sent_num', default=5, type=int)
    # parser.add_argument("-lower", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('-snlp_path', default='/home/cc/stanford-corenlp-full-2018-10-05')
    parser.add_argument('-log_file', default='../../logs/cnndm.log')

    parser.add_argument('-dataset', default='', help='train, valid or test, defaul will process all datasets')

    parser.add_argument('-n_cpus', default=10, type=int)

    args = parser.parse_args()
    # init_logger(args.log_file)

    if args.mode == 'split':
        if not os.path.exists(os.path.join(args.data_dir, args.rel_split_doc_path)):
            os.mkdir(os.path.join(args.data_dir, args.rel_split_doc_path))
        if not os.path.exists(os.path.join(args.data_dir, args.rel_split_sum_path)):
            os.mkdir(os.path.join(args.data_dir, args.rel_split_sum_path))

        # split doc and summary for cnndm
        split_article_summary(os.path.join(args.data_dir, args.rel_raw_story_path),
                              os.path.join(args.data_dir, args.rel_split_doc_path),
                              path_tgt_summary=os.path.join(args.data_dir, args.rel_split_sum_path))
        print("finishing spliting doc and sum.")
    elif args.mode == 'tokenize':
        tokenized_doc = os.path.join(args.data_dir, args.rel_tok_path)
        if not os.path.exists(tokenized_doc):
            os.mkdir(tokenized_doc)
        tokenize(raw_path=os.path.join(args.data_dir, args.rel_split_doc_path),
                 save_path=tokenized_doc,
                 snlp=args.snlp_path)
        print("finishing tokenization")
    elif args.mode == 'dplp':
        if not os.path.join(args.data_dir, args.rel_rst_seg_path):
            os.mkdir(os.path.join(args.data_dir, args.rel_rst_seg_path))

        dplp_interface(args.dplp_path,
                       os.path.join(args.data_dir, args.rel_tok_path)
                       )
    elif args.mode == 'rst':
        if not os.path.join(args.data_dir, args.rel_rst_seg_path):
            os.mkdir(os.path.join(args.data_dir, args.rel_rst_seg_path))

        dplp_rst_parser(args.dplp_path,
                        os.path.join(args.data_dir, args.rel_rst_seg_path)
                        )
    elif args.mode == 'format_to_lines':
        seg_path = os.path.join(args.data_dir, args.rel_rst_seg_path)
        tokenized_doc = os.path.join(args.data_dir, args.rel_tok_path)
        save_path = os.path.join(args.data_dir,
                                 args.rel_save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        summary_path = os.path.join(args.data_dir, args.rel_split_sum_path)
        format_to_lines(map_urls_path=args.map_path, seg_path=seg_path, tok_path=tokenized_doc,
                        shard_size=args.shard_size,
                        save_path=save_path, summary_path=summary_path, data_name=args.data_name)
    elif args.mode == 'format_to_bert':
        from data_preparation.nlpyang_data_builder import format_to_bert

        save_path = os.path.join(args.data_dir,
                                 args.rel_save_path)

        start_time = time.time()
        format_to_bert(save_path,
                       oracle_mode=args.oracle_mode,
                       oracle_sent_num=args.oracle_sent_num,
                       bert_model_name=args.bert_model_name)
        duration = time.time() - start_time
        print("Mode {}\tDuration {}".format(args.oracle_mode, duration))
