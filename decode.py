# External dependencies
import argparse as arg
import pprint

# External dependencies
import os.path
import gensim.models as gs
import numpy as np
import gensim.models.keyedvectors as word2vec
import pickle as pk
from gensim import matutils
from datetime import datetime

class CliParser:
    """Parameter parser for all arguments"""
    def __init__(self):
        self.parser = arg.ArgumentParser(description='Parser for decoding/scoring emoji2vec model')

        # Directories/files
        self.parser.add_argument('-w', '--word',
                                 type=str, help='path to the word embeddings file')

        self.parser.add_argument('-c', '--cache',
                                 type=str, help='file for cached phrase embeddings')
        
        self.parser.add_argument('-m', '--model', default='emoji2vec.bin',
                                 type=str, help='path to the model embeddings file')

        self.parser.add_argument('-s', '--stop',
                                 type=str, help='path to the stop words list file')

        self.parser.add_argument('-l', '--interactive', action='store_true',
                                 help='interactive mode; requires word embeddings file')

        self.parser.add_argument('-i', '--input',
                                 type=str, help='path to the input data file for decoding/scoring')
        
        self.parser.add_argument('-o', '--output',
                                 type=str, help='path to the output file')
        
        self.parser.add_argument('-t', '--threshold', default=0.5, type=float,
                                 help='threshold for binary classification')

        self.parser.add_argument('-n', '--topn', default=2, type=int,
                                 help='binary classification based on top-N hypothesis')
        
        args = self.parser.parse_args()

        # file for generated embeddings
        self.model_file = args.model

        # word2vec file
        self.word2vec_file = args.word

        # stop words list
        self.stop_list_file = args.stop if args.stop is not None else None

        # input and output
        self.interactive = args.interactive
        self.input_file  = args.input if args.input   is not None else None
        self.output_file = args.output if args.output is not None else None
        self.cache_file  = args.cache  if args.cache  is not None else None

        #top-N for scoring
        self.topn = args.topn


class LRDecoder:
    stopList = list()
    
    def __init__(self, modelfile, w2vfile, cachefile=None):
        try:
            if w2vfile is None and cachefile is None:
                raise Exception('Either word2vec file or phrase embeddings cache file has to be provided')
            
            if modelfile is None or not os.path.exists(modelfile):
                raise(Exception('Model file {} is missing'.format(modelfile)))

            self.p2v = None
            self.w2v = None
            self.w2v_dim = -1
            if cachefile is not None and os.path.exists(cachefile):
                self.p2v = dict()
                print('Loading word embeddings from cache file %s ...' % cachefile)
                self.p2v = pk.load(open(cachefile, 'rb'))
                self.w2v_dim = len(list(self.p2v.values())[0])
            else:
                print('Loading word embeddings file %s ...' % w2vfile)
                is_binary = not (w2vfile.lower().endswith('.txt'))
                self.w2v = word2vec.KeyedVectors.load_word2vec_format(w2vfile, binary=is_binary)
                self.w2v_dim = self.w2v.vector_size
                print('...finished reading {} words of {} dimensions'.format(len(self.w2v.vocab), self.w2v_dim))

            print('Loading label embeddings file %s ...' % modelfile)
            is_binary = not (modelfile.lower().endswith('.txt'))
            self.model = word2vec.KeyedVectors.load_word2vec_format(modelfile, binary=is_binary)
            self.num_labels = len(self.model.vocab)
            self.model_dim = self.model.vector_size
            print('...finished reading {} labels of {} dimensions'.format(self.num_labels, self.model_dim))

            if(self.w2v_dim != self.model_dim):
                raise(Exception('Dimesion of word embeddings ({}) should match that of the model ({})'.format(self.w2v_dim, self.model_dim)))

            self.dim = self.w2v_dim

        except Exception as e:
            print(e)
            exit()

            
    def __sigmoid(self, x):
        return 1 / (1 + np.math.exp(-x))


    def read_stop_list(self, stopListFile):
        try:
            with open(stopListFile, 'r', encoding='utf-8') as fp:
                self.stopList = fp.read().splitlines()
        except Exception as e:
            print(e)
            exit()
            
    def process(self, phrase, topN=2, threshold=-1):
        try:
            phr_sum = None
            input_text = ''            
            if self.p2v is not None:
                if phrase in self.p2v:
                    phr_sum = self.p2v[phrase]
                else:
                    raise Exception('Phrase %s is not present in the cache -- need to regenerate cache' % phrase)
            else:
                tokens = phrase.split(' ')
                phr_sum = np.zeros(self.dim, np.float32)
                num_valid = 0
                
                for token in tokens:
                    if (token.lower() not in self.stopList) and (token in self.w2v):
                        phr_sum += self.w2v[token]
                        if num_valid > 0:
                            input_text += ' '
                        num_valid += 1
                        input_text += token

                if num_valid == 0:
                    return (None, '', '')

            scores = list()

            for label in self.model.vocab:
                prob = self.__sigmoid(np.dot(matutils.unitvec(phr_sum), matutils.unitvec(self.model[label])))
                scores.append({'label' : label, 'prob': prob})

            scores.sort(key=lambda elem: elem['prob'], reverse=True)

            hyp_labels = dict()
            hypstr = ''
            for i in range(topN):
                prob = scores[i]['prob']
                label = scores[i]['label']
                if threshold is not None and prob < threshold:
                    break
                hyp_labels[label] = prob
                if hypstr != '':
                    hypstr += "\t"
                hypstr += '{}:{:.2}'.format(label, prob)
                
            return(hyp_labels, input_text, hypstr)
        
        except Exception as e:
            print(e)
            exit()

if __name__ == '__main__':
    # Setup
    args = CliParser()

    if args.interactive and args.word2vec_file is None:
        raise Exception('Word embeddings file has to be specified in interactive mode')
    
    # Initialize decoder
    decoder = LRDecoder(args.model_file, args.word2vec_file, args.cache_file)

    if args.stop_list_file is not None:
        decoder.read_stop_list(args.stop_list_file)

    if args.interactive:
        while 1:
            text = input('Please enter sentence: ')
            text = text.rstrip().lower()
            if text in ['exit', 'done', 'stop']:
                break
            _, instr, hypstr = decoder.process(text, args.topn)
            print('\nIcons for \"%s\": %s' % (instr, hypstr))
            print('\n')
            
    else:
        ofp = open(args.output_file, "w", encoding='utf-8') if args.output_file is not None else None

        __np =  0
        __nn =  0
        __tp =  0
        __fp =  0
        __fn =  0
        __tn =  0
        __tot = 0
        __cor = 0

        times = list()
        noScore = False
        
        #open input file
        with open(args.input_file, "r", encoding='utf-8') as ifp:

            lines = ifp.readlines()
            for line in lines:
                comps = list()
                comps = line.rstrip().split('\t')
                phrase, label, truthStr = (comps[0], None, None)
                if len(comps) == 3:
                    label = comps[1]
                    truthStr = comps[2]
                else:
                    noScore = True
                tstart = datetime.now()
                hypdict, _, hypstr = decoder.process(phrase, args.topn)
                delta = datetime.now() - tstart
                tmsec = delta.total_seconds() * 1000
                times.append(tmsec)
                nplist = np.array(times)
                time_mean = np.mean(nplist, dtype=np.float64).item()
                time_95p  = np.percentile(nplist, 95).item()

                truth = (truthStr == 'True')
                hypothesis = (hypdict is not None) and (label is not None) and (label in hypdict)

                if ofp is not None:
                    if label is None:
                        ofp.write(str.format('{}\t{}\n', line.rstrip(), hypstr))
                        continue
                    else:
                        ofp.write(str.format('{} {}\t{}\n', line.rstrip(), hypothesis, hypstr))
                else:
                    if label is None:
                        print(str.format('{}\t{}\n', line.rstrip(), hypstr))
                        continue

                __tot += 1
                if truth == True:
                    __np += 1
                    if hypothesis == truth:
                        __cor += 1
                        __tp += 1
                    else:
                        __fn += 1
                else:
                    __nn += 1
                    if hypothesis == truth:
                        __cor += 1
                        __tn += 1
                    else:
                        __fp += 1

        if ofp is not None:
            ofp.close()

        if noScore == False:
            print(str.format('\nP@{}: {:.2}, Tot:{}, Corr: {}, NP:{}, NN:{}, TP:{}, TN:{}, FP:{}, FN:{}', args.topn, (__cor/__tot), __tot, __cor, __np, __nn, __tp, __tn, __fp, __fn))
        print("\nTime(mean): %.2f  Time(95p): %.2f\n" % (time_mean, time_95p))
        
