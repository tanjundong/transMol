import collections
import re
from typing import List, Dict
import glob


class SmilesTokenizer():

    ID_PAD = 0 #padding
    ID_SOS = 1 # start of smiles
    ID_EOS = 2 # end of smiles
    ID_MASK = 3 # end of smiles

    TOKEN_PAD = '_'
    TOKEN_SOS = '^'
    TOKEN_EOS = '$'
    TOKEN_MASK = '~'

    SPECIAL_TOKENS=[TOKEN_PAD, TOKEN_SOS,
                    TOKEN_EOS, TOKEN_MASK,
                    ]


    def __init__(self):

        self.vocab = dict()

        self.special_tokens = self.build_special_vocab()


        self.vocab.update(self.special_tokens)


    def build_special_vocab(self) -> Dict[str, int]:
        ret = dict()
        for i,name in enumerate(self.SPECIAL_TOKENS):
            ret[name] = i

        return ret




    @classmethod
    def build_from_corpus(cls,
                          path: str):
        counter = collections.defaultdict(int)
        f = open(path, 'r')
        for line in f.readlines():
            line = line.strip('\n')
            for char in line:
                val = counter.get(char, 0)
                counter[char] = val + 1

        obj = cls()
        special = obj.build_special_vocab()

        max_special_id = max(special.values())
        sorted_items = sorted(counter.items(), key=lambda x:x[1], reverse=True)


        obj.vocab.update(special)
        for i, item in enumerate(sorted_items):
            v = max_special_id + i +1
            k, _  = item
            obj.vocab[k] = v

        obj.init()

        return obj


    def init(self):
        self._id2char = dict()
        self._char2id = dict()
        for char, id in self.vocab.items():
            self._id2char[id] = char
            self._char2id[char] = id

        self.size = len(self._id2char.keys())



    def id2char(self, id):
        return self._id2char.get(id, self.TOKEN_MASK)

    def char2id(self, char):
        return self._char2id.get(char, self.ID_MASK)


    @classmethod
    def load(cls,
             path: str):

        obj = cls()
        obj.vocab = dict()
        #for i, token in enumerate(cls.SPECIAL_TOKENS):
        #    obj.vocab[token] = i

        f = open(path,'r')
        for t, line in enumerate(f.readlines()):
            line = line.strip('\n')
            #obj.vocab[t+i] = line
            obj.vocab[line] = t

        obj.init()
        f.close()
        return obj



    def smiles2ids(self,
                   smiles: str,
                   max_length: int):
        ret = [self.ID_PAD] * max_length
        ret[0] = self.ID_SOS
        for i, char in enumerate(smiles):
            i = i+1
            if i<max_length -1:
                ret[i] = self.char2id(char)

        if i<max_length-1:
            ret[i+1] = self.ID_EOS

        return ret


    def ids2smiles(self,
                   ids: List[int]) -> str:
        ret = []
        for id in ids:
            ret.append(self.id2char(id))

        return ''.join(ret)


    def dump(self,
             path: str):
        f = open(path, 'w')
        for i in range(self.size):
            char = self.id2char(i)
            f.write(char +'\n')
        f.close()

