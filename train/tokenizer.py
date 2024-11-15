from tokenizers import Tokenizer, PreTokenizedString, NormalizedString
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import PreTokenizer
import tokenizers.normalizers as normalizers
from tokenizers.normalizers import NFD
import os

class XMLPreTokenizer:
    def split_XML(self, i: int, normalized_str: NormalizedString):
        tokens = []  # Tokens after pre-tokenizing
        curr_words = []
        pos = 0
        for char in str(normalized_str):
            if char == '<':
                if curr_words:
                    # tokens.append(("".join(curr_words), (pos - len(curr_words), pos)))
                    tokens.append("".join(curr_words))
                curr_words = []
                curr_words.append(char)
            elif char == '>':
                curr_words.append(char)
                # tokens.append(("".join(curr_words), (pos + 1 - len(curr_words), pos)))
                tokens.append("".join(curr_words))
                curr_words = []
            else:
                curr_words.append(char)
            pos += 1

        if curr_words:
            # tokens.append(("".join(curr_words), (pos - len(curr_words), pos)))
            tokens.append("".join(curr_words))
        
        tokens = list(map(NormalizedString, tokens))
        return tokens
    
    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.split_XML)

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
normalizer = normalizers.Sequence([NFD()])
pre_tokenizer = PreTokenizer.custom(XMLPreTokenizer())

tokenizer.normalizer = normalizer
tokenizer.pre_tokenizer = pre_tokenizer

trainer = BpeTrainer()

path = r"..\train_dataset\xml"
files = os.listdir(path)

tokenizer.train(files=[os.path.join(path, x) for x in files], trainer=trainer)
encoded = tokenizer.encode("<measure number=\"10380\">")

print(encoded.tokens)

tokenizer.pre_tokenizer = None  # Set custom pretokenizer to None in order to save
tokenizer.save(r"..\model\tokenizer.json")
