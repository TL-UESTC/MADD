import pymongo
import argparse
import jieba
from collections import defaultdict
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


parser = argparse.ArgumentParser(description='Doc2Vec Train')
parser.add_argument('--domain', action='append', help='domain names needed Doc2Vec')
parser.add_argument('--size', required=True, help='embedding size after Doc2Vec', type=int)
parser.add_argument('--Epoch', default=50, type=int)
parser.add_argument('--wordCutSave', action='store_true', help='whether to save words to DB after jieba')
args = parser.parse_args()

def is_chinese(uchar):
    """is this a chinese word?"""
    return True if uchar >= u'\u4e00' and uchar <= u'\u9fa5' else False
def is_number(uchar):
    """is this unicode a number?"""
    return True if uchar >= u'\u0030' and uchar <= u'\u0039' else False
def is_alphabet(uchar):
    """is this unicode an English word?"""
    return True if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a') else False

def format_str(content, lag):
    content_str = ''
    if lag == 0:  # English
        for i in content:
            if is_alphabet(i): content_str += i
    if lag == 1:  # Chinese
        for i in content:
            if is_chinese(i): content_str += i
    if lag == 2:  # Number
        for i in content:
            if is_number(i): content_str += i
    return content_str

class ReviewEmbedding():
    def __init__(self):
        self.userDoc = {}
        self.itemDoc = {}
    def connectDB(self):
        client = pymongo.MongoClient('localhost', 27017)
        db = client['MADD']
        collection = db[domain + 's']
        return collection
    def docStatictis(self, dbData):
        # nltk.download('stopwords')
        for record in dbData:
            Content = record[labels[domain][-1]]
            Cleaned = format_str(Content, 1)
            words = list(jieba.cut(Cleaned, cut_all=False))
            if self.userDoc.get(record[labels[domain][0]], -1) != -1:
                self.userDoc[record[labels[domain][0]]] += words
            else:
                self.userDoc[record[labels[domain][0]]] = words

            if self.itemDoc.get(record[labels[domain][1]], -1) != -1:
                self.itemDoc[record[labels[domain][1]]] += words
            else:
                self.itemDoc[record[labels[domain][1]]] = words

        self.userDoc = dict(sorted(self.userDoc.items(), key=lambda x: x[0]))
        self.itemDoc = dict(sorted(self.itemDoc.items(), key=lambda x: x[0]))

        frequencyu = defaultdict(int)
        for text in self.userDoc.values():
            for token in text:
                frequencyu[token] += 1
        textsu = [[token for token in text if frequencyu[token] > 1] for text in self.userDoc.values()]
        if args.wordCutSave:
            for index, key in enumerate(self.userDoc.keys()):
                self.userDoc[key] = textsu[index]
            data = []
            for key, item in self.userDoc.items():
                data.append({str(key):','.join(item)})
            client = pymongo.MongoClient('localhost', 27017)
            db = client['MADD']
            Collection = db[domain+'Words_users']
            Collection.insert_many(data)
            print(f'Words belonging to users in domain {domain} insert to DB finish!')

        frequencym = defaultdict(int)
        for text in self.itemDoc.values():
            for token in text:
                frequencym[token] += 1
        textsm = [[token for token in text if frequencym[token] > 1] for text in self.itemDoc.values()]
        if args.wordCutSave:
            for index, key in enumerate(self.itemDoc.keys()):
                self.itemDoc[key] = textsm[index]
            data1 = []
            for key, item in self.itemDoc.items():
                data1.append({str(key):','.join(item)})
            client = pymongo.MongoClient('localhost', 27017)
            db = client['MADD']
            Collection1 = db[domain+'Words_items']
            Collection1.insert_many(data1)
            print(f'Words belonging to items in domain {domain} insert to DB finish!')
        return textsu, textsm

    def train(self, text, size, mode):
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(text)]
        model = Doc2Vec(documents, vector_size=size, window=2, min_count=5, negative=5, workers=6)
        model.train(documents, total_examples=model.corpus_count, epochs=args.Epoch)
        model.save("./doc2vecdata/Doc2vec_Domain{}_{}_V{}.model".format(domain.title(), "users" if not mode else domain+'s', size))
    def main(self, size): # 0 user 1 item
        collection = self.connectDB()
        textsu, textsm = self.docStatictis(collection.find())
        self.train(textsu, size, 0)
        self.train(textsm, size, 1)

if __name__ == '__main__':
    labels = {'book' : ['uID', 'bID', 'rating', 'ubContent'],
              'movie' : ['uID', 'mID', 'rating', 'umContent'],
              'music' : ['uID', 'muID', 'rating', 'muContent']}
    for domain in args.domain:
        r = ReviewEmbedding()
        r.main(args.size)