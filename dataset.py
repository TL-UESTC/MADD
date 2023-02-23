import argparse
import pymongo
from torch.utils.data import Dataset
import numpy as np

class dualDataSet(Dataset):
    def __init__(self, uID, iID, u_embedding, i_embedding, rating):
        self.uID = uID
        self.iID = iID
        self.u_embedding = u_embedding
        self.i_embedding = i_embedding
        self.rating = rating
    def __getitem__(self, index):
        return self.uID[index], self.iID[index], self.u_embedding[index], self.i_embedding[index], self.rating[index]
    def __len__(self):
        return len(self.rating)

def splitTrainTest(data):
    data = sorted(data, key=lambda x: (x[0], x[2]))
    train, test = [], []
    for i in range(len(data)-1):
        if int(data[i][0]) != int(data[i+1][0]):
            test.append(data[i])
        else:
            train.append(data[i])
    test.append(data[-1])
    return np.array(train), np.array(test)

def getEmbeddings(data, size):
    users, items = {}, {}
    for i in range(data.shape[0]):
        users[data[i][0]] = data[i][3: 3+size*2]
        items[data[i][1]] = data[i][3+size*2:]
    return users, items

# ./testdata  generate testNeg.npy
class genTestNeg():
    def __init__(self, args):
        self.domainA = args.domainA.title()
        self.domainB = args.domainB.title()
        self.negNum = args.negNum
        self.gentargetneg = args.gentargetneg.title()
    def getTrainDict(self, trainData):
        trainDict = []
        for row in trainData:
            trainDict.append((int(row[0]), int(row[1])))
        return trainDict
    def getTestNeg(self, trainData, testData):
        user, item = [], []
        trainDict = self.getTrainDict(trainData)
        trainitems, testitems = trainData[:, 1], testData[:, 1]
        minvalue, maxvalue = min(trainitems.min(), testitems.min()), max(trainitems.max(), testitems.max())
        for s in testData:
            tmp_user, tmp_item = [], []
            u, i = int(s[0]), int(s[1])
            tmp_user.append(u)
            tmp_item.append(i)
            neglist = set()
            neglist.add(i)
            for t in range(self.negNum):
                j = np.random.randint(minvalue, maxvalue+1)
                while (u, j) in trainDict or j in neglist or j not in trainitems or j not in testitems:
                    j = np.random.randint(minvalue, maxvalue+1)
                neglist.add(j)
                tmp_user.append(u)
                tmp_item.append(j)
            user.append(tmp_user)
            item.append(tmp_item)
        np.save('./testdata/{}{}_{}TestNeg.npy'.format(self.domainA, self.domainB, self.gentargetneg.title), np.array([np.array(user), np.array(item)]))

class txtDataToDB():
    def __init__(self, args):
        self.bookpath = args.bookpath
        self.moviepath = args.moviepath
        self.musicpath = args.musicpath
        self.dbname = args.dbname
        self.collectionbook = args.collectionbook
        self.collectionmovie = args.collectionmovie
        self.collectionmusic = args.collectionmusic

    def bookTxtClean(self):
        cleanDate = []
        with open(self.bookpath) as fp:
            for line in fp.readlines():
                uID, bookID, rating, detail, ubContent, *_ = map(lambda x: \
                                                        int(x[1:-1]) if x[1:-1].isdigit() else x[1:-1], line.split('\t'))
                dirtyWord = ['补签推荐', '补签力荐', '补签可读', '补签经典', '补签'] # raw data里面的脏数据
                for word in dirtyWord:
                    if word in ubContent: ubContent = ubContent.replace(word, '')
                if ubContent != '': cleanDate.append([uID, bookID, rating, str(ubContent).strip()])
        return cleanDate[1:]

    def moviesTxtClean(self):
        cleanDate = []
        with open(self.moviepath) as fp:
            for line in fp.readlines()[1:]:
                try:
                    uID, mID, rating, umContent, *_ = map(lambda x: \
                                                    int(x[1:-1]) if x[1:-1].isdigit() else x[1:-1], line.split('\t'))
                except: continue
                if type(uID) == str:
                    cleanDate[-1][-1] += uID
                    continue
                if type(uID)==int and type(mID)==int and umContent != '': cleanDate.append([uID, mID, rating, str(umContent).strip()])
        return cleanDate

    def musicTxtClean(self):
        cleanDate = []
        with open(self.musicpath) as fp:
            for line in fp.readlines():
                try:
                    uID, mID, rating, labels, umComment, *_ = map(lambda x: \
                                                    int(x[1:-1]) if x[1:-1].isdigit() else x[1:-1], line.split('\t'))
                except: continue
                if umComment != '': cleanDate.append([uID, mID, rating, str(umComment).strip()])
        return cleanDate[1:]

    def saveToDB(self, booklabel, bookdata, movielabel, moviedata, musiclabel, musicdata):
        books = [dict(zip(booklabel, record)) for record in bookdata]
        movies = [dict(zip(movielabel, record)) for record in moviedata]
        musics = [dict(zip(musiclabel, record)) for record in musicdata]
        client = pymongo.MongoClient('localhost', 27017)
        db = client[self.dbname]
        # books 数据插入数据库
        CollectionBook = db[self.collectionbook]
        CollectionBook.insert_many(books)
        # movies 数据插入数据库
        CollectionMoive = db[self.collectionmovie]
        CollectionMoive.insert_many(movies)
        # music 数据插入数据库
        CollectionMusic = db[self.collectionmusic]
        CollectionMusic.insert_many(musics)

    def initDBData(self):
        booklabel, movielabel, musiclabel = ['uID', 'bID', 'rating', 'ubContent'], ['uID', 'mID', 'rating', 'umContent'], ['uID', 'muID', 'rating', 'muContent']
        bookdata, moviedata, musicdata = self.bookTxtClean(), self.moviesTxtClean(), self.musicTxtClean()
        self.saveToDB(booklabel, bookdata, movielabel, moviedata, musiclabel, musicdata)
        print("----------------------------------------------------------\n"
              "MongoDB数据生成完成！ 其中：\n  books: {} 名用户对 {} 本书进行了 {} 条评论,\n  movies: {} 名用户对 {} 部电影进行了 {} 条评论, \n  musics: {} 名用户对 {} 个音乐进行了 {} 条评论"
              "\n----------------------------------------------------------"
              .format(len(set([_[0] for _ in bookdata])),len(set([_[1] for _ in bookdata])),len(bookdata),
                   len(set([_[0] for _ in moviedata])),len(set([_[1] for _ in moviedata])),len(moviedata),
                      len(set([_[0] for _ in musicdata])), len(set([_[1] for _ in musicdata])), len(musicdata)))

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--bookpath', default='./data/bookreviews_cleaned.txt', help="domain book raw txt path")
    parser.add_argument('--moviepath', default='./data/moviereviews_cleaned.txt', help="domain movie raw txt path")
    parser.add_argument('--musicpath', default='./data/musicreviews_cleaned.txt', help="domain music raw txt path")
    parser.add_argument('--dbname', default='MADD', help='please input mongo database name')
    parser.add_argument('--collectionbook', default='books', help='please input collection book name')
    parser.add_argument('--collectionmovie', default='movies', help='please input collection movie name')
    parser.add_argument('--collectionmusic', default='musics', help='please input collection music name')
    parser.add_argument('--gentargetneg', default='', help='which domain needs to generate testneg.')
    parser.add_argument('--negNum', default=99, help='len of recommendataion list')
    args = parser.parse_args()
    txtToDb = txtDataToDB(args)
    txtToDb.initDBData()

if __name__ == '__main__':
    main()