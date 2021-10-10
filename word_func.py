from konlpy.tag import Okt
import re
import numpy as np
import pandas as pd


# DTM Class
class DTM:
    def __init__(self):
        self.okt = Okt()
        self.vocabulary = None

    def dtm_build(self, comments):
        comments_remake = []
        col_word = set()

        # 형태소 및 단어에서 단어 글자 크기가 1 이하이면 제외(sklearn과 동일하게 작동하도록 함)
        for com in comments:
            comments_remake.append([])
            tmp = com.split()

            for word in tmp:

                word_re = self.okt.pos(word)
                if len(word_re[0][0]) < 2:
                    continue
                col_word.add(word_re[0][0])
                comments_remake[-1].append(word_re[0][0])
        
        col_word = list(col_word)

        dtm = np.zeros(shape=(len(comments_remake), len(col_word)), dtype=np.int32) # row = comments, col = set of unique voca
        
        for i in range(len(comments_remake)):
            comment = comments_remake[i]
            for j in range(len(comment)):
                word = comment[j]
                if word in col_word:
                    idx = col_word.index(word)
                    dtm[i, idx] += 1

        self.vocabulary = {word : i for i, word in enumerate(col_word)}
        return dtm

    def tfidf_score(self, dtm):

        def tf(dtm):
            result = []
            for line in dtm:
                result.append(0.5+ (0.5 * line / np.max(line)))
            print('tf: ', result)
            return np.array(result)
        
        def idf(dtm):
            numerator = len(dtm)
            denominator = 1 + np.count_nonzero(dtm, axis=0)
            print('de: ', np.log10(numerator/denominator))
            return np.log10(numerator/denominator)
            
        return tf(dtm) * idf(dtm)


def make_corpus(path):
    df = read_file(path)

    authors = set(['@' + re.sub('\W+', ' ', i) for i in df['author']])
    authors_regex = "|".join(authors)
    
    # author processing
    for line in df['comment']:
        line = re.sub('\W+',' ', line)  
        line = re.sub(authors_regex, ' ', line)

    # comment processing
    subs_expr = [authors_regex, '\W+', '\d+', '\n', '[\[\]]', '[a-zA-Z]', '[ㄱ-ㅎ | ㅏ-ㅣ]', '\s+']
    comments = []

    for line in df['comment']:
        for expr in subs_expr:
            line = re.sub(expr, ' ', line)
        
        # 정제한 후, 스페이스만 남아있으면 제외함
        f_line = line.strip()
        if len(f_line) > 0:
            comments.append(f_line) 
    
    return comments


def read_file(path):
    df = pd.read_excel(path)
    df.fillna(' ', inplace=True)
    return df


    