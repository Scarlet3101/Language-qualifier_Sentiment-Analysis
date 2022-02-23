import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.ensemble import VotingClassifier
from deep_translator import GoogleTranslator
import string
import joblib

class LanguageDetect_and_SentimentAnalysis:
    def __init__(self):
        self.tfidf = joblib.load('models/tfidf.sav')
        self.model_sa = joblib.load('models/model.sav')
        self.model_ld = joblib.load('models/done_language_model.sav')
        
        self.lmtzr = WordNetLemmatizer()
        
        list_of_stop_words = []
        for stop in ['english', 'french','dutch', 'spanish', 'greek', 'russian', 'danish', 'italian', 'turkish', 'german']:
            list_of_stop_words.append(stopwords.words(stop))
        self.stops = [element for sub in list_of_stop_words for element in sub]
        
        print('Sentiment Analysis is Ready!')

    def make_pre_processing_sa(self, df_train):
        #rewrite in lowercase
        df_train['Text'] = df_train['Text'].apply(lambda x: x.lower())

        #get rid of any characters apart from alphabets
        pattern = r"[^a-z ]"
        df_train['Text'] = df_train['Text'].apply(lambda x: ' '.join( nltk.word_tokenize(re.sub(pattern, " ", x).strip()) ))

        #remove stop words
        stop_words = stopwords.words('english')
        df_train['Text'] = df_train['Text'].apply(lambda x: ' '.join([word for word in nltk.word_tokenize(x) if word not in (stop_words)]))

        #text lemmatization
        df_train['Text'] = df_train['Text'].apply(lambda x: ' '.join([self.lmtzr.lemmatize(i, wordnet.VERB) for i in nltk.word_tokenize(x)]))

        #drop duplicate values (keep first)
        df_train.drop_duplicates(keep="first", inplace = True)

        return df_train
    
    def make_pre_processing_ld(self, text):
        text = str(text).lower() # lowercase
        text = re.sub(r'[{}]'.format(string.punctuation), '', text)
        text = re.sub(r'\d+', '', text)

        return ' '.join([self.lmtzr.lemmatize(word,wordnet.VERB) for word in nltk.word_tokenize(text) if word.lower() not in self.stops])
    
    def SentimentAnalysis(self, txt):
        df_input = pd.DataFrame([txt], columns=['Text'])
        df_input = self.make_pre_processing_sa(df_input)
        
        input_test = self.tfidf.transform(df_input.Text)
        return self.model_sa.predict(input_test)
    
    def LanguageDetect(self, text):
        text = self.make_pre_processing_ld(text)
        return self.model_ld.predict([text])[0]
        
    def runer(self, text):
        predicted_lang = self.LanguageDetect(text)
        try:
            translated = GoogleTranslator(source= predicted_lang, target='english').translate(text)
        except:
            translated = text
        
        sa_res = self.SentimentAnalysis(translated)
        
        return [predicted_lang, sa_res]
        
# def main():
#     text = "I love hard work"
    
#     ld_sa = LanguageDetect_and_SentimentAnalysis()
#     res = ld_sa.runer(text)
    
#     print(res)
    
# if __name__ == '__main__':
#     main()