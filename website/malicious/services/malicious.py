from website.common import send_format
# from website.common.exceptions import CustomError
import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import os
import seaborn as sns
from wordcloud import WordCloud
import re
from urllib.parse import urlparse
from googlesearch import search
from tld import get_tld
import os.path
from sklearn.preprocessing import LabelEncoder

import logging
logger = logging.getLogger(__name__)

class Service:

    '''
    * init
    '''
    def __init__(self):
        self.task = None

    '''
    * Parsing Input data
    '''
    def _parse_input(self, input_data: dict):
        self.url = input_data.get('task')

    def having_ip_address(self, url):
        match = re.search(
            '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
            '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
            '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
            '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
        if match:
            # print match.group()
            return 1
        else:
            # print 'No matching pattern found'
            return 0

    def abnormal_url(self, url):
        hostname = urlparse(url).hostname
        hostname = str(hostname)
        match = re.search(hostname, url)
        if match:
            # print match.group()
            return 1
        else:
            # print 'No matching pattern found'
            return 0

    def google_index(self, url):
        site = search(url, 5)
        return 1 if site else 0

    def count_dot(self, url):
        count_dot = url.count('.')
        return count_dot

    def count_www(self, url):
        url.count('www')
        return url.count('www')

    def count_atrate(self, url):
        return url.count('@')

    def no_of_dir(self, url):
        urldir = urlparse(url).path
        return urldir.count('/')

    def no_of_embed(self, url):
        urldir = urlparse(url).path
        return urldir.count('//')

    def shortening_service(self, url):
        match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                          'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                          'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                          'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                          'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                          'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                          'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                          'tr\.im|link\.zip\.net',
                          url)
        if match:
            return 1
        else:
            return 0
    
    def count_https(self, url):
        return url.count('https')

    def count_http(self, url):
        return url.count('http')

    def count_per(self, url):
        return url.count('%')

    def count_ques(self, url):
        return url.count('?')

    def count_hyphen(self, url):
        return url.count('-')

    def count_equal(self, url):
        return url.count('=')

    def url_length(self, url):
        return len(str(url))

    #Hostname Length
    def hostname_length(self, url):
        return len(urlparse(url).netloc)

    def suspicious_words(self, url):
        match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
                          url)
        if match:
            return 1
        else:
            return 0

    def digit_count(self, url):
        digits = 0
        for i in url:
            if i.isnumeric():
                digits = digits + 1
        return digits

    def letter_count(self, url):
        letters = 0
        for i in url:
            if i.isalpha():
                letters = letters + 1
        return letters

    def fd_length(self, url):
        urlpath= urlparse(url).path
        try:
            return len(urlpath.split('/')[1])
        except:
            return 0

    def tld_length(self, tld):
        try:
            return len(tld)
        except:
            return -1


    def main(self, url):
    
        status = []
        
        status.append(self.having_ip_address(url))
        status.append(self.abnormal_url(url))
        status.append(self.count_dot(url))
        status.append(self.count_www(url))
        status.append(self.count_atrate(url))
        status.append(self.no_of_dir(url))
        status.append(self.no_of_embed(url))
        
        status.append(self.shortening_service(url))
        status.append(self.count_https(url))
        status.append(self.count_http(url))
        
        status.append(self.count_per(url))
        status.append(self.count_ques(url))
        status.append(self.count_hyphen(url))
        status.append(self.count_equal(url))
        
        status.append(self.url_length(url))
        status.append(self.hostname_length(url))
        status.append(self.suspicious_words(url))
        status.append(self.digit_count(url))
        status.append(self.letter_count(url))
        status.append(self.fd_length(url))
        tld = get_tld(url,fail_silently=True)
          
        status.append(self.tld_length(tld))

        return status

    def get_prediction_from_url(self, test_url, X_train, y_train):
        features_test = self.main(test_url)
        features_test = np.array(features_test).reshape((1, -1))
        lgb = LGBMClassifier(objective='multiclass',boosting_type= 'gbdt',n_jobs = 5, 
          silent = True, random_state=5)
        LGB_C = lgb.fit(X_train, y_train)
        pred = lgb.predict(features_test)
        if int(pred[0]) == 0:
            
            res="SAFE"
            return res
        elif int(pred[0]) == 1.0:
            
            res="DEFACEMENT"
            return res
        elif int(pred[0]) == 2.0:
            res="PHISHING"
            return res
            
        elif int(pred[0]) == 3.0:
            
            res="MALWARE"
            return res

    '''
    * 
    '''
    def _get_malicious_data(self):
        #str_date = utils.get_date_to_string(self.data_date)
        #raw_result = db.get_dashboard_digital_value(self.user_id, str_date)
        df = pd.read_csv('../data/malicious_phish.csv')
        df_phish = df[df.type=='phishing']
        df_malware = df[df.type=='malware']
        df_deface = df[df.type=='defacement']
        df_benign = df[df.type=='benign']

        df['use_of_ip'] = df['url'].apply(lambda i: self.having_ip_address(i))
        df['abnormal_url'] = df['url'].apply(lambda i: self.abnormal_url(i))
        df['google_index'] = df['url'].apply(lambda i: self.google_index(i))
        df['count.'] = df['url'].apply(lambda i: self.count_dot(i))

        df['count-www'] = df['url'].apply(lambda i: self.count_www(i))
        df['count@'] = df['url'].apply(lambda i: self.count_atrate(i))
        df['count_dir'] = df['url'].apply(lambda i: self.no_of_dir(i))
        df['count_embed_domian'] = df['url'].apply(lambda i: self.no_of_embed(i))

        df['short_url'] = df['url'].apply(lambda i: self.shortening_service(i))

        df['count-https'] = df['url'].apply(lambda i : self.count_https(i))
        df['count-http'] = df['url'].apply(lambda i : self.count_http(i))
        df['count%'] = df['url'].apply(lambda i : self.count_per(i))
        df['count?'] = df['url'].apply(lambda i: self.count_ques(i))
        df['count-'] = df['url'].apply(lambda i: self.count_hyphen(i))
        df['count='] = df['url'].apply(lambda i: self.count_equal(i))
        df['url_length'] = df['url'].apply(lambda i: self.url_length(i))
        df['hostname_length'] = df['url'].apply(lambda i: self.hostname_length(i))
        df['sus_url'] = df['url'].apply(lambda i: self.suspicious_words(i))
        df['count-digits']= df['url'].apply(lambda i: self.digit_count(i))
        df['count-letters']= df['url'].apply(lambda i: self.letter_count(i))
        df['fd_length'] = df['url'].apply(lambda i: self.fd_length(i))
        #Length of Top Level Domain
        df['tld'] = df['url'].apply(lambda i: get_tld(i,fail_silently=True))
        df['tld_length'] = df['tld'].apply(lambda i: self.tld_length(i))

        lb_make = LabelEncoder()
        df["type_code"] = lb_make.fit_transform(df["type"])


        X = df[['use_of_ip','abnormal_url', 'count.', 'count-www', 'count@',
       'count_dir', 'count_embed_domian', 'short_url', 'count-https',
       'count-http', 'count%', 'count?', 'count-', 'count=', 'url_length',
       'hostname_length', 'sus_url', 'fd_length', 'tld_length', 'count-digits',
       'count-letters']]

        #Target Variable
        y = df['type_code']
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2,shuffle=True, random_state=5)

        # urls = ['titaniumcorporate.co.za','en.wikipedia.org/wiki/North_Dakota',]
        # for url in urls:
        #      print(self.get_prediction_from_url(url, X_train, y_train))

        return self.get_prediction_from_url(self.url, X_train, y_train)

    '''
    * Run Service
    '''
    def run(self, validated_data: dict):
        try:
            self._parse_input(validated_data)
            raw_result = self._get_malicious_data()
            #for result in raw_result:
            #    logger.debug('sum_dv : {}'.format(result.sum_dv))
            #print(self.task)
            res = {
                "result": raw_result
            }
            return send_format.success(res)
        # except CustomError as e:
        #     return send_format.custom_error(e.code, e.msg)
        except Exception as e:
            return send_format.exception(str(e))
