"Original source code from http://stackoverflow.com/questions/4028697/how-do-i-download-a-zip-file-in-python-using-urllib2"
import os
from urllib2 import urlopen, URLError, HTTPError

def download():
    url = ('http://ufldl.stanford.edu/housenumbers/train_32x32.mat')
    get(url)
    url = ('http://ufldl.stanford.edu/housenumbers/test_32x32.mat')
    get(url)

def get(url):
    # Open the url
    try:
        f = urlopen(url)
        print("try to download {}".format(url))
        
        if not os.path.isdir('./data/svhn/'):
            os.makedirs('./data/svhn/')
        # Open our local file for writing
        filename = os.path.join('./data/svhn/', os.path.basename(url))
        if not os.path.exists(filename):
            with open(filename, 'wb') as local_file:
                print("downloading {}".format(url))
                local_file.write(f.read())
        else:
            print("file exists!!")

    #handle errors
    except HTTPError, e:
        print 'HTTP Error:', e.code, url
    except URLError, e:
        print 'URL Error:', e.reason, url
