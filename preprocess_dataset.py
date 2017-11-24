import sys
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

fh = open(sys.argv[1],'r')
lines=fh.readlines()
fh.close()


arr=[]

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stopwrd = set(stopwords.words('english'))
for ln in lines:
	ln = ln.replace(r"don't","do not")
	ln = ln.replace(r"doesn't","does not")
	ln = ln.replace(r"can't","cannot")
	ln = ln.replace(r"caore","core")
	ln = ln.replace(r'&quot;',' ')
	ln = ln.replace(r'&nbsp;',' ')
	ln = ln.replace(r'.',' . ')
	ln = ln.replace(r'...',' . ')
	ln = ln.replace(r'-',' - ')
	ln = ln.replace(r'/',' / ')
	ln = ln.replace(r"'s",' ')
	ln = ln.replace(r"'",' ')
	ln = ln.replace(r"<!-- rich text -->",'')
	words = word_tokenize(ln)
	new_words = [w for w in words if w not in stopwrd]
	new_ln = " ".join(new_words)
	new_ln = new_ln.replace(r'``','')
	new_ln = new_ln.replace(r"''",'')
	new_ln = new_ln.replace(r"< ! -- rich text -- > ",'')
	#words = word_tokenize(new_ln)
	#new_words = [porter_stemmer.stem(w) for w in words]
	#new_ln = " ".join(new_words)
	arr.append(new_ln.lower())

fw = open(sys.argv[1]+'.1','w')

for ln in arr:
	fw.write(ln + "\n")

fw.close()
