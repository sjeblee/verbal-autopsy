# Original code by Parthkumar Parmar

#-*-coding:UTF-8-*-
import string

#list of consonants 
consonants = ["k","kh","g","gh","ch","Ch","j","jh","T","Th","D","Dh","N","t","th","d","dh","n","p","ph","b","bh","m","y",\
                    "r","l","v","w","sh","Sh","s","h"]

#list of vowels 
vowels = ["a","A","i","I","u","U","E","ai","O","au","RRi","RRI","LLi","LLI","M","H","OM"]

#list of consonant_cluster
consonants_cluster = ["xa","tra","GYa","shra"]

#list of consonants with nuqta 
consonants_nuqta = ["qa","Kha","Ga","za","fa","Ra","Rha"]

#list of half_forms 
half_forms = ["tta" ,"lkA","svA","sthya","spa","chcha","kta","nhEM","nhOM","sya","mba","ndhI","kTa","bla","bbE","sthi",\
                    "tyu","chchA","ntu","lga","shki","stE","sta","jyA","ddha","vya","kka","stA","bra","pra","nma","rva"]
#list of punctuations 
punctuations = [".","\n"]


#dictionary mapping consonants to the hindi characters 
consonant_mapping = {"k":'क',"kh":'ख',"g":'ग',"gh":'घ',"ch":'च',"Ch":'छ',"j":'ज',"jh":'झ',"T":'ट',"Th":'ठ',"D":'ड',"Dh":'ढ',"N":'ण',\
            "t":'त',"th":'थ',"d":'द',"dh":'ध',"n":'न',"p":'प',"ph":'फ',"b":'ब',"bh":'भ',"m":'म',"y":'य',"r":'र',"l":'ल',\
                "v":'व',"w":'व',"sh":'श',"Sh":'ष',"s":'स',"h":'ह'}

#dictionary mapping vowels to the hindi characters 
vowels_mapping = {"a":'अ',"A":'आ',"i":'इ',"I":'ई',"u":'उ',"U":'ऊ',"E":'ए',"ai":'ऐ',"O":'ओ',"au":'औ',"RRi":'ऋ',\
                    "RRI":'ॠ',"LLi":'ऌ',"LLI":'ॡ',"M":'अं',"H":'अः',"OM":'ॐ'}



#dictionary mapping consonant clusters to the hindi characters 
cc_mapping={"xa":'क्ष',"tra":'त्र',"GYa":'ज्ञ',"shra":'श्र'}

#dictionary mapping half forms to the hindi characters 
halfforms_mapping={"tta":'त्त' ,"lkA":'ल्का ',"svA":'स्वा',"sthya":'स्थ्य',"spa":'स्प',"chcha":'च्च',"kta":'क्त',"nhEM":'न्हें',"nhOM":'न्हों',\
                        "sya":'स्य',"mba":'म्ब',"ndhI":'न्धी',"kTa":'क्ट',"bla":'ब्ल',"bbE":'ब्बे',"sthi":'स्थि',"tyu":'त्यु',"chchA":'च्चा',\
                    "ntu":'न्तु',"lga":'ल्ग',"shki":'श्कि',"stE":'स्ते',"sta":'स्त',"jyA":'ज्या',"ddha":'द्ध',"vya":'व्य',"kka":'क्क',"stA":'स्ता',\
                    "bra":'ब्र',"pra":'प्र',"nma":'न्म',"rva":'र्व'}

#dictionary mapping consonant with nuqtas to the hindi characters 
conso_nuqta_mapping ={"qa":'क़',"Kha":'ख़',"Ga":'ग़',"za":'ज़',"fa":'फ़',"Ra":'ड़',"Rha":'ढ़'}

hindi_consonant=['क','ख','ग','घ','च','छ','ज','झ','ट','ठ','ड','ढ','ण',\
            'त','थ','द','ध','न','प','फ','ब','भ','म','य','र','ल',\
                'व','श','ष','स','ह']
#mapping of vowels when they are used with consonants 
vowels_translation = {'अ':'','आ':'ा','इ':'ि','ई':'ी','उ':'ु','ऊ':'ू','ए':'े','ऐ':'ै','ओ':'ो','औ':'ौ','ऋ':'ृ',\
                    'ॠ':'ॄ','ऌ':'ॢ','ॡ':'ॣ','अं':'ं','अः':'ः','ॐ':''}
# ----

#get file with hindi nrratives 
def get_file(num):
  file_name = "/Users/parthparmar/Desktop/Research/docs/hindi_narrative_sample_"+num+".txt"
  return open(file_name,'r')

#write the converted text to the file

def write_into_file(result):
     file_name = "./translation.txt"
     file = open(file_name,"a")
     file.write(result)
     file.write("\n")

# ----

#remove punctuation from the text
def remove_punctuation(words):

    words_processed=[]
    for word in words:
        no_punct = ""
        for char in word:
            if char not in punctuations:
                no_punct = no_punct + char
        words_processed.append(no_punct)
    return words_processed

#returns true if given char is in the specified group
def check(group,char):
    if char in group:
        return True
    else:
        return False

#returns hindi character for specified letter in english 
def find_mapping(mapping_dict,letter):
    return mapping_dict[letter]


#returns closest hindi mapping along with type of the hindi character 
def get_letters(word):

    type = ''
    if check(vowels,word):
        hindi = find_mapping(vowels_mapping,word)
        type = 'vowel'
        return True,hindi,type

    elif check(consonants,word):
        #print ("Consonants:  "+word)
        #print find_mapping(consonant_mapping, word)
        hindi = find_mapping(consonant_mapping,word)
        type = 'consonant'
        return True,hindi,type
    elif check(consonants_cluster,word):
        hindi = find_mapping(cc_mapping,word)
        return True,hindi,type
    elif check(consonants_nuqta,word):
        hindi = find_mapping(conso_nuqta_mapping,word)
        return True,hindi,type
    elif check(half_forms,word):
        hindi = find_mapping(halfforms_mapping,word)
        return True,hindi,type
    elif word in string.punctuation and word !="\"":
        hindi = word
        return True,hindi,type
    else:
        hindi=""
        return False,hindi,type


def process(word,is_english):
    #results={}


    current_index = 0
    last_index = len(word)
    result =''
    meaning = ''
    previous_previous_type=''
    previous_type = ''
    current_type=''
    previous_char = ''
    previous_previous_char=''
    
    if is_english:
        result = word
        meaning = word
        return result,meaning
    else:
        
        while(current_index != len(word) and last_index>0):

            is_found,hindi,current_type = get_letters(word[current_index:last_index])
            if(is_found and hindi != ""):
                result = result + word[current_index:last_index]

                if current_type == 'vowel':
                    if previous_type == 'consonant':
                        meaning = meaning+vowels_translation[hindi]
                        #previous_type = ''
                    elif previous_type == 'vowel':
                        meaning = meaning+vowels_translation[hindi]
                        #previous_type = ''
                        
                    else:
                        meaning = meaning+hindi
                        #previous_type =''
                else:
                    previous_type = current_type
                    meaning = meaning+hindi


                current_index = last_index
                last_index = len(word)
            else:
                last_index = last_index-1
        return result,meaning


#if hindi word is made up of consonants followed by vowels, then it replaces the consonant and vowel
#with appropriate mapping 
def process_hindi_words(word):
    
    i = 0
    j= 1
    while(j <= len(word)):
        if(word[i] in hindi_consonant and word[j] in vowels_translation.keys()):
            char = word[i]+vowels_translation[word[j]]
            print char
            i = i + 2
            j = j + 2
        else:
            i = i+2
            j=j+2

            
#read from hindi narratives and write the translation into the results file             
def process_data(entry):
    words = remove_punctuation(entry.split(" "))

    result = ''
    is_english = False
    for word in words:

        if len(word)>0:
            if word[0]=="\"" and word[len(word)-1]=="\"":
                is_english = True
                item,trans = process(word,is_english)
                is_english = False  
            elif word[0]=="\"":
                is_english = True
                item,trans = process(word,is_english)
            elif word[len(word)-1]=="\"":
                item,trans = process(word,is_english)
                is_english = False
            else:
                item,trans = process(word,is_english)
        #print item+":"+trans
            result = result+" "+trans
    write_into_file(result)
    return (result)

# ----

from lxml import etree
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('../wiki.hi.vec', binary=False,encoding='UTF8',unicode_errors='ignore')

# ----

total = 0
matched = 0
unmatched = 0
tree = etree.parse('../data/mds+rct/train_adult_cat.xml')
for e in tree.iter("hindi_narrative"):
    t,m,u = get_coverage(e.text.encode('utf-8'))
    total = total + t
    matched = matched + m
    unmatched = unmatched +u
percent_matched = (float(matched)/total)*100
percent_unmatched = (float(unmatched)/total)*100
print ("words matched:{}({}%)".format(matched,percent_matched))
print ("words matched:{}({}%)".format(unmatched,percent_unmatched))
print ("total words:%d"%total)

#####

vowel_regex = re.compile(r"((?<=a)|(?<=A)|(?<=i)|(?<=I)|(?<=u)|(?<=U)|(?<=E)|(?<=ai)|(?<=O)|(?<=au)|(?<=RRi)|(?<=RRI)|(?<=LLi)|(?<=LLI)|(?<=M)|(?<=H)|(?<=OM))")

def preprocess_translit(translit_text):
    '''
    Input:
        translit_text: initial block of text transliteration
    Output:
        preprocessed text transliteration as [list]
    '''

    # remove punctuation
    # split by word 
    # return list 

    return translit_text.translate({ord(c): None for c in ' '}).split()

def translit_word(tr_word):
    '''
    Input:
        tr_word: single string of transliteration
    Output:
        nagari version of the transliteration
    '''

    # add a space after each instance of any vowel
    # split by space
    # the result is that we have consonant-vowel clusters now
    # perform the mapping and return 

    split_word_list = vowel_regex.sub(r' ', tr_word).split()
    nagari = "".join([nagari_dict[character] for character in split_word_list])

    return nagari 

def convert_translit(translit_text_processed):
    '''
    Input:
        translit_text_processed: preprocessed [list] of text transliteration
    Output:
        devanagari version of text
    '''
    return " ".join([translit_word(word) for word in translit_text_processed])

def translit_to_nagari(translit_text):
    '''
    Input:
        translit_text: initial block of text transliteration
    Output:
        devanagari version of text
    '''

    return convert_translit(preprocess_translit(translit_text))

