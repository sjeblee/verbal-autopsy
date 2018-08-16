from google.cloud import translate
from lxml import etree
import time


# Instantiates a client
translate_client = translate.Client()

#target_language
target = 'hi'

tree = etree.parse('../data/mds+rct/all_adult_cat_last.xml')
root = tree.getroot()
i = 0
for e in tree.iter():

    if e.tag == 'narrative':
        i+=1

        parent = e.getparent()
        index_num = parent.index(e)
        new_node = etree.Element("hindi_narrative")
        translation = translate_client.translate(e.text,target_language=target)
        time.sleep(1)
        new_node.text = u'{}'.format(translation['translatedText'])
        parent.insert(index_num+1,new_node)
        print(i)
        # print(u'Text: {}'.format(e.text))
        # print(u'Translation: {}'.format(translation['translatedText']))
        # print("2")
tree.write('../data/mds+rct/all_adult_cat_last_hi.xml')
