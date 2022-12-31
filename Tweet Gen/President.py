import json
import csv
import re
import time
    
text = []
clean_text = []
reply = []
retweet = []

csv_dir = "Tweets_csv.csv"

def gather_data(year):
    if year == 2019:
        current_dir = "condensed_" + str(year) + ".json" + "/condensed_" + str(year) + ".json"
        with open(current_dir, 'r', encoding = 'utf-8') as f:
            data = json.load(f)
            print('opening: ' + current_dir)

        for item in data:
            text.append(item['text'])
            retweet.append(item['is_retweet'])
    
    else:    
        for i in range(10):
            current_dir = "condensed_" + str(year) + ".json" + "/condensed_" + str(year) + ".json"
            with open(current_dir, 'r', encoding = 'utf-8') as f:
                data = json.load(f)
                print('opening: ' + current_dir)

            for item in data:
                text.append(item['text'])
                reply.append(item['in_reply_to_user_id_str'])
                retweet.append(item['is_retweet'])
        
            year += 1

def csv_gather():
    with open(csv_dir, 'r', encoding = 'utf-8') as r:
        data = csv.reader(r)
        print('Opening csv')
    
        for item in data:
            text.append(item[1])

gather_data(2009)        
gather_data(2019)
#csv_gather()

num = 0
passed = 0

def english(s):
    #use the ascii encoding to check for english chars
    try:
        s.encode(encoding = 'utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True
        
    
at_pattern = re.compile(r'@(\s)?\w+')
exact_pattern = re.compile(r'(\s?\s:)?(https?://?t?.?)?(https?:?)?(http://nixonssecrets\sdot\scom)?(http://GaryJohnson2012)?(RT\s\s?_?:)?([rR][eE]:)?(&amp;)?')
website_pattern = re.compile(r'(https?)?(:)?(//)?(www\.)?(\w+)?(\.\w+\/?)(\/)?(\w+\/?)?(\/)?(\w+\/?)?(\/)?(\w+\/?)?(\/)?(\w+\/?)?(\/)?(\w+\/?)?')

matches = []
matches_2 = []
matches_3 = []

for i in text:
    x = at_pattern.findall(i)
        
    if len(x) != 0:
        matches.append(x)
        
    y = website_pattern.findall(i)
    
    if len(y) != 0:
        matches_2.append(y)
            
    z = exact_pattern.findall(i)
    
    if len(z) != 0:
        matches_3.append(z)
            
    
for i in text:
    x = website_pattern.sub('', i)
    x = at_pattern.sub('', x)
    x = exact_pattern.sub('', x)
    x = x.strip()
    x = x.replace('\n', ' ')
        
    if len(x) != 0:
        clean_text.append(x)
    
current_text = ""
cleanest_text = ""
    
for i in clean_text:
    for x in i:
        z = english(x)
            
        if z == True:
            current_text += x
    cleanest_text += " "
    cleanest_text += current_text
    current_text = ""
    
if retweet is None:
    with open('cheat_sheet.txt', 'a', encoding = 'utf-8') as w:
        w.write(f"Num:{num + 1} Text:{cleanest_text[num]}\n")
        passed += 1
if retweet[num] == False:
    with open('cheat_sheet_human2.txt', 'a', encoding = 'utf-8') as w:
        w.write(f"{cleanest_text}\n")
        #w.write(f"Num: {num + 1},Text: {text[num]}\n")
        passed += 1
num += 1
