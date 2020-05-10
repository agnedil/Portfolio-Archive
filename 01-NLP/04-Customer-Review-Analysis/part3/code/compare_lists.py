import re

def read_list20(myfile):
    mylist = []
    with open(myfile) as f:
        for i in range(20):
            phrase = f.readline()
            phrase = re.sub('[^a-z]+', ' ', phrase.decode('utf-8').lower())
            phrase = phrase.strip()
            mylist.append(phrase)
    return mylist

def read_list(myfile):
    mylist = []
    with open(myfile) as f:
        lines = f.readlines()
    for line in lines:
        line = re.sub('[^a-z]+', ' ', line.decode('utf-8').lower())
        line = line.strip()
        mylist.append(line)
    return mylist

mylist = read_list20('/home/andrew/Documents/2_UIUC/CS598_Data_Mining_Capstone/task3/AutoPhrase_Results/Italian_ALL_mineQuality_andWikiAll/AutoPhrase.txt')
print(mylist)
mylist2 = read_list('/home/andrew/Documents/2_UIUC/CS598_Data_Mining_Capstone/task3/task3_pycharms_project/00_complete_Italian_dishes.txt')
#print(mylist2)
mylist3 = filter(lambda x: x not in mylist2, mylist)
for item in mylist3:
    print(item)
print(len(mylist3))
