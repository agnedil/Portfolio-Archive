
# clean dataset for a decision tree algorithm

# function to remove attribute names before attribute values (e.g. 1:2, 56:3, 127:1 - the part up to ":" is removed
def removeAttrName(fromFile, toFile):
    listNames = []
    for i in range(127, -1, -1):
        string = str(i) + ":"
        listNames.append(string)
    with open(fromFile) as f1, open(toFile, 'w') as f2:
        for line in f1:
            for name in listNames:
                line = line.replace(name, '')
            f2.write(line)
    print("file " + toFile + " created")

# function to check if any attribute values has more than 1 digit (checkpoint)
def checkNum(file):
    i = 0
    with open(file) as f:
        for line in f:
            line = line.strip()
            line = line.split()
            j = 0
            for num in line:
                if len(num) > 1:
                    print(str(i) + ": " + str(num) + "\n")
                    j = j + 1
            i = i + 1
    print("finished checking file " + file)
    if j == 0: print("no multiple values found")

if __name__ == '__main__':

    # existing input files and new output files
    myfile1   = 'training.txt'
    myfile2   = 'testing.txt'
    myfile1_m = 'trainingmod.txt'
    myfile2_m = 'testingmod.txt'

    # initialize two output files
    a1 = open(myfile1_m, 'w+')
    a2 = open(myfile2_m, 'w+')
    a1.close()
    a2.close()

    # remove attribute names
    removeAttrName(myfile1, myfile1_m)
    removeAttrName(myfile2, myfile2_m)

    # check for categories with more than 1 digit
    checkNum(myfile1_m)
    checkNum(myfile2_m)

    # All done!
    print("All done!")