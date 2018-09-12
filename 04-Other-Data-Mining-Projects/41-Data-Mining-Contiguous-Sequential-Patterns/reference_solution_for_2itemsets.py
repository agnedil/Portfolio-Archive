# Frequent two-itemsets

def counted(phrase):
    count = 0
    with open("reviews_sample.txt", "r") as f3:
        for line in f3:
            if phrase in line:
                count += 1
    return count

if __name__ == '__main__':

    myset = []
    with open("patterns_my1&2.txt") as f2:
        for line in f2:
            split_line = line.strip().split(":")
            myset.append(split_line)

    otherset = []
    with open("patterns_fromWinAsWin.txt") as f2:
        for line in f2:
            split_line = line.strip().split(":")
            otherset.append(split_line)

    for i in range (len(myset)-1, -1, -1):
        for j in range (len(otherset)-1, -1, -1):
            if myset[i][1] == otherset[j][1]:
                if myset[i][0] < otherset[j][0]:
                    del otherset[j]
                    myset[i][0] = myset[i][0].decode('utf-8')
                else:
                    del myset[i]
                    otherset[j][0] = otherset[j][0].decode('utf-8')

    f4 = open("patterns.txt", "w")
    for item1 in myset:
        f4.write(str(item1[0]) + ":" + item1[1] + "\n")
    for item2 in otherset:
        f4.write(str(item2[0]) + ":" + item2[1] + "\n")
    f4.close()
    print("Done!")