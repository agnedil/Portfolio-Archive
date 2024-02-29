from scipy import stats

def loadfile (file):
    arr = list()
    with open(file) as f:
        for line in f:
            a = float (line.rstrip())
            arr.append(a)
    print (arr)                         #Additional checkpoint to see if the numbers are right
    return arr

def main():
    bm25 = loadfile("bm25.avg_p.txt")
    ln2L = loadfile("inl2.avg_p.txt")
    t = stats.ttest_rel (bm25, ln2L)
    print (t)

if __name__ == '__main__':
    main()
