myset = []
with open("patterns.txt", 'r') as f2:
    for line in f2:
        line = line.decode('utf-8')
        split_line = line.strip().split(":")
        myset.append(split_line)
f4 = open("patterns1.txt", "w")
for item1, item2 in myset:
    toFile = item1 + ":" + item2
    f4.write(toFile + "\n")
f4.close()

print("Done!")
