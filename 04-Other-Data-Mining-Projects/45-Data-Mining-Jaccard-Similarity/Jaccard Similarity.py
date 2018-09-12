import itertools

# The actual Jaccard similarity function from https://gist.github.com/ramhiser/c990481c387058f3cce7
def jaccard(labels1, labels2):
    """
    Jaccard similarity between two lists of clustering labels. The value of the function varies from (different) and
    to 1 (similar), inclusively. http://en.wikipedia.org/wiki/Jaccard_index
    Example:
    labels1 = [1, 2, 2, 3]
    labels2 = [3, 4, 4, 4]
    print jaccard(labels1, labels2)

    """
    # Need to throw an exception instead of if here
    if len(labels1) != len(labels2):
        print ('Different lengths!')

    # n11 = TP true positive (the pair of points is in both sets),
    # n01 = FN false negative (the pair of points is in the second set, but not in the first one)
    # n10 = FP false negative (the pair of points is in the first set, but not in the second one)
    # n00 = TN true negatives are ignored according to the definition of Jaccard similarity)
    n11 = n10 = n01 = 0

    # The length of both sets should be equal, so it does not matter which set you use here
    n = len(labels1)

    for i, j in itertools.combinations(xrange(n), 2):
        comembership1 = (labels1[i] == labels1[j])
        comembership2 = (labels2[i] == labels2[j])
        if comembership1 and comembership2:
            n11 += 1
        elif comembership1 and not comembership2:
            n10 += 1
        elif not comembership1 and comembership2:
            n01 += 1

    # return the Jaccard similarity (note that one number is float() to make the entire result float too
    return float(n11) / (n11 + n10 + n01)

# The text files in this directory contain the ground truth and clistering results for 5 cases
# These files contain only labels; the ordinal numbers identifying each label (1 through 300) were removed)
# Building lists x and y from the ground truth file and one of the clustering results file.
# Change the digit in the file name for each consecutive file (1, 2, 3, 4, 5)
x, y = [], []
with open("partitions.txt", "r") as file1:
    for line in file1:
        a = int(line)
        x.append (a)
with open ("clustering_1.txt", "r") as file2:
    for line in file2:
        b = int(line)
        y.append (b)

# Function call
c = jaccard(x, y)
print ("Jaccard index: " + str(c))
#END OF SOLUTION FILE

# The following in-built jaccard similarity measures were tried and found incorrect for this assignment
# Apparently they should be used for something else
#
# OTHER PY FILE 1
#
# from sklearn.metrics import jaccard_similarity_score
# import scipy.spatial.distance
#
# x, y = [], []
# with open("partitions.txt", "r") as file1:
#     for line in file1:
#         a = int(line)
#         x.append (a)
# with open ("clustering_5.txt", "r") as file2:
#     for line in file2:
#         b = int(line)
#         y.append (b)
#
# print x
# print y
# c = jaccard_similarity_score(x, y)
# d = jaccard_similarity_score(x, y, normalize=False)
# print ("Jaccard similarity score: " + str(c))
# print ("jaccard similarity score with normalize=False: " + str(d))
#
# # The code below has the same result as "c" above, i.e. c == e.
# # Found advice to use "1-Hamming Distance" here: https://stackoverflow.com/questions/37003272/how-to-compute-jaccard-similarity-from-a-pandas-dataframe
# # Hamming distance is exaplained here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.hamming.html#scipy.spatial.distance.hamming
# e = scipy.spatial.distance.hamming(x, y)
# print ("1 - Hamming distance: " + str(1-e))
#
# # Jaccard spacial distance is explaned here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jaccard.html#scipy.spatial.distance.jaccard
# g = scipy.spatial.distance.jaccard(x, y)
# print("Jaccard spacial disance: " + str(g))
#
# OTHER PY FILE 2
#
# def jaccard_similarity(x,y):
#
#  intersection_cardinality = float(len(set.intersection(*[set(x), set(y)])))
#  union_cardinality = float(len(set.union(*[set(x), set(y)])))
#  return intersection_cardinality/union_cardinalitydef jaccard_similarity(x,y):
#
#  intersection_cardinality = float(len(set.intersection(*[set(x), set(y)])))
#  union_cardinality = float(len(set.union(*[set(x), set(y)])))
#  return intersection_cardinality/union_cardinality
#
# x, y = [], []
# with open("partitions.txt", "r") as file1:
#     for line in file1:
#         a = str(int(line))
#         x.append (a)
# with open ("clustering_5.txt", "r") as file2:
#     for line in file2:
#         b = str(int(line))
#         y.append (b)
# c = jaccard_similarity(x, y)
# print c
#
# x, y = [], []
# with open("partitions.txt", "r") as file1:
#     for line in file1:
#         a = str(int(line))
#         x.append (a)
# with open ("clustering_5.txt", "r") as file2:
#     for line in file2:
#         b = str(int(line))
#         y.append (b)
# c = jaccard_similarity(x, y)
# print c
