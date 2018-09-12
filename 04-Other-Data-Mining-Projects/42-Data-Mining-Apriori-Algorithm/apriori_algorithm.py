# Apriori Algorithm

# My modifications:
# 1) Support was changed from relative (0.01) to absolute (771) for my specific case
# 2) Lambda function was used instead of just "str.strip" in the original file because
#    I am using Python 2.7 and not 3
# This let me get a 90% score. But the autograder accepted the file only after I opened it in Windows MS Word,
# resaved it as a text file using the Windows text encoding (default), and then brought it back to Linux for resubmission

import argparse
from itertools import chain, combinations


def joinset(itemset, k):
    return set([i.union(j) for i in itemset for j in itemset if len(i.union(j)) == k])


def subsets(itemset):
    return chain(*[combinations(itemset, i + 1) for i, a in enumerate(itemset)])
    

def itemset_from_data(data):
    itemset = set()
    transaction_list = list()
    for row in data:
        transaction_list.append(frozenset(row))
        for item in row:
            if item:
                itemset.add(frozenset([item]))
    return itemset, transaction_list


def itemset_support(transaction_list, itemset, min_support=771):
    len_transaction_list = len(transaction_list)
    l = [
        (item, sum(1 for row in transaction_list if item.issubset(row)))
        for item in itemset
    ]
    return dict([(item, support) for item, support in l if support > min_support])


def freq_itemset(transaction_list, c_itemset, min_support):
    f_itemset = dict()

    k = 1
    while True:
        if k > 1:
            c_itemset = joinset(l_itemset, k)
        l_itemset = itemset_support(transaction_list, c_itemset, min_support)
        if not l_itemset:
            break
        f_itemset.update(l_itemset)
        k += 1

    return f_itemset    


def apriori(data, min_support, min_confidence):
    # Get first itemset and transactions
    itemset, transaction_list = itemset_from_data(data)

    # Get the frequent itemset
    f_itemset = freq_itemset(transaction_list, itemset, min_support)

    # Association rules
    rules = list()
    for item, support in f_itemset.items():
        if len(item) > 1:
            for A in subsets(item):
                B = item.difference(A)
                if B:
                    A = frozenset(A)
                    AB = A | B
                    confidence = float(f_itemset[AB]) / f_itemset[A]
                    if confidence >= min_confidence:
                        rules.append((A, B, confidence))    
    return rules, f_itemset


def print_report(rules, f_itemset):
    print('--Frequent Itemset--')
    for item, support in sorted(f_itemset.items(), key=lambda (item, support): support):
        print('[I] {} : {}'.format(tuple(item), round(support, 4)))
        with open("patterns.txt", "a") as f:
            f.write('{}:{}\n'.format(round(support, 4), tuple(item)))

    print('')
    print('--Rules--')
    for A, B, confidence in sorted(rules, key=lambda (A, B, confidence): confidence):
        print('[R] {} => {} : {}'.format(tuple(A), tuple(B), round(confidence, 4))) 


def data_from_csv(filename):
    f = open(filename, 'rU')
    for l in f:
        row = map(lambda it: it.strip(), l.split(';'))  #lambda is needed for Python 2.7; Python 3 would say "map(str.strip..."
        yield row


def parse_options():
    optparser = argparse.ArgumentParser(description='Apriori Algorithm.')
    optparser.add_argument(
        '-f', '--input_file',
        dest='filename',
        help='filename containing csv',
        required=True
    )
    optparser.add_argument(
        '-s', '--min_support',
        dest='min_support',
        help='minimum support',
        default=0.01,
        type=float
    )
    optparser.add_argument(
        '-c', '--min_confidence',
        dest='min_confidence',
        help='minimum confidence',
        default=0.5,
        type=float
    )
    return optparser.parse_args()


def main():

    options = parse_options()

    f = open("patterns.txt", "w")
    f.close()
    data = data_from_csv(options.filename)
    rules, itemset = apriori(data, options.min_support, options.min_confidence)
    print_report(rules, itemset)


if __name__ == '__main__':
    main()

# The MIT License (MIT)

# Copyright (c) 2016 Ariel Barmat

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
