# Mining Contiguous Sequential Patterns

Input: 10,000 online reviews from Yelp users in 'reviews_sample.txt'. The reviews have been stemmed, and most of the punctuation has been removed. Example line:

  cold cheap beer good bar food good service looking great pittsburgh style fish
 
  sandwich place breading light fish plentiful good side home cut fry good

  grilled chicken salad steak soup day homemade lot special great place lunch
 
  bar snack beer

Output: a text file 'patterns.txt' with frequent contiguous sequential patterns along with their absolute supports, each line should look like this: support:item_1;item_2;item_3

Notes: implementing an algorithm to mine contiguous sequential patterns that are frequent in the input data. A contiguous sequential pattern is a sequence of items that frequently appears as a consecutive subsequence in a database of many sequences. Notice that multiple appearances of a subsequence in a single sequence record only counts once. Set the relative minimum support to 0.01 (an absolute support no smaller than 100)
