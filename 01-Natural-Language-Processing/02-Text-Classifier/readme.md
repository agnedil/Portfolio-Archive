## Sentiment classifier for a large set of customer reviews

PURPOSE: develop a classifier first based solely on text, then include additional features. The labels should be converted from 5 (stars) to 3 labels: positive (4 and 5 stars), negative (1 and 2), mixed (3). Avoid overfitting (by using cross validation)

DATASET: Amazon Product Reviews (http://snap.stanford.edu/data/amazon/productGraph/). Deduplicated dataset in a one review per line json file.

Example of one review:  
{  
"reviewerID ": "  
"  
asin ": "  
"  
reviewerName ": "J.  
"helpful": [2, 3],  
"  
reviewText ": "I bought this for my husband who plays the piano. He is having a wonderful time playing  
these old hymns. The music is at times hard to read because we think the book was published  
for singing from more than playing from. Great purchase though!",  
"overall": 5.0,  
"summary": "Heavenly Highway Hymns",  
"  
unixReviewTime ":  
"  
reviewTime ": "09 14,  
}