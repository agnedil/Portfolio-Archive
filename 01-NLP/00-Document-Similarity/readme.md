## This repository contains research and final solution for finding duplicates, near-duplicates, versions in a large collection of documents

_Steps:_  
1. Comparison of different libraries (01_text_similarity_sklearn_vs_gensim.ipynb)
2. Comparison of different approaches using the selected library (02_doc_similarity_sklear_test_approaches.ipynb)
3. Final solution (03_doc_similarity_final_solution.ipynb)

_The final solution implements the following:_

__a. Connect to a remote relational database__

__b. Run SQL query and retrieve document text__

__c. Convert documents into word vectors (tf-idf)__

__d. Compute cosine similarity for all pairs of documents (numpy allows for a faster implementation)__

__e. Write the results back to the remote relational database as a separate table / relation with the following schema: doc1, doc2, similarity score. This will allow to use them in a dashboard with the help of a visualization tool__

Local runtime on a real-life collection of 800 large multi-page documents: 7 minutes. Manual analysis of results: a lot of (near-) duplicates and different versions of the same document found
