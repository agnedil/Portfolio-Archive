# DRAFT REST API BATCHER (working)  

This repository contains several batchers:

* __batcher_classification.py__ - a) sends the text of each document from the test set to the text mining engine via a REST API call (xml-based); b) receives the text classification category(-ies) assigned by the model and computes the classification metrics for the entire test set: precision, recall, f1 score, accuracy, and the confusion matrix. Cases covered: binary or multiclass text classification with one or more labels per text - the entire range of possible text classification scenarios;

* __batcher_concepts.py__ - a) sends the text of each document to the text mining engine via a REST API call (xml-based); b) receives simple and complex concepts (otherwise called key words and key phrases) from the engine in the form of xml responses, parses and processes them; c) prints and saves to file the frequency statistics for the concepts;

* __batcher_trace.py__ - a) parses a special-purpose log file created by the text mining engine; b) retrieves information about the keywords and text classification categories accepted or rejected by the text mining engine; c) displays and writes the results to file. The retrieved information is used to debug the models available in the text mining engine.

* __batcher_utils.py__ - utility functions used by all the above batchers which do the real computations

* __settings file__ - parsed by the above batchers to retrieve any particular settings for the jobs to be done: file directories, URLs for REST API calls, server names, credentials, etc.

One of possible usages:

1.	Install Anaconda: https://docs.anaconda.com/anaconda/install/windows/. When implementing Step 1 of the Guide – select the Anaconda version with Python 3.7 for WINDOWS (the default one is for MacOS). Do not check “Add Anaconda to your PATH environment variable” as advised in Step 8 of the Guide;
2.	Start the Anaconda Prompt (I rarely use the Anaconda Navigator mentioned in Step 8)
3.	In the Anaconda Prompt, 'cd' into your working directory where you copied the batchers and settings.txt – the Anaconda Prompt works exactly the same way as the Windows Command Prompt in this regard. More here: https://www.digitalcitizen.life/command-prompt-how-use-basic-commands
4.	In any text editor of your choice, carefully complete and SAVE the 'settings.txt' file which contains detailed instructions inside
5.	In the Anaconda Prompt, type 'python {full batcher file name}' and hit Enter. Summary of results will be shown on the screen and saved in a results/metrics.txt file while the complete per-document results will be saved in results/results.csv. If something is not clear in the summary, it may be a good idea to look inside the per-document csv file (remember, batcher.py is a draft). Filenames include a timestamp, and the files will accumulate in the results folder, so with time you may want to delete the results folder completely and it will be re-created automatically the next time you run the script