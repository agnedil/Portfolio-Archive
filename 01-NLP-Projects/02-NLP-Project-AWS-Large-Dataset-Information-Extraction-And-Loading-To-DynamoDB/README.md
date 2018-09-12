# Project - Information Extraction from Large AWS Public Dataset and Loading Results to DynamoDB

INPUT - the complete MSDS dataset is available as a snapshot in the AWS collection of large public datasets (https://aws.amazon.com/datasets/material-safety-data-sheets/) and contains over 230,000 individual MSDS, each saved in a separate file. They were processed using extract_info.py.

OUTPUT - each line of the processed 2 output csv files (titled results_entire_dataset_part1 & part2.txt) has 5 semicolon-delimited categories and looks like this:

Product; Ingredients; Reactivity; Conditions_to_avoid; PPE;\n

Each such line corresponds to one MSDS (material safety data sheet). Commas may be used to separate elements within each category for convenience. The PPE category contains the complete text from the corresponding MSDS section because everything there may be important. ALL semicolons were removed from inside each category and replaced with commas to facilitate the subsequent parsing. These results were uploaded to a DynamoDB database in the form of 5 columns corresponding to the above 5 categories using populate_dynamodb.py. This information is to be used by a web app that provides safety recommendations after the users enter information about chemicals they are planning to utilize as part of their new work cycle (a separate project).

Due to a very large size, only a small subset of the AWS MSDS public dataset is provided in this repository in the 'sample_data' folder as an example; the results of processing it are in results_sample_dataset.txt
