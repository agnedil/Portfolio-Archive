# Apache HBase

A: Create HBase Tables.
Create 2 tables:

Table 1: a table named “powers” in HBase with three column families as given below. For simplicity, treat all data as String (i.e., don’t use any numerical data type).
Table 2: a table named “food” in HBase with two column families as given below. For simplicity, treat all data as String (i.e., don’t use any numerical data type).

B: List HBase Tables.
List all the tables created in the previous part

C: Populate HBase Table with Data.
Insert data in the “powers“ table with following schema according to input.csv. You can assume that input.csv is in the same folder as the java files.

D: Read Data.
Read some of the data in the populated “powers “ table. Read the values for the following row ids in order and the given attributes:
Id: "row1", Values for (hero, power, name, xp, color)
Id: "row19", Values for (hero, color)
Id: "row1", Values for (hero, name, color)

E: Scan Data.
Scan the data in the populated “powers “ table
