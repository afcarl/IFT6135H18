# IFT6135H18_assignment

## MNIST
to download the MNIST dataset with standard train/valid/test splits, do `python download.py --dataset=mnist --savedir={where-you-want-to-save-the-dataset}`

## 20newsgroups

The data for the training and testing dataset is in 20-news-by-date/matlab.
The data is divided as follow:

vocabulary.txt: The mapping of the different words. The word #1 is `archive`, the word number 2 is `name`, etc.

train.data: The file contains 3 column. The first one is the document index, the second one is the word index and that word count. The first line for example is `1 1 4`, meaning that the first document has four time the word `archive` in it. **Be careful, the indexing starts at 1!**

train.label: The class for each of the document.

train.map: The mapping of the classes.

It's the same thing for the test set.

The home page for the 20 newsgroups is: http://qwone.com/~jason/20Newsgroups/
# IFT6135H18
# IFT6135H18
# IFT6135H18
