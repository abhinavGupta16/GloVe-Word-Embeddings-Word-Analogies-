# GloVe Word Embeddings (Word Analogies)
This module uses GloVe word representations for Word Embeddings.

## Dataset
1) GloVe dataset can be downloaded from https://nlp.stanford.edu/projects/glove/.
2) analogy.txt contains 3 words separated by a space, to predict the 4th word in the equation - </br>
A - B + C = D
3) wordsim-353.txt - contains word pairs with a source of human judgements of how related the 2 words are. We compte the cosine similarity between the 2 word and then find correlation between the given and output words.

## Directory Structure
<pre>
output/
analogy_output/
wordsim-353.txt
analogy.txt
</pre>

## Running the Script
Contains 3 arguments - 
1) -f (required) : path to the GloVe dataset
2) -w : To calculate teh correlation between human scores and calculated scores on wordsim-353.txt
3) -a : To calulate the analogy between the given words in analogy.txt
<pre>
pyhton GloVe_WordEmbeddings.py  -f glove.6B.50d.txt -a
python GloVe_WordEmbeddings.py  -f glove.6B.50d.txt -w
</pre>
