# Transliteration from English to Hindi

## Overview
The project uses parallel hindi-english phonetic pairs to train a model that can transliterate an english text to hindi.  


![demo1](https://user-images.githubusercontent.com/35737849/193358107-c13dcd97-8063-4b8f-a41c-998278defaa7.PNG)

<br/>

## Motivation

There are numerous languages used by people from different parts of the world. In a country like India, there is a huge variety of vernacular languages across the nation. Sometimes, these difference in languages becomes a hindrance in communication and learning of the language. Transliteration helps people pronounce words and names in foreign languages. It makes it easier for people to read and know foreign scripts.

## Task

We divide the task for this project into:
- Preprocessing: Extract the raw text from the source, and make necessary transformation and tuning to turn it to a form which works well for our model.
- Network Architecture and Training: Design our network and set up training methods which helps in building a model that can transliterate any english text to hindi with good accuracy. 

## About Data

The data has 14919 pairs of English-Hindi text.


## Roadmap

Preprocessing:

- English-Hindi Pair extraction 

- Textual Noise removal


Network Design and Training:

- Model Design (Sequence to Sequence with Attention)

- Set up Training for Model

- Train and Evaluate model


[Full Notebook](https://colab.research.google.com/drive/1B9Gn-d7ypihgCwtNf6RexDdDwx9VBisj#scrollTo=_k0iRI-uzUfe)

## Tech Stack

- Pytorch
- pandas
- matplotlib
- numpy
- scikit learn
- nltk



## Acknowledgements/Credits

 The dataset was obtained through crowd sourcing by Amazon Mechanical Turk. 

 Kindly cite this paper if you are using this dataset for research:

Mitesh M. Khapra, Ananthakrishnan Ramanathan, Anoop Kunchukuttan, Karthik Visweswariah, Pushpak Bhattacharyya. When Transliteration Met Crowdsourcing : An Empirical Study of Transliteration via Crowdsourcing using Efficient, Non-redundant and Fair Quality Control . Language and Resources and Evaluation Conference (LREC 2014). 2014.
