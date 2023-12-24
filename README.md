# FlavorQuest: A Revolutionary Drink-Taste Search Engine

## Overview
Our goal is to develop an innovative drink-taste search engine. This search engine will revolutionize how users discover and enjoy beverages by considering not only the ingredients but also the flavors and characteristics that define each drink. Instead of asking users to specify ingredients or provide drink names, our system will empower users to describe their desired taste experiences, such as "sweety taste for iced tea" or "sour taste with alcoholic drink" allowing them to explore a world of flavor possibilities. 

To achieve our goals, word embedding techniques and document augmentation will play important roles in this project. We'll leverage them for capture the complicated relationship between flavor, ingredient, and making processes. Also, Pretrained deep learning models will be used to rank beverages, ensuring accurate recommendations. So far, we have tried pure BM25 to rank documents by queries without any documentation and word embedding. It will be am important baseline for our further works.

*Flow of the Drink-Taste Search Engine*
![image](https://github.com/show0117/Beverage_Recommendation_System_Based_on_Descriptive_Query/blob/main/Flow of Search Engine.jpg)

## Database
Our data is sourced from the website <https://www.allrecipes.com/>. Initially, we conducted web scraping on the URLs of beverage recipes from the "Drinks" category, found under <https://www.allrecipes.com/recipes/77/drinks/>. We obtained URLs from 18 subcategories within 
this broad category of beverages, ensuring data diversity by taking the union and converting it into a set to avoid duplication. There are over 800 types of beverages in the database of our search engine.

## Methodology
### Query Expansion
For those sparse queries or queries which don’t contain precise information, query augmentation will help us provide similar type of beverages and a more diverse searching result. What we do is to use transformer’s fill-mask mechanism and add masked token at the end of a sentence to generate the most possible ingredients according to context and words in a sentence.

*Examples of our augmented query:*

*Signature cocktail for parties → Signature cocktail for parties with ingredient such as vodka and gin and champagne*

*Rich and velvety flavor of a dark chocolate hot cocoa → Rich and velvety flavor of a dark chocolate hot cocoa with ingredient such as vanilla and cinnamon and cocoa*

*Bubbly refreshing pineapple punch → Bubbly refreshing pineapple punch with ingredient such as pineapple and lime and mango*

### Document Augmentation
Due to the limit of our text which only mainly includes flavor, ingredients, and making processes, this technique can help generate more useful information to improve our rankings. Besides, we use some weak rankers, which means traditional rankers such as TF-IDF, BM25, or Dirichlet will still be implemented in this project for the further stacked model. The results of these weak learners can be good baselines to judge the performance of our works.

For word embedding, first, we want to employ the traditional NLP techniques like TF-IDF, BM25 to capture the complicated relationship among flavor, ingredient, and making processes. However, these traditional techniques didn’t do great job. Thus, we implement Bi-Encoder to encode our text data to reflect the similarities between terms more precisely, this benefits our ranking results.

### Latent Dirichlet Allocation (LDA)
LDA is a probabilistic graphical model commonly used for topic modeling, particularly in the analysis of textual data. Initially, we performed essential preprocessing steps, including tokenization, removal of stop words, and stemming. We utilized the *NLTK* library for tokenization and applied a set of stop words provided by the instructor for stop-word removal.

Subsequently, we employed the *CountVectorizer* from the feature_extraction module in the scikit-learn library to transform the text into a vectorized form, facilitating the calculation of similarity later. We then utilized the *LatentDirichletAllocation* model from the *scikit-learn's* decomposition module for topic classification. LDA describes the generation process of a document collection by representing documents as a mixture of topics. We fine-tuned the number of topics in the LDA model to find the optimal way to classify topics. Additionally, we utilized latent factors for classification, incorporating the classification results as a feature in the L2R model. This was done to observe whether it contributed to the overall performance of the model.

### Learning to Rank System
Due to the previous works on information retrieval system, we learn that Cross Encoder is beneficial to the performance of system. Thus, we also employ the deep learning ranker, Cross-Encoder to our system to get a more robust and accurate ranking.

Our input are the queries and documents which are preprocessed by augmentation, and pass the Bi-Encoder to the IR system, and we use features generated by rankers, such as BM25, and encoder features, grouping labels by LDA, and the basic statistics of documents or terms to train the *LGBMRanker* from lightgbm module, thus, we get the final stacked model to generate the ranking results by other queries.

## document_preprocessor.py
This class is designed for transform each document in the collection into a list of concepts (e.g., words, phrases). It not only helps tokenize but also augment documents. 

## indexing.py
Inverted Index is one of the most important technique for storing and search the huge amount of information through search engine. This class helps transform the processed documents into an inverted index
data structure for easy retrieval.

## ranker.py
Ranking function that uses the inverted index to score how relevant is a document with respect to a query. It mainly includes traditional word-frequency based ranker such as BM25, Dirichlet, TF-iDF. 

## l2r.py
Use learning to rank models to estimate the relevance of a document for a query on the basis of features. This file help extract features through deep encoders or machine learning models. The final stacked model for all statistics features or features extracted by machine learning / deep learning methods can be LGBM or XGBoost. 

## relevance.py
Evaluation functions to measure how different ranking functions perform is implemented by this file to judge the effectiveness of different models. 

## Interactive_Example.ipynb
Sample demo for users. 
