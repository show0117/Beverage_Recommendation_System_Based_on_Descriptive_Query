from tqdm import tqdm
import pandas as pd
import lightgbm
from indexing import InvertedIndex
import multiprocessing
from collections import defaultdict, Counter
import numpy as np
from document_preprocessor import Tokenizer
from ranker import *
import xgboost as xgb

# TODO: scorer has been replaced with ranker in initialization, check README for more details
class L2RRanker:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex, model_type,
                 document_preprocessor: Tokenizer, stopwords: set[str], ranker: Ranker,
                 feature_extractor: 'L2RFeatureExtractor') -> None:
        self.document_index = document_index
        self.title_index = title_index
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.ranker = ranker
        self.feature_extractor = feature_extractor
        self.model = LambdaMART(model_type=model_type)

    def prepare_training_data(self, query_to_document_relevance_scores: dict[str, list[tuple[int, int]]]):
        """
        Prepares the training data for the learning-to-rank algorithm.

        Args:
            query_to_document_relevance_scores (dict): A dictionary of queries mapped to a list of 
                documents and their relevance scores for that query
                The dictionary has the following structure:
                    query_1_text: [(docid_1, relance_to_query_1), (docid_2, relance_to_query_2), ...]

        Returns:
            tuple: A tuple containing the training data in the form of three lists: x, y, and qgroups
                X (list): A list of feature vectors for each query-document pair
                y (list): A list of relevance scores for each query-document pair
                qgroups (list): A list of the number of documents retrieved for each query
        """
        # NOTE: qgroups is not the same length as X or y.
        # This is for LightGBM to know how many relevance scores we have per query.

        X = []
        y = []
        qgroups = []

        for key in query_to_document_relevance_scores.keys():
            qgroup = len(query_to_document_relevance_scores[key])
            qgroups = qgroups + [qgroup]
            for i in range(qgroup):
                docid = query_to_document_relevance_scores[key][i][0]
                relevance = query_to_document_relevance_scores[key][i][1]
                query_parts = self.document_preprocessor.tokenize(key)
                feature = self.feature_extractor.generate_features(docid, dict({}), dict({}), query_parts)
                X = X + [feature]
                y = y + [relevance]
            

        return X, y, qgroups

    @staticmethod
    def accumulate_doc_term_counts(index: InvertedIndex, query_parts: list[str]) -> dict[int, dict[str, int]]:

        result = dict({})
        for token in query_parts:
            if token in index.index.keys():
                docids = index.index[token]
                for docid in docids:
                    term = token
                    frequency = index.document_metadata[docid]['tokens'].count(token)
                    result[docid] = {term:frequency}

        return result
        

    def train(self, training_data_filename: str) -> None:
        """
        Trains a LambdaMART pair-wise learning to rank model using the documents and relevance scores provided 
        in the training data file.

        Args:
            training_data_filename (str): a filename containing documents and relevance scores
        """
        # TODO: Convert the relevance data into the right format for training data preparation
        
        # TODO: prepare the training data by featurizing the query-doc pairs and
        # getting the necessary datastructures
        
        # TODO: Train the model
        train_data = pd.read_csv(training_data_filename,encoding='unicode_escape')
        query_to_document_relevance_scores = dict({})
        for i in range(len(train_data)):
            query = train_data['query'][i]
            docid = train_data['docid'][i]
            relevance = train_data['rel'][i]
            if query in query_to_document_relevance_scores:
                query_to_document_relevance_scores[query] = query_to_document_relevance_scores[query] + [(docid,relevance)]
            else:
                query_to_document_relevance_scores[query] = [(docid,relevance)]
        X, y, qgroups = self.prepare_training_data(query_to_document_relevance_scores)

        self.model.fit(X,y,qgroups)
        

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        else:
            prediction = self.model.predict(X)
        return prediction
    
    def maximize_mmr(thresholded_search_results: list[tuple[int, float]], similarity_matrix: np.ndarray,
                     list_docs: list[int], mmr_lambda: int) -> list[tuple[int, float]]:
        """
        Takes the thresholded list of results and runs the maximum marginal relevance diversification algorithm
        on the list.
        It should return a list of the same length with the same overall documents but with different document ranks.
        
        Args:
            thresholded_search_results: The thresholded search results
            similarity_matrix: Precomputed similarity scores for all the thresholded search results
            list_docs: The list of documents following the indexes of the similarity matrix
                       If document 421 is at the 5th index (row, column) of the similarity matrix,
                       it should be on the 5th index of list_docs.
            mmr_lambda: The hyperparameter lambda used to measure the MMR scores of each document

        Returns:
            A list containing tuples of the documents and their MMR scores when the documents were added to S
        """
        # NOTE: This algorithm implementation requires some amount of planning as you need to maximize
        #       the MMR at every step.
        #       1. Create an empty list S
        #       2. Find the element with the maximum MMR in thresholded_search_results, R (but not in S)
        #       3. Move that element from R and append it to S
        #       4. Repeat 2 & 3 until there are no more remaining elements in R to be processed

        S = []
        existed_docid = []
        rank_docid = [doc[0] for doc in thresholded_search_results]

        initial_doc = rank_docid[0]
        initial_score = mmr_lambda*thresholded_search_results[0][1]
        S.append((initial_doc,initial_score))
        thresholded_search_results[0]
        existed_docid.append(initial_doc)

        while (len(S) < len(thresholded_search_results)):
            max_score = -100
            max_docid = -100
            for doc in thresholded_search_results:
                docid = doc[0]
                doc_index = list_docs.index(docid)
                doc_similarity = similarity_matrix[doc_index]
                existed_index = [list_docs.index(i) for i in existed_docid]
                max_rel_ij = np.sort([doc_similarity[i] for i in existed_index])[-1]
                score = mmr_lambda*doc[1]-(1-mmr_lambda)*max_rel_ij
                if (score > max_score and docid not in existed_docid):
                    max_score = score
                    max_docid = docid

            S.append((max_docid,max_score))
            existed_docid.append(max_docid)

        return S

    def query(self, query: str) -> list[dict[str, int]]:
        """
        Retrieves potentially-relevant documents, constructs feature vectors for each query-document pair,
        uses the L2R model to rank these documents, and returns the ranked documents.

        Args:
            query: A string representing the query to be used for ranking

        Returns:
            A list of dictionaries representing the ranked documents, sorted in descending order of relevance
        """
        # TODO: Retrieve potentially-relevant documents
        
        # TODO: Fetch a list of possible documents from the index and create a mapping from
        # a document ID to a dictionary of the counts of the query terms in that document.
        # You will pass the dictionary to the RelevanceScorer as input.
        try:
            query_parts = self.document_preprocessor.tokenize(query)
            ranked_data = self.ranker.query(query)
            ranked_data_100 = [score[0] for score in ranked_data[0:100]]
        
            feature_vectors = []
            for docid in ranked_data_100:
                feature_vector = self.feature_extractor.generate_features(docid, dict({}), dict({}), query_parts)
                feature_vectors = feature_vectors + [feature_vector]
        
            l2r_predictions = self.predict(feature_vectors)
            ranked_data_l2r = []
            for i in range(len(ranked_data_100)):
                docid = ranked_data_100[i]
                l2r_score = l2r_predictions[i]
                ranked_data_l2r.append((docid,l2r_score))

            all_rank = ranked_data_l2r + ranked_data[100:]
        except:
            all_rank = []

        all_rank.sort(key=lambda a: a[1], reverse=True)

        return all_rank

class L2RFeatureExtractor:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 document_preprocessor: Tokenizer, stopwords: set[str],
                 ce_scorer: CrossEncoderScorer, category_data) -> None:
        
        self.document_index = document_index
        self.title_index = title_index
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.ce_scorer = ce_scorer
        self.category_data = category_data
        """
        Initializes a L2RFeatureExtractor object.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            doc_category_info: A dictionary where the document id is mapped to a list of categories
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            recognized_categories: The set of categories to be recognized as binary features
                (whether the document has each one)
            docid_to_network_features: A dictionary where the document id is mapped to a dictionary
                with keys for network feature names "page_rank", "hub_score", and "authority_score"
                and values with the scores for those features
        """
        # TODO: Set the initial state using the arguments

        # TODO: for the recognized categories (i.e,. those that are going to be features), considering
        # how you want to store them here for faster featurizing

        # TODO (HW2): Initialize any RelevanceScorer objects you need to support the methods below.
        #             Be sure to use the right InvertedIndex object when scoring.

    
    # TODO: Article Length
    def get_article_length(self, docid: int) -> int:
        article_length = self.document_index.document_metadata[docid]['length']

        return article_length

    # TODO: Title Length
    def get_title_length(self, docid: int) -> int:
        title_length = self.title_index.document_metadata[docid]['length']

        return title_length

    # TODO 2: TF
    def get_tf(self, index: InvertedIndex, docid: int, word_counts: dict[str, int], query_parts: list[str]) -> float:
        score = 0
        for query in query_parts:
            if query in index.index and docid in index.index[query]:
                tf = np.log(index.document_metadata[docid]['tokens'].count(query)+1)
                score += tf
        return score

    # TODO: TF-IDF
    def get_tf_idf(self, index: InvertedIndex, docid: int,
                   word_counts: dict[str, int], query_parts: list[str]) -> float:
        query_word_counts = Counter(query_parts)
        score = TF_IDF(index).score(docid,word_counts,query_word_counts)
        return score


    def get_BM25_score(self, docid: int, doc_word_counts: dict[str, int],
                       query_parts: list[str]) -> float:
        query_word_counts = Counter(query_parts)
        score = BM25(self.document_index).score(docid,doc_word_counts,query_word_counts)
        return score
    
    # TODO: Pivoted Normalization
    def get_pivoted_normalization_score(self, docid: int, doc_word_counts: dict[str, int],
                                        query_parts: list[str]) -> float:
        query_word_counts = Counter(query_parts)
        score = PivotedNormalization(self.document_index).score(docid,doc_word_counts,query_word_counts)
        return score

    # TODO: Document Categories
    def get_document_categories(self, docid: int) -> list:
        categories = self.doc_category_info[docid]
        document_categories = [0]*len(self.recognized_categories)
        for i in range(len(self.recognized_categories)):
            category = self.recognized_categories[i]
            if category in categories:
                document_categories[i] = 1
        
        return document_categories


    # TODO 11: Add at least one new feature to be used with your L2R model.
    
    def get_cross_encoder_score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.

        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The Cross-Encoder score
        """        
        score = self.ce_scorer.score(docid, {}, query)
        return score

    # TODO: Add at least one new feature to be used with your L2R model

    def generate_features(self, docid: int, doc_word_counts: dict[str, int],
                          title_word_counts: dict[str, int], query_parts: list[str]) -> list:

        query_word_counts = Counter(query_parts)

        # TODO: Document Length
        document_length = self.get_article_length(docid)

        # TODO: Title Length
        title_length = self.get_title_length(docid)
        
        # TODO Query Length
        query_length = len(query_parts)
 
        # TODO: TF (document)
        tf_d = self.get_tf(self.document_index, docid, doc_word_counts, query_word_counts)

        # TODO: TF-IDF (document)
        tfidf_d = self.get_tf_idf(self.document_index, docid, doc_word_counts, query_word_counts)

        # TODO: TF (title)
        tf_t = self.get_tf(self.title_index, docid, title_word_counts, query_word_counts)

        # TODO: TF-IDF (title)
        tfidf_t = self.get_tf_idf(self.title_index, docid, title_word_counts, query_word_counts)
                              
        # TODO: BM25
        bm25 = self.get_BM25_score(docid, doc_word_counts, query_word_counts) 
                              
        # TODO: Pivoted Normalization
        pn = self.get_pivoted_normalization_score(docid, doc_word_counts, query_word_counts) 
        
        query = " ".join(query_parts)
        cross_encoder = self.get_cross_encoder_score(docid, query)

        # TODO: Add at least one new feature to be used with your L2R model.
        tokens_title = self.title_index.document_metadata[docid]['tokens']
        tokens_document = self.document_index.document_metadata[docid]['tokens']
        tokens = tokens_title + tokens_document
        num_stopwords = 0
        for token in tokens:
            if token in self.stopwords:
                num_stopwords += 1
        stop_word_ratio = num_stopwords / len(tokens)

        # TODO: Calculate the Document Categories features.
        # NOTE: This should be a list of binary values indicating which categories are present.

        category = [0]*3
        label = self.category_data[self.category_data['docid']==docid]['LDA_3']
        category[int(label)] = 1

        feature_vector = [document_length, title_length, query_length, 
                          tf_d, tfidf_d, tf_t, tfidf_t, bm25, pn,
                          cross_encoder, stop_word_ratio] + category
        
        return feature_vector

class LambdaMART:
    def __init__(self, params=None, model_type = 'LGBM') -> None:
        """
        Initializes a LambdaMART (LGBRanker) model using the lightgbm library.

        Args:
            params (dict, optional): Parameters for the LGBMRanker model. Defaults to None.
        """
        default_params = {
            'objective': "lambdarank",
            'boosting_type': "gbdt",
            'n_estimators': 10,
            'importance_type': "gain",
            'metric': "ndcg",
            'num_leaves': 20,
            'learning_rate': 0.04,
            'max_depth': -1,
            # NOTE: You might consider setting this parameter to a higher value equal to 
            # the number of CPUs on your machine for faster training
            "n_jobs": 1, 
        }

        if params:
            default_params.update(params)

        if model_type == 'LGBM':
            self.model = lightgbm.LGBMRanker(params)
        elif model_type == 'XGB':
            self.model = xgb.XGBRanker(tree_method="hist", objective="rank:ndcg", lambdarank_pair_method="topk")

        # TODO: initialize the LGBMRanker with the provided parameters and assign as a field of this class        

    def fit(self,  X_train, y_train, qgroups_train):
        self.model.fit(X = X_train, y = y_train, group = qgroups_train)

        """
        Trains the LGBMRanker model. (Change to XGBoost)

        Args:
            X_train (array-like): Training input samples.
            y_train (array-like): Target values.
            qgroups_train (array-like): Query group sizes for training data.

        Returns:
            self: Returns the instance itself.
        """
        
        # TODO: fit the LGBMRanker's parameters using the provided features and labels
        return self.model

    def predict(self, featurized_docs):

        result = self.model.predict(featurized_docs)

        return result

