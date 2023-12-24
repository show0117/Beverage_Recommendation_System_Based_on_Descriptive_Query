"""
This is the template for implementing the rankers for your search engine.
You will be implementing WordCountCosineSimilarity, DirichletLM, TF-IDF, BM25, Pivoted Normalization,
and your own ranker.
"""
import numpy as np
from collections import Counter, defaultdict
from sentence_transformers import CrossEncoder
from indexing import InvertedIndex
from document_preprocessor import RegexTokenizer

class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """
    # TODO: implement this class properly. This is responsible for returning a list of sorted relevant documents.
    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str], 
                 scorer: 'RelevanceScorer', raw_text_dict: dict[int,str]) -> None:
        """
        Initializes the state of the Ranker object.

        NOTE: Previous homeworks had you passing the class of the scorer to this function.
            This has been changed as it created a lot of confusion.
            You should now pass an instantiated RelevanceScorer to this function.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
            raw_text_dict: A dictionary mapping a document ID to the raw string of the document
        """
        self.index = index.index
        self.statistics = index.statistics
        self.get_statistics = index.get_statistics()
        self.vocabulary = index.vocabulary
        self.document_metadata = index.document_metadata 
        self.tokenizer = document_preprocessor
        self.stopwords = stopwords
        self.scorer = scorer
        self.raw_text_dict = raw_text_dict
 
    def query(self, query: str, pseudofeedback_num_docs=0, pseudofeedback_alpha=0.8,
              pseudofeedback_beta=0.2) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).
 
        Args:
            query: The query to search for
            pseudofeedback_num_docs: If pseudo-feedback is requested, the number
                 of top-ranked documents to be used in the query,
            pseduofeedback_alpha: If pseudo-feedback is used, the alpha parameter for weighting
                how much to include of the original query in the updated query
            pseduofeedback_beta: If pseudo-feedback is used, the beta parameter for weighting
                how much to include of the relevant documents in the updated query
 
        Returns:
            A list containing tuples of the documents (ids) and their relevance scores

        NOTE: We are standardizing the query output of Ranker to match with L2RRanker.query and VectorRanker.query
        The query function should return a sorted list of tuples where each tuple has the first element as the document ID
        and the second element as the score of the document after the ranking process.
        """
        # TODO: Tokenize the query and remove stopwords, if needed

        # TODO (HW4): If the user has indicated we should use feedback,
        #  create the pseudo-document from the specified number of pseudo-relevant results.
        #  This document is the cumulative count of how many times all non-filtered words show up
        #  in the pseudo-relevant documents. See the equation in the write-up. Be sure to apply the same
        #  token filtering and normalization here to the pseudo-relevant documents.

        # TODO (HW4): Combine the document word count for the pseudo-feedback with the query to create a new query
        # NOTE (HW4): Since you're using alpha and beta to weight the query and pseudofeedback doc, the counts
        #  will likely be *fractional* counts (not integers) which is ok and totally expected.

        # TODO: Fetch a list of possible documents from the index and create a mapping from
        #  a document ID to a dictionary of the counts of the query terms in that document.
        #  You will pass the dictionary to the RelevanceScorer as input.

        # TODO: Rank the documents using a RelevanceScorer

        # TODO: Return the **sorted** results as format [{docid: 100, score:0.5}, {{docid: 10, score:0.2}}]
        if (pseudofeedback_num_docs == 0):
            results = []
            query = query.lower()
            tokenized_query = self.tokenizer.tokenize(query)
            for i in range(len(tokenized_query)):
                token = tokenized_query[i]
                if token in self.stopwords:
                    tokenized_query[i] = 'nonexxx'
            query_word_counts = Counter(tokenized_query)
            
            docids = []
            for token in tokenized_query:
                if token != 'nonexxx':
                    if token in self.index:
                        docids += list(self.index[token])
            docids = list(set(docids))

            for docid in docids:
                result = (docid, self.scorer.score(docid, {}, query_word_counts))
                results.append(result)
            results.sort(key=lambda a: a[1], reverse=True)
            
            return results
        
        elif (pseudofeedback_num_docs > 0):
            results = []
            query = query.lower()
            tokenized_query = self.tokenizer.tokenize(query)
            for i in range(len(tokenized_query)):
                token = tokenized_query[i]
                if token in self.stopwords:
                    tokenized_query[i] = 'nonexxx'
            q = Counter(tokenized_query)

            docids = []
            for token in tokenized_query:
                if token != 'nonexxx':
                    if token in self.index:
                        docids += list(self.index[token])
            docids = list(set(docids))
 
            for docid in docids:
                result = (docid, self.scorer.score(docid, {}, q))
                results.append(result)
            results.sort(key=lambda a: a[1], reverse=True)

            q = Counter(tokenized_query)

            pseudofeedback_docs = results[0:pseudofeedback_num_docs]
            
            d_i = Counter([])
            for i in range(pseudofeedback_num_docs):
                docid = pseudofeedback_docs[i][0]
                tokens = self.document_metadata[docid]['tokens']
                text = ' '.join(tokens)
                tokens = self.tokenizer.tokenize(text)
                for i in range(len(tokens)):
                    token = tokens[i]
                    if token in self.stopwords:
                        tokens[i] = 'nonexxx'
                d_i += Counter(tokens)

            for key in q.keys():
                q[key] = q[key] * pseudofeedback_alpha

            for key in d_i.keys():
                d_i[key] = d_i[key] * pseudofeedback_beta / pseudofeedback_num_docs
            
            for key in d_i.keys():
                if (key in q):
                    q[key] += d_i[key]
                else:
                    q[key] = d_i[key]

            docids = []
            for token in q.keys():
                if token in self.index:
                    docids += list(self.index[token])
            docids = list(set(docids))

            pseudofeedback_results = []

            for docid in docids:
                result = (docid,self.scorer.score(docid, {}, q))
                pseudofeedback_results.append(result)
            
            pseudofeedback_results.sort(key=lambda a: a[1], reverse=True)

            return pseudofeedback_results

class RelevanceScorer:
    '''
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    '''
    # TODO Implement the functions in the child classes (WordCountCosineSimilarity, DirichletLM, BM25, PivotedNormalization, TF_IDF) 
    #      and not in this one
    
    def __init__(self, index, parameters) -> None:
        self.index = index.index
        self.statistics = index.statistics
        self.get_statistics = index.get_statistics()
        self.vocabulary = index.vocabulary
        self.document_metadata = index.document_metadata 
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> dict[str, int]:
        raise NotImplementedError    
    
# TODO: Implement BM25
class BM25(RelevanceScorer):
    def __init__(self, index, parameters: dict = {'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        super().__init__(index, parameters)
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Get necessary information from index

        # 2. Find the dot product of the word count vector of the document and the word count vector of the query

        # 3. For all query parts, compute the TF and IDF to get a score

        # 4. Return score

        score = 0
        try:
            for query in query_word_counts.keys():
                if query in self.index and docid in self.index[query]:
                    N = self.statistics['num_documents']
                    df = len(self.index[query])
                    cd_w = self.document_metadata[docid]['tokens'].count(query)
                    cq_w = query_word_counts[query]
                    avdl = self.get_statistics['mean_document_length']
                    dl = self.document_metadata[docid]['length']
                    tmp = np.log((N-df+0.5)/(df+0.5))*((self.k1+1)*cd_w)/(self.k1*((1-self.b)+self.b*dl/avdl)+cd_w)*((self.k3+1)*cq_w)/(self.k3+cq_w)
                    score += tmp * query_word_counts[query]
        except:
            score = 0

        return score


# TODO (HW4): Implement Personalized BM25
class PersonalizedBM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, relevant_doc_index: InvertedIndex,
                 parameters={'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        """
        Initializes Personalized BM25 scorer.

        Args:
            index: The inverted index used to use for computing most of BM25
            relevant_doc_index: The inverted index of only documents a user has rated as relevant,
                which is used when calculating the personalized part of BM25
            parameters: The dictionary containing the parameter values for BM25

        Returns:
            The Personalized BM25 score
        """
        super().__init__(index, parameters)
        self.index = index.index
        self.statistics = index.statistics
        self.get_statistics = index.get_statistics()
        self.vocabulary = index.vocabulary
        self.document_metadata = index.document_metadata 

        self.rel_index = relevant_doc_index.index
        self.rel_statistics = relevant_doc_index.statistics
        self.rel_get_statistics = relevant_doc_index.get_statistics()
        self.rel_vocabulary = relevant_doc_index.vocabulary
        self.rel_document_metadata = relevant_doc_index.document_metadata 

        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # TODO (HW4): Implement Personalized BM25

        score = 0
        for query in query_word_counts.keys():
            if query in self.index and docid in self.index[query]:
                N = self.statistics['num_documents']
                df = len(self.index[query])
                cd_w = self.document_metadata[docid]['tokens'].count(query)
                cq_w = query_word_counts[query]
                avdl = self.get_statistics['mean_document_length']
                dl = self.document_metadata[docid]['length']
                R = self.rel_statistics['num_documents']
                try:
                    r_i = len(self.rel_index[query])
                except:
                    r_i = 0

                tmp = np.log((r_i+0.5)*(N-df-R+r_i+0.5)/((df-r_i+0.5)*(R-r_i+0.5)))*((self.k1+1)*cd_w)/(self.k1*((1-self.b)+self.b*dl/avdl)+cd_w)*((self.k3+1)*cq_w)/(self.k3+cq_w)
                score += tmp
        
        return score


# TODO: Implement Pivoted Normalization
class PivotedNormalization(RelevanceScorer):
    def __init__(self, index, parameters: dict = {'b': 0.2}) -> None:
        super().__init__(index, parameters)
        self.b = parameters['b']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score

        # 4. Return the score
        score = 0
        
        for query in query_word_counts.keys():
            if query in self.index and docid in self.index[query]:
                cq_w = query_word_counts[query]
                cd_w = self.document_metadata[docid]['tokens'].count(query)
                idf = np.log((self.statistics['num_documents']+1)/len(self.index[query])) 
                avdl = self.get_statistics['mean_document_length']
                dl = self.document_metadata[docid]['length']
                tmp = cq_w*((1+np.log(1+np.log(cd_w)))/(1-self.b+self.b*dl/avdl))*idf
                score += tmp * query_word_counts[query]

        return score

 
# TODO: Implement TF-IDF
class TF_IDF(RelevanceScorer):
    def __init__(self, index, parameters: dict = {}) -> None:
        super().__init__(index, parameters)

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        score = 0
        for query in query_word_counts.keys():
            if query in self.index and docid in self.index[query]:
                tf = np.log(self.document_metadata[docid]['tokens'].count(query)+1)
                idf = np.log(self.statistics['num_documents']/len(self.index[query]))+1
                score += tf*idf*query_word_counts[query]
        return score


class CrossEncoderScorer(RelevanceScorer):
    def __init__(self, raw_text_dict: dict[int, str],
                 cross_encoder_model_name: str = 'cross-encoder/msmarco-MiniLM-L6-en-de-v1') -> None:
        """
        Initializes a CrossEncoderScorer object.

        Args:
            raw_text_dict: A dictionary where the document id is mapped to a string with the first 500 words
                in the document
            cross_encoder_model_name: The name of a cross-encoder model

        NOTE 1: The CrossEncoderScorer class uses a pre-trained cross-encoder model
            from the Sentence Transformers package to score a given query-document pair.

        NOTE 2: This is not a RelevanceScorer object because the method signature for score() does not match,
            but it has the same intent, in practice.
        """
        # TODO: Save any new arguments that are needed as fields of this class

        """
        Initializes a CrossEncoderScorer object.

        Args:
            raw_text_dict: A dictionary where the document id is mapped to a string with the first 500 words
                in the document
            cross_encoder_model_name: The name of a cross-encoder model
        """
        self.raw_text_dict = raw_text_dict
        self.model = CrossEncoder(cross_encoder_model_name)

    def score(self, docid: int, doc_word_counts: dict[str, int], query) -> float:
        """
        Gets the cross-encoder score for the given document.
        
        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The score returned by the cross-encoder model
        """
        # NOTE: Do not forget to handle an edge case
        # (e.g., docid does not exist in raw_text_dict or empty query, both should lead to 0 score)

        # TODO (HW3): Get a score from the cross-encoder model
        #             Refer to IR_Encoder_Examples.ipynb in Demos folder on Canvas if needed

        score = self.model.predict([query, ' '.join(self.raw_text_dict[docid].split(' ')[0:200])])
        return score

class DirichletLM(RelevanceScorer):
    def __init__(self, index, parameters={'mu': 2000}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]):
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query_parts, compute score

        # 4. Return the score
        pass #(`score` should be defined in your code; you can call it whatever you want)

class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index, parameters = {}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]):

        # 1. Find the dot product of the word count vector of the document and the word count vector of the query

        # 2. Return the score
        pass #(`score` should be defined in your code; you can call it whatever you want)


# TODO: Implement your own ranker with proper heuristics
class YourRanker(RelevanceScorer):
    pass


class SampleScorer(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters) -> None:
        pass

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Scores all documents as 10.
        """
        # Print randomly ranked results
        return 10

