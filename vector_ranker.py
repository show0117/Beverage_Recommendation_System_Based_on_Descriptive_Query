from sentence_transformers import SentenceTransformer, util
from numpy import ndarray
from ranker import Ranker


class VectorRanker(Ranker):
    def __init__(self, bi_encoder_model_name: str, encoded_docs: ndarray,
                 row_to_docid: list[int]) -> None:
        """
        Initializes a VectorRanker object.

        Args:
            bi_encoder_model_name: The name of a huggingface model to use for initializing a 'SentenceTransformer'
            encoded_docs: A matrix where each row is an already-encoded document, encoded using the same encoded
                as specified with bi_encoded_model_name
            row_to_docid: A list that is a mapping from the row number to the document id that row corresponds to
                the embedding

        Using zip(encoded_docs, row_to_docid) should give you the mapping between the docid and the embedding.
        """
        # Use device='cpu' when doing model instantiation (for AG)
        # If you know what the parameter does, feel free to play around with it
        # TODO: Instantiate the bi-encoder model here
        self.model = SentenceTransformer(bi_encoder_model_name)
        self.encoded_docs = encoded_docs
        self.row_to_docid = row_to_docid

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Encodes the query and then scores the relevance of the query with all the documents.

        Args:
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            A sorted list of tuples containing the document id and its relevance to the query,
            with most relevant documents first
        """
        # NOTE: Do not forget to handle edge cases

        # TODO: Encode the query using the bi-encoder

        # TODO: Score the similarity of the query vector and document vectors for relevance
        # Calculate the dot products between the query embedding and all document embeddings
        
        # TODO: Generate the ordered list of (document id, score) tuples

        # TODO: Sort the list so most relevant are first
        encoded_query = self.model.encode(query)
        scores = []
        for encoded_doc, docid in zip(self.encoded_docs, self.row_to_docid):
            score = sum(encoded_query * encoded_doc)
            scores.append((docid,score))
        scores.sort(key=lambda a: a[1], reverse=True)
        
        return scores
        

