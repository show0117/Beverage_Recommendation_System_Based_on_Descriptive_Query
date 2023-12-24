from nltk.tokenize import RegexpTokenizer
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class Tokenizer:
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        self.lowercase = lowercase
        self.multiword_expressions = multiword_expressions
        self.input_tokens = []
    
    def postprocess(self, input_tokens: list[str]) -> list[str]:
        self.input_tokens = input_tokens
        if (self.lowercase==True):
            for i in range(len(self.input_tokens)):
                self.input_tokens[i] = self.input_tokens[i].lower()        
        return self.input_tokens
        
    def tokenize(self, text: str) -> list[str]:
        if (self.lowercase==True):
            text = text.lower()
        split_text = text.split(' ')
        return split_text
        
class RegexTokenizer(Tokenizer):
    def __init__(self, token_regex: str, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        super().__init__(lowercase, multiword_expressions)
        # TODO: Save a new argument that is needed as a field of this class.
        # TODO: Initialize the NLTK's RegexpTokenizer 
        self.token_regex = token_regex
        self.lowercase = lowercase
        self.multiword_expressions = multiword_expressions
        self.tokenizer = RegexpTokenizer('\\w+')

    def tokenize(self, text: str) -> list[str]:
        if (self.lowercase==True):
            text = text.lower()
        tokens = self.tokenizer.tokenize(text)
        return tokens

# TODO (HW3): Take in a doc2query model and generate queries from a piece of text
# Note: This is just to check you can use the models;
#       for downstream tasks such as index augmentation with the queries, use doc2query.csv
class Doc2QueryAugmenter:
    """
    This class is responsible for generating queries for a document.
    These queries can augment the document before indexing.

    MUST READ: https://huggingface.co/doc2query/msmarco-t5-base-v1

    OPTIONAL reading
        1. Document Expansion by Query Prediction (Nogueira et al.): https://arxiv.org/pdf/1904.08375.pdf
    """
    def __init__(self, doc2query_model_name: str = 'doc2query/msmarco-t5-base-v1') -> None:
        """
        Creates the T5 model object and the corresponding dense tokenizer.
        
        Args:
            doc2query_model_name: The name of the T5 model architecture used for generating queries
        """
        self.device = torch.device('cpu')  # Do not change this unless you know what you are doing
        self.tokenizer = T5Tokenizer.from_pretrained(doc2query_model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(doc2query_model_name)

    def get_queries(self, document: str, n_queries: int = 5, prefix_prompt: str = '') -> list[str]:
        """
        Steps
            1. Use the dense tokenizer/encoder to create the dense document vector.
            2. Use the T5 model to generate the dense query vectors (you should have a list of vectors).
            3. Decode the query vector using the tokenizer/decode to get the appropriate queries.
            4. Return the queries.
         
            Ensure you take care of edge cases.
         
        OPTIONAL (DO NOT DO THIS before you finish the assignment):
            Neural models are best performing when batched to the GPU.
            Try writing a separate function which can deal with batches of documents.
        
        Args:
            document: The text from which queries are to be generated
            n_queries: The total number of queries to be generated
            prefix_prompt: An optional parameter
                Some models are not fine-tuned to generate queries.
                So we need to add a prompt to coax the model into generating queries.
                This string enables us to create a prefixed prompt to generate queries for the models.
                Prompt-engineering: https://en.wikipedia.org/wiki/Prompt_engineering
        
        Returns:
            A list of query strings generated from the text
        """
        # Note: Feel free to change these values to experiment
        document_max_token_length = 400  # as used in OPTIONAL Reading 1
        top_p = 0.85

        # TODO: For the given model, generate a list of queries that might reasonably be issued to search
        #       for that document
        # NOTE: Do not forget edge cases
        text = prefix_prompt + document
        input_ids = self.tokenizer.encode(text, max_length=320, truncation=True, return_tensors='pt')
        outputs = self.model.generate(input_ids=input_ids, max_length=document_max_token_length,
                                 do_sample=True, top_p=top_p, num_return_sequences=n_queries)
        query_list = []
        for i in range(len(outputs)):
          query = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
          query_list.append(query)

        return query_list
        

# Don't forget that you can have a main function here to test anything in the file

