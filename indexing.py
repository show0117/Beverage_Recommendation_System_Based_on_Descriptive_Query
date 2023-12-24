from enum import Enum
import json
import os
from tqdm import tqdm
from collections import Counter, defaultdict
import shelve
from document_preprocessor import Tokenizer
import gzip

 
'''
Here you will be implemeting the indexing strategies for your search engine. You will need to create, persist and load the index.
This will require some amount of file handling.
Use libraries such as tqdm, orjson, collections.Counter, shelve if you need them.
DO NOT use the pickle module.
NOTE: 
There are a few changes to the indexing file for HW2.
The changes are marked with a comment `# NOTE: changes in this method for HW2`. 
Please see more in the README.md.
'''
from enum import Enum
import json
import os
from tqdm import tqdm
from collections import Counter, defaultdict
from document_preprocessor import Tokenizer
import gzip


class IndexType(Enum):
    InvertedIndex = 'BasicInvertedIndex'
    PositionalIndex = 'PositionalIndex'
    OnDiskInvertedIndex = 'OnDiskInvertedIndex'
    SampleIndex = 'SampleIndex'


class InvertedIndex:
    def __init__(self) -> None:
        self.statistics = {'vocab': {}, 'total_tokens': 0, 'num_documents': 0}  # the central statistics of the index
        # metadata like length, number of unique tokens of the documents
        self.document_metadata = {}
        self.index = {}  # the index
        self.vocabulary = set()  

    def remove_doc(self, docid: int) -> None:
        if docid in self.document_metadata.keys():
            self.statistics['num_documents'] -= 1
            tokens = self.document_metadata[docid]['tokens']
            self.statistics['total_tokens'] -= len(tokens)
            
            del self.document_metadata[docid]
            
            for token in tokens:
                if token in self.index.keys():
                    if (len(self.index[token])==1):
                        del self.index[token]
                        self.vocabulary.discard(token)
                        self.statistics['vocab'][token] = 0
                    else:
                        self.index[token].remove(docid)
                        self.statistics['vocab'][token] -= 1

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        tokens = [token.lower() for token in tokens]
        if docid not in self.document_metadata:
            self.document_metadata[docid] = {'tokens': tokens, 'length': len(tokens), 'num_unique_token': len(set(tokens))}
            self.statistics['num_documents'] += 1
        else:
            self.document_metadata[docid]['tokens'] += tokens
            self.document_metadata[docid]['length'] += len(tokens)
            self.document_metadata[docid]['num_unique_token'] += len(set(tokens))
        self.statistics['total_tokens'] += len(tokens)
        
        for token in set(tokens):
            if token in self.index.keys():
                self.index[token].append(docid)
            else:
                self.index[token] = [docid]
                
        for token in tokens:
            if token != 'nonexxx':
                if token not in self.vocabulary:
                    self.vocabulary.add(token)
                    self.statistics['vocab'][token] = 1
                else:
                    self.statistics['vocab'][token] += 1

    def get_postings(self, term) -> list:
        postings = []
        if term in self.index:
            for docid in self.index[term]:
                postings.append((docid, self.document_metadata[docid]['tokens'].count(term)))
        sorted_postings = sorted(postings, key=lambda x: x[0])
        return sorted_postings

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        doc_metadata = self.document_metadata.get(doc_id, {})
        if doc_metadata == {}:
            length = 0
            num_unique_token = 0
        else:
            length = self.document_metadata[doc_id]['length']
            num_unique_token = self.document_metadata[doc_id]['num_unique_token']

        return {'length':length, 'num_unique_token':num_unique_token}

    def get_term_metadata(self, term_id: int) -> dict[str, int]:
        num_docs = len(self.index[term_id])
        return {'term_frequency': term_frequency, 'num_docs': num_docs}

    def get_statistics(self) -> dict[str, int]:
        unique_token_count = len(self.vocabulary)
        total_token_count = self.statistics['total_tokens']
        number_of_documents = self.statistics['num_documents']
        if umber_of_documents > 0:
            mean_document_length = total_token_count / number_of_documents
        else:
            mean_document_length = 0
                            
        result = {'unique_token_count':unique_token_count,
                  'total_token_count': total_token_count,
                  'number_of_documents': number_of_documents,
                  'mean_document_length': mean_document_length}
                            
        return result

    # NOTE: changes in this method for HW2    
    def save(self, index_directory_name) -> None:
                
        inverted_index = {'statistics': self.statistics, 'index': self.index, 'vocabulary': list(self.vocabulary), 'document_metadata': self.document_metadata}
        if not os.path.exists(index_directory_name):
            os.makedirs(index_directory_name)
        
        filepath = os.path.join(index_directory_name, 'inverted_index.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(inverted_index, f, ensure_ascii=False, indent=4)
            

    # NOTE: changes in this method for HW2
    def load(self, index_directory_name) -> None:
        # TODO load the index files from disk to a Python object
        filepath = os.path.join(index_directory_name, 'inverted_index.json')
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                inverted_index = json.load(f)
                self.statistics = inverted_index['statistics']
                self.index = inverted_index['index']
                self.vocabulary = set(inverted_index['vocabulary'])
                self.document_metadata = inverted_index['document_metadata']
                self.document_metadata = {int(k):v for k,v in self.document_metadata.items()}
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            return None
        except json.JSONDecodeError:
            print(f"Invalid JSON format in file: {filepath}")
            return None   


class BasicInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'
        
    def add_doc(self, docid: int, tokens: list[str]) -> None:
        tokens = [token.lower() for token in tokens]
        if docid not in self.document_metadata:
            self.document_metadata[docid] = {'tokens': tokens, 'length': len(tokens), 'num_unique_token': len(set(tokens))}
            self.statistics['num_documents'] += 1
        else:
            self.document_metadata[docid]['tokens'] += tokens
            self.document_metadata[docid]['length'] += len(tokens)
            self.document_metadata[docid]['num_unique_token'] += len(set(tokens))
        self.statistics['total_tokens'] += len(tokens)
        
        for token in set(tokens):
            if token in self.index.keys():
                self.index[token].append(docid)
            else:
                self.index[token] = [docid]
                
        for token in tokens:
            if token != 'nonexxx':
                if token not in self.vocabulary:
                    self.vocabulary.add(token)
                    self.statistics['vocab'][token] = 1
                else:
                    self.statistics['vocab'][token] += 1


    
    def remove_doc(self, docid) -> None:
        if docid in self.document_metadata.keys():
            self.statistics['num_documents'] -= 1
            tokens = self.document_metadata[docid]['tokens']
            self.statistics['total_tokens'] -= len(tokens)
            
            del self.document_metadata[docid]
            
            for token in tokens:
                if token in self.index.keys():
                    if (len(self.index[token])==1):
                        del self.index[token]
                        self.vocabulary.discard(token)
                        self.statistics['vocab'][token] = 0
                    else:
                        self.index[token].remove(docid)
                        self.statistics['vocab'][token] -= 1
                        
            
            
    def get_postings(self, term) -> list:
        postings = []
        if term in self.index:
            for docid in self.index[term]:
                postings.append((docid, self.document_metadata[docid]['tokens'].count(term)))
        sorted_postings = sorted(postings, key=lambda x: x[0])
        return sorted_postings

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        doc_metadata = self.document_metadata.get(doc_id, {})
        if doc_metadata == {}:
            length = 0
            num_unique_token = 0
        else:
            length = self.document_metadata[doc_id]['length']
            num_unique_token = self.document_metadata[doc_id]['num_unique_token']

        return {'length':length, 'num_unique_token':num_unique_token}
    

    def get_term_metadata(self, term_id: int) -> dict[str, int]:
        num_docs = len(self.index[term_id])
        term_frequency = self.statistics['vocab'][term_id]
        return {'term_frequency': term_frequency, 'num_docs': num_docs}

    def get_statistics(self) -> dict[str, int]:
        unique_token_count = len(self.vocabulary)
        total_token_count = self.statistics['total_tokens']
        number_of_documents = self.statistics['num_documents']
        if number_of_documents > 0:
            mean_document_length = total_token_count / number_of_documents
        else:
            mean_document_length = 0
                            
        result = {'unique_token_count':unique_token_count, 'total_token_count': total_token_count,
                  'number_of_documents': number_of_documents, 'mean_document_length': mean_document_length}
                            
        return result

    # NOTE: changes in this method for HW2    
    def save(self, index_directory_name) -> None:
                
        inverted_index = {'statistics': self.statistics, 'index': self.index, 'vocabulary': list(self.vocabulary), 'document_metadata': self.document_metadata}
        if not os.path.exists(index_directory_name):
            os.makedirs(index_directory_name)
        
        filepath = os.path.join(index_directory_name, 'inverted_index.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(inverted_index, f, ensure_ascii=False, indent=4)
            

    # NOTE: changes in this method for HW2
    def load(self, index_directory_name) -> None:
        # TODO load the index files from disk to a Python object
        filepath = os.path.join(index_directory_name, 'inverted_index.json')
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                inverted_index = json.load(f)
                self.statistics = inverted_index['statistics']
                self.index = inverted_index['index']
                self.vocabulary = set(inverted_index['vocabulary'])
                self.document_metadata = inverted_index['document_metadata']
                self.document_metadata = {int(k):v for k,v in self.document_metadata.items()}
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            return None
        except json.JSONDecodeError:
            print(f"Invalid JSON format in file: {filepath}")
            return None   

class PositionalInvertedIndex(BasicInvertedIndex):
    '''
     This is the positional inverted index where each term keeps track of documents and positions of the terms occring in the document.
    '''
    def __init__(self, index_name) -> None:
        super().__init__(index_name)
        self.statistics['index_type'] = 'PositionalInvertedIndex'
        # for example, you can initialize the index and statistics here:
        # self.statistics['offset'] = [0]
        # self.statistics['docmap'] = {}
        # self.doc_id = 0
        # self.postings_id = -1

    # TODO: Do nothing, unless you want to explore using a positional index for some cool features



class OnDiskInvertedIndex(BasicInvertedIndex):
    '''
    This is an inverted index where the inverted index's keys (words) are kept in memory but the
    postings (list of documents) are on desk. The on-disk part is expected to be handled via
    a library.
    '''
    def __init__(self, shelve_filename) -> None:
        super().__init__()
        self.shelve_filename = shelve_filename
        self.statistics['index_type'] = 'OnDiskInvertedIndex'
        # # Ensure that the directory exists        
        # self.index = shelve.open(self.shelve_filename, 'index')
        # self.statistics['docmap'] = {}
        # self.doc_id = 0

    # NOTE: Do nothing, unless you want to re-experience the pain of cross-platform compatibility :'( 

class Indexer:
    '''The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    '''
    @staticmethod
    def create_index(index_type: IndexType, dataset_path: str, 
                     document_preprocessor: Tokenizer, stopwords: set[str], 
                     minimum_word_frequency: int, text_key="text",
                     max_docs=-1, doc_augment_dict: dict[int, list[str]] | None = None) -> InvertedIndex:
        
        if index_type == IndexType.InvertedIndex:
            inverted_index = BasicInvertedIndex()
        elif index_type == IndexType.PositionalIndex:
            inverted_index = PositionalInvertedIndex()
        elif index_type == IndexType.OnDiskInvertedIndex:
            inverted_index = OnDiskInvertedIndex()

        with open(dataset_path, 'r', encoding='utf-8') as f:
            files = f.readlines()
        
        stopwords = [i.strip().lower() for i in stopwords]

        overall_tokens = []

        for file in files:
            tokens = document_preprocessor.tokenize(json.loads(file)[text_key].lower())
            overall_tokens = overall_tokens + tokens
            for i in range(len(tokens)):
                token = tokens[i]
                if token in stopwords:
                    tokens[i] = 'nonexxx'

            if minimum_word_frequency > 0:
                token_frequency = Counter(overall_tokens)
                tokens = [token if token_frequency[token] >= minimum_word_frequency else 'nonexxx' for token in tokens]
                print(token_frequency)

            inverted_index.add_doc(json.loads(file)['docid'], tokens)

        if doc_augment_dict != None:
            for docid in doc_augment_dict:
                for query in doc_augment_dict[docid]:
                    tokens = document_preprocessor.tokenize(query.lower())
                    overall_tokens = overall_tokens + tokens
                    for i in range(len(tokens)):
                        token = tokens[i]
                        if token in stopwords:
                            tokens[i] = 'nonexxx'

                    if minimum_word_frequency > 0:
                        token_frequency = Counter(overall_tokens)
                        tokens = [token if token and token_frequency[token] >= minimum_word_frequency else 'nonexxx' for token in tokens]
                        print(token_frequency)
                    inverted_index.add_doc(docid, tokens)    

        return inverted_index
    