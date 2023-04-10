import pandas as pd
import numpy as np
import openai
import tiktoken
import json
import pymongo
from constants import COMPLETIONS_MODEL, EMBEDDING_MODEL


class SearchEngine:

    def __init__(self, openai_key):


        openai.api_key = openai_key
        self.MAX_SECTION_LEN = 500
        self.SEPARATOR = "\n* "
        self.MAX_SECTION_LEN = 500
        self.SEPARATOR = "\n* "
        self.ENCODING = "gpt2" 
        self.encoding = tiktoken.get_encoding(self.ENCODING)
        self.COMPLETIONS_API_PARAMS = {
            # We use temperature of 0.0 because it gives the most predictable, factual answer.
            "temperature": 0.0,
            "max_tokens": 300,
            "model": COMPLETIONS_MODEL,
        }


    def parse_block_data(self, data, dtype):
        if dtype == 'list':
            return ','.join(data['items'])
        if dtype == 'code':
            return data['code']
        if dtype == 'link':
            return data['link']
        return data['text']

    def num_tokens_from_string(self, string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        num_tokens = len(self.encoding.encode(string))
        return num_tokens

    def get_embedding(self, text: str, model: str = EMBEDDING_MODEL) -> list[float]:
        result = openai.Embedding.create(
            model=model,
            input=text
        )
        return result["data"][0]["embedding"]

    def compute_doc_embeddings(self, df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
        """
        Create an embedding for each row in the dataframe using the OpenAI Embeddings API.

        Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
        """
        return {
            idx: self.get_embedding(r.content) for idx, r in df.iterrows()
        }

    def parse_data(self, uri, user_id):
        try:
            self.mongoclient = pymongo.MongoClient(uri)
            self.db = self.mongoclient["Carton"]
        except Exception as e:
            print("Error connecting to MongoDB Atlas:", e)
        obj_coll = self.db["objects"]
        cells_coll = self.db["cells"]
        context = []

        # TODO: Add user ID filter
        objects = obj_coll.find({'userID': user_id})

        for o in objects:
            identifier = str(o['_id'])
            # add title
            context.append([identifier, 'title', o['title']])
            # properties
            for p in o['properties']:
                context.append(
                    [identifier, p['type'] + ' property', p['value']])
            # left + right cells references
            children = o['leftCol']['cellIDs'] + o['rightCol']['cellIDs']

            if len(children) > 0:
                context.append(
                    [identifier, 'relationship with blocks', ','.join(children)])

        cells = cells_coll.find({})
        for c in cells:
            identifier = str(c['_id'])
            if 'data' in c and 'blocks' in c['data']:
                
                blocks = c['data']['blocks']
                for block in blocks:
                    context.append([identifier, block['type'], self.parse_block_data(
                        block['data'], block['type'])])
        df = pd.DataFrame(context, columns=['identifier', 'key', 'value'])
        df['content'] = df['identifier'] + \
            ' has a ' + df['key'] + ' ' + df['value']

        return df

    def vector_similarity(self, x: list[float], y: list[float]) -> float:
        """
        Returns the similarity between two vectors.

        Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
        """
        return np.dot(np.array(x), np.array(y))

    def construct_prompt(self, question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
        """
        Fetch relevant 
        """
        most_relevant_document_sections = self.order_document_sections_by_query_similarity(question, context_embeddings)
        separator_len = len(self.encoding.encode(self.SEPARATOR))
        chosen_sections = []
        chosen_sections_len = 0
        chosen_sections_indexes = []
        
        for _, section_index in most_relevant_document_sections:
            # Add contexts until we run out of space.        
            document_section = df.loc[section_index]
            
            chosen_sections_len += self.num_tokens_from_string(document_section.content, 'gpt2') + separator_len
            if chosen_sections_len > self.MAX_SECTION_LEN:
                break
                
            chosen_sections.append(self.SEPARATOR + document_section.content.replace("\n", " "))
            chosen_sections_indexes.append(str(section_index))
                
        # Useful diagnostic information
        print(f"Selected {len(chosen_sections)} document sections:")
        print("\n".join(chosen_sections_indexes))
        
        header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
        
        return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

    def order_document_sections_by_query_similarity(self, query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
        """
        Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
        to find the most relevant sections. 

        Return the list of document sections, sorted by relevance in descending order.
        """
        query_embedding = self.get_embedding(query)

        document_similarities = sorted([
            (self.vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
        ], reverse=True)

        return document_similarities
    def answer_query_with_context(
        self,
        query: str,
        df: pd.DataFrame,
        document_embeddings: dict[(str, str), np.array],
        show_prompt: bool = False
    ) -> str:
        prompt = self.construct_prompt(
            query,
            document_embeddings,
            df
        )
        
        if show_prompt:
            print(prompt)

        response = openai.Completion.create(
                    prompt=prompt,
                    **self.COMPLETIONS_API_PARAMS
                )

        return response["choices"][0]["text"].strip(" \n")
