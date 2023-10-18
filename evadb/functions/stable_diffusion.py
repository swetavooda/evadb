# coding=utf-8
# Copyright 2018-2023 EvaDB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from PIL import Image
import string
import torch
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from evadb.catalog.catalog_type import NdArrayType
from evadb.configuration.configuration_manager import ConfigurationManager
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import forward, setup
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe
from evadb.utils.generic_utils import try_to_import_replicate


class StableDiffusion(AbstractFunction):
    def __init__(self):
        self.cache = {}  # Initialize an empty dictionary for caching results
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

    @property
    def name(self) -> str:
        return "StableDiffusion"

    def setup(
        self,
    ) -> None:
        pass

    @staticmethod
    def preprocess_prompt(prompt):
        # Perform preprocessing tasks here (e.g., converting to lowercase, removing punctuation, stemming)
        # ...

        # For example, lowercasing and removing punctuation:
        processed_prompt = prompt.lower().translate(str.maketrans("", "", string.punctuation))
        return processed_prompt

    def calculate_embedding(self, prompts):
        tokenized_prompts = self.tokenizer(
            prompts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.bert_model(**tokenized_prompts)
        embeddings = outputs.pooler_output.numpy()

        # Normalize embeddings
        normalized_embeddings = normalize(embeddings)

        return normalized_embeddings

    @staticmethod
    def calculate_cosine_similarity(query_embedding, cached_embeddings):
        # print("in cosine similarity")
        # Calculate cosine similarity between the query embedding and cached embeddings
        similarities = cosine_similarity(query_embedding, cached_embeddings)
        #print("returning cosine similarity", similarities)
        return similarities

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["prompt"],
                column_types=[
                    NdArrayType.STR,
                ],
                column_shapes=[(None,)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["response"],
                column_types=[
                    NdArrayType.FLOAT32
                ],
                column_shapes=[(None, None, 3)],
            )
        ],
    )
    def forward(self, text_df):
        prompt = text_df[text_df.columns[0]].iloc[0]
        processed_prompt = self.preprocess_prompt(prompt)
        query_embedding = self.calculate_embedding([processed_prompt])

        if self.cache:
            cached_prompts = list(self.cache.keys())
            cached_embeddings = self.calculate_embedding(cached_prompts)

            # Normalize embeddings
            query_embedding = normalize(query_embedding)
            cached_embeddings = normalize(cached_embeddings)

            # Calculate cosine similarity between the query prompt and cached prompts
            similarities = self.calculate_cosine_similarity(
                query_embedding, cached_embeddings)

            # Find the most similar cached prompt
            most_similar_index = np.argmax(similarities)
            most_similar_prompt = cached_prompts[most_similar_index]
            # print("The processed prompt = ", processed_prompt, "Similar prompt = ",
            #       most_similar_prompt, "the threshold = ", similarities[0, most_similar_index])
            if similarities[0, most_similar_index] > 0.97:
                print(f"Using cached result for prompt : {prompt}")
                return self.cache[most_similar_prompt]

        try_to_import_replicate()
        import replicate

        # Register API key, try configuration manager first
        replicate_api_key = ConfigurationManager().get_value(
            "third_party", "REPLICATE_API_TOKEN"
        )
        # If not found, try OS Environment Variable
        if len(replicate_api_key) == 0:
            replicate_api_key = os.environ.get("REPLICATE_API_TOKEN", "")
        assert (
            len(replicate_api_key) != 0
        ), "Please set your Replicate API key in evadb.yml file (third_party, replicate_api_token) or environment variable (REPLICATE_API_TOKEN)"
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_key

        model_id = (
            replicate.models.get("stability-ai/stable-diffusion").versions.list()[0].id
        )

        def generate_image(text_df: PandasDataframe):
            print("Generating image using stable diffusion for prompt: ", prompt)
            results = []
            queries = text_df[text_df.columns[0]]
            #print("queries:", queries)
            for query in queries:
                output = replicate.run(
                    "stability-ai/stable-diffusion:" + model_id, input={"prompt": query}
                )

                # Download the image from the link
                response = requests.get(output[0])
                image = Image.open(BytesIO(response.content))

                # Convert the image to an array format suitable for the DataFrame
                frame = np.array(image)
                results.append(frame)

            return results

        df = pd.DataFrame({"response": generate_image(text_df=text_df)})
        self.cache[processed_prompt] = df
        return df
