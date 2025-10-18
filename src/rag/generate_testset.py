"""Generate a Golden Synthetic Testset Dataset using RAGAS"""
import os
import glob
import json
from loguru import logger
from typing import Any, List, Dict
from dotenv import load_dotenv

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset.synthesizers import (
    default_query_distribution, 
    SingleHopSpecificQuerySynthesizer, 
    MultiHopAbstractQuerySynthesizer, 
    MultiHopSpecificQuerySynthesizer
)

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

from src.utils.document_utils import json_to_document, batch_files

load_dotenv()

KG_FILEPATH = 'data/kg/financial_crimes_kg.json'

class GenerateSyntheticTestset:
    def __init__(
            self, 
            generator_llm: LangchainLLMWrapper = None,
            generator_embeddings: LangchainEmbeddingsWrapper = None
        ):
        self.generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))
        self.generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
        self.query_distribution = [
                (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.5),
                (MultiHopAbstractQuerySynthesizer(llm=generator_llm), 0.25),
                (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.25),
        ]
        self.kg = KnowledgeGraph()
    
    def convert_json_to_document(self) -> List[Document]:
        """Convert JSON files to Langchain Documents"""
        all_documents = []
        for batch_file in batch_files[:1]:
            with open(batch_file, 'r') as f:
                data = json.load(f)
                releases = data["releases"][:20]
            documents = []
            for release in releases:
                document = json_to_document(release)
                if document:
                    documents.append(document)
            all_documents.extend(documents)
        logger.info(f"Total documents {len(all_documents)}")
        return all_documents
    
    def convert_document_to_kg_node(self) -> List[Node]:
        """Convert Langchain Document to RAGAS Node"""
        docs = self.convert_json_to_document()
        for doc in docs:
            self.kg.nodes.append(
                Node(
                    type=NodeType.DOCUMENT,
                    properties={
                        "page_content": doc.page_content,
                        "document_metadata": doc.metadata
                    }
                )
        )
    
    def transform_kg(self, docs: List[Node]) -> None:
        """Transform kg to extract entities"""
        def_transforms = default_transforms(
            documents=docs, 
            llm=self.generator_llm, 
            embedding_model=self.generator_embeddings
        )
        
        apply_transforms(
            self.kg,
            def_transforms
        )
    
    def save_transform_kg(self, filepath: str = KG_FILEPATH):
        """Save Knowledge Graph metadata"""
        os.makedirs(filepath, exist_ok=True)
        self.kg.save(filepath)
    
    # def generate_testset(self):
    #     """Generate RAGAS testset using kg"""
    #     generator = TestsetGenerator(
    #         llm=self.generator_llm,
    #         embedding_model=self.generator_embeddings,
    #         knowledge_graph=KG_FILEPATH
    #     )
    #     testset = generator.generate(testset_size=10, query_distribution=self.query_distribution)
    #     return testset
    
    def generate_testset(self, docs: List[Node], testset_size: int = 10):
        """Generat test queries using default distribution"""
        generator = TestsetGenerator(llm=self.generator_llm, embedding_model=self.generator_embeddings)
        testset = generator.generate_with_langchain_docs(
            docs,
            testset_size=testset_size,
        )
        return testset
    
def main():

    os.makedirs("evaluation", exist_ok=True)

    sdg = GenerateSyntheticTestset()
    logger.info("Convert JSON files to Langchain Documents")
    docs = sdg.convert_json_to_document()

    logger.info("Generate RAGAS testset dataset")
    ragas_testset = sdg.generate_testset(docs)

    testset_df = ragas_testset.to_pandas()
    logger.info("Save the RAGAS tesetset as a csv file")
    testset_df.to_csv("evaluation/financial_crimes_testset_ragas.csv", index=False)

    logger.info("Save the RAGAS testset in native format")
    ragas_testset.to_jsonl("evaluation/financial_crimes_testset_ragas.json")

if __name__ == "__main__":
    main()
    # sdg = GenerateSyntheticTestset()

    # # sdg.convert_document_to_kg_node()
    # # ragas_testset_default = sdg.generate_testset()
    # docs = sdg.convert_json_to_document()
    # ragas_testset = sdg.generate_testset(docs)
    # df = ragas_testset.to_pandas()
    # logger.info(f"Size of RAGAS testset {len(df)}")
    # logger.info(df.show())


