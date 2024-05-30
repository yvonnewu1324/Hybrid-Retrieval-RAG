from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from getpass import getpass
import os
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.joiners import DocumentJoiner
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.utils import ComponentDevice
from haystack import Pipeline
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack import Document
from datasets import load_dataset
from haystack.document_stores.in_memory import InMemoryDocumentStore

# Initializing the DocumentStore
document_store = InMemoryDocumentStore()


# Load data
dataset = load_dataset("allenai/sciq", split="train")

docs = []
for doc in dataset:
    docs.append(
        Document(content=doc["support"], meta={
                 "question": doc["question"], "correct_answer": doc["correct_answer"]})
    )


# Indexing Documents with a Pipeline
document_splitter = DocumentSplitter(
    split_by="word", split_length=512, split_overlap=32)
document_embedder = SentenceTransformersDocumentEmbedder(
    model="BAAI/bge-small-en-v1.5", device=ComponentDevice.from_str("cuda:0")
)
document_writer = DocumentWriter(document_store)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("document_splitter", document_splitter)
indexing_pipeline.add_component("document_embedder", document_embedder)
indexing_pipeline.add_component("document_writer", document_writer)

indexing_pipeline.connect("document_splitter", "document_embedder")
indexing_pipeline.connect("document_embedder", "document_writer")

indexing_pipeline.run({"document_splitter": {"documents": docs}})


# Creating a Pipeline for Hybrid Retrieval
# 1) Initialize Retrievers and the Embedder
text_embedder = SentenceTransformersTextEmbedder(
    model="BAAI/bge-small-en-v1.5", device=ComponentDevice.from_str("cuda:0")
)
embedding_retriever = InMemoryEmbeddingRetriever(document_store)
bm25_retriever = InMemoryBM25Retriever(document_store)

# 2) Join Retrieval Results
document_joiner = DocumentJoiner()

# 3) Rank the Results
ranker = TransformersSimilarityRanker(model="BAAI/bge-reranker-base")
# 4) Add promt builder and generator
os.environ["OPENAI_API_KEY"] = getpass("Enter OpenAI API key:")
generator = OpenAIGenerator(model="gpt-3.5-turbo")
template = """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""

prompt_builder = PromptBuilder(template=template)
# 5) Create the Hybrid Retrieval Pipeline
hybrid_retrieval = Pipeline()
hybrid_retrieval.add_component("text_embedder", text_embedder)
hybrid_retrieval.add_component("embedding_retriever", embedding_retriever)
hybrid_retrieval.add_component("bm25_retriever", bm25_retriever)
hybrid_retrieval.add_component("document_joiner", document_joiner)
hybrid_retrieval.add_component("ranker", ranker)
hybrid_retrieval.add_component("prompt_builder", prompt_builder)
hybrid_retrieval.add_component("llm", generator)

hybrid_retrieval.connect("text_embedder", "embedding_retriever")
hybrid_retrieval.connect("bm25_retriever", "document_joiner")
hybrid_retrieval.connect("embedding_retriever", "document_joiner")
hybrid_retrieval.connect("document_joiner", "ranker")
hybrid_retrieval.connect("ranker", "prompt_builder.documents")
hybrid_retrieval.connect("prompt_builder", "llm")

# 6) Visualize the Pipeline (Optional)
hybrid_retrieval.draw("hybrid-retrieval-RAG.png")


question = "What type of organism is commonly used in preparation of foods such as cheese and yogurt?"

response = hybrid_retrieval.run({"text_embedder": {"text": question}, "bm25_retriever": {
                                "query": question}, "ranker": {"query": question}, "prompt_builder": {"question": question}})

print(response["llm"]["replies"][0])

"""## Testing the Hybrid Retrieval

Pass the query to `text_embedder`, `bm25_retriever` and `ranker` and run the retrieval pipeline:

"""


def pretty_print_results(prediction):
    for doc in prediction["documents"]:
        print(doc.meta["question"], "\t", doc.score)
        print(doc.meta["correct_answer"])
        print("\n", "\n")


print("ranker ranking")
print("response")
#pretty_print_results(response)
print(response)
