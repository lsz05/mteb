from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


class JQaRARerankingLite(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JQaRARerankingLite",
        dataset={
            "path": "sbintuitions/JMTEB-lite",
            "revision": "main",
            "trust_remote_code": True,
        },
        description=(
            "JQaRA (Japanese Question Answering with Retrieval Augmentation) is a reranking dataset "
            "consisting of questions from JAQKET and corpus from Japanese Wikipedia. This is the lightweight "
            "version with a reduced corpus (172,897 documents) constructed using hard negatives from "
            "5 high-performance models."
        ),
        reference="https://huggingface.co/datasets/hotchpotch/JQaRA",
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2025-01-01"),
        domains=["Encyclopaedic", "Non-fiction", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=["jpn-Jpan"],
        sample_creation="found",
        bibtex_citation=r"""
@misc{yuichi-tateno-2024-jqara,
  author = {Yuichi Tateno},
  title = {JQaRA: Japanese Question Answering with Retrieval Augmentation
  - 検索拡張(RAG)評価のための日本語Q&Aデータセット},
  url = {https://huggingface.co/datasets/hotchpotch/JQaRA},
}
@misc{jmteb_lite,
  author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan and Fukuchi, Akihiko and Shibata, Tomohide
  and Kawahara, Daisuke},
  title = {{J}{M}{T}{E}{B}-lite: {T}he {L}ightweight {V}ersion of {JMTEB}},
  howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB-lite}},
  year = {2025},
}
""",
    )

    def load_data(self, **kwargs):
        """Load JMTEB-lite dataset."""
        if self.data_loaded:
            return

        for split in self.metadata.eval_splits:
            # Load queries and corpus separately
            queries = load_dataset(
                self.metadata.dataset["path"],
                "jqara-query",
                split=split,
                trust_remote_code=True,
            )
            corpus = load_dataset(
                self.metadata.dataset["path"],
                "jqara-corpus",
                split="corpus",
                trust_remote_code=True,
            )

            relevant_docs = {}
            top_ranked = {}
            all_retrieved_doc_ids = set()

            queries_list = []
            for i, query in enumerate(queries):
                # Generate sequential query IDs
                qid = split + "_" + str(i + 1)
                queries_list.append({"id": qid, "text": query["query"]})

                # For reranking: top_ranked contains candidates
                if "retrieved_docs" in query:
                    top_ranked[qid] = [
                        str(doc_id) for doc_id in query["retrieved_docs"]
                    ]
                    all_retrieved_doc_ids.update(query["retrieved_docs"])

                    # relevance_scores indicates which docs are relevant
                    relevant_docs[qid] = {}
                    for doc_id, score in zip(
                        query["retrieved_docs"], query.get("relevance_scores", [])
                    ):
                        relevant_docs[qid][str(doc_id)] = int(score)

            # For reranking, only include docs that are candidates
            corpus_list = []
            for doc in corpus:
                if doc["docid"] in all_retrieved_doc_ids:
                    corpus_list.append(
                        {
                            "id": str(doc["docid"]),
                            "text": doc["text"],
                            "title": doc.get("title", ""),
                        }
                    )

            self.dataset["default"][split] = RetrievalSplitData(
                corpus=Dataset.from_list(corpus_list),
                queries=Dataset.from_list(queries_list),
                relevant_docs=relevant_docs,
                top_ranked=top_ranked,
            )

        self.data_loaded = True
