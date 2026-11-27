"""Qwen3-Embedding-8B query encoding and exact vector search over BrowseComp-Plus."""
import json
from pathlib import Path
import pickle

from .assets import BCP_INDEX_REPO, digest


MODEL_REPO = "Qwen/Qwen3-Embedding-8B"
BACKEND = "qwen3-embedding-8b"
VERSION = "browsecomp-plus-qwen3-dense-v1"
DIMENSION = 4096
QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
QUERY_MAX_LENGTH = 8192
SHARDS = tuple(f"{BACKEND}/corpus.shard{i}_of_4.pkl" for i in range(1, 5))


class VectorUnpickler(pickle.Unpickler):
    """Accept the NumPy array primitives used by the official vector shards."""

    def find_class(self, module, name):
        import numpy as np

        if module == "numpy" and name in {"ndarray", "dtype"}:
            return getattr(np, name)
        allowed = {"multiarray": "_reconstruct", "numeric": "_frombuffer"}
        for submodule, member in allowed.items():
            if module in {f"numpy.core.{submodule}", f"numpy._core.{submodule}"} and name == member:
                core = np._core if hasattr(np, "_core") else np.core
                return getattr(getattr(core, submodule), member)
        raise pickle.UnpicklingError(f"unexpected vector shard global: {module}.{name}")


def official_blocks(directory, provenance, verify=False):
    for name in SHARDS:
        filename = directory / name
        if verify and digest(filename) != provenance["files"][name]:
            raise ValueError(f"downloaded vector shard checksum changed: {filename}")
        with filename.open("rb") as stream:
            vectors, ids = VectorUnpickler(stream).load()
        yield vectors, [str(docid) for docid in ids]


def model_identity(directory):
    directory = Path(directory)
    metadata = directory / "download_manifest.json"
    if not (directory / "config.json").is_file():
        raise ValueError("EMBEDDING_MODEL_PATH must point to a downloaded local embedding model")
    manifest = json.loads(metadata.read_text(encoding="utf-8"))
    if manifest.get("repo_id") != MODEL_REPO or not manifest.get("revision"):
        raise ValueError(f"embedding model must be a versioned {MODEL_REPO} snapshot")
    return dict(repo_id=MODEL_REPO, revision=manifest["revision"], manifest_sha256=digest(metadata))


class Encoder:
    def __init__(self, directory, device):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("embedding device must be cpu or cuda[:ordinal]")
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, padding_side="left")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(directory, local_files_only=True, torch_dtype=dtype)
        if self.model.config.hidden_size != DIMENSION:
            raise ValueError("unexpected Qwen3-Embedding-8B hidden size")
        self.model.to(self.device).eval()

    def encode(self, texts, max_length):
        torch = self.torch
        features = self.tokenizer(texts, padding=True, truncation=True,
                                  max_length=max_length, return_tensors="pt").to(self.device)
        # With left padding the final position is the last non-padding token of each input.
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16,
                                                    enabled=self.device.type == "cuda"):
            last = self.model(**features, use_cache=False).last_hidden_state[:, -1]
            return torch.nn.functional.normalize(last, p=2, dim=-1).float().cpu().numpy()


def read_vector_manifest(directory, corpus, model_dir, verify=False):
    directory = Path(directory)
    if not (directory / "manifest.json").is_file() and (directory / "download_manifest.json").is_file():
        upstream = json.loads((directory / "download_manifest.json").read_text(encoding="utf-8"))
        if (upstream.get("repo") != BCP_INDEX_REPO or not upstream.get("revision")
                or set(upstream.get("files", {})) != set(SHARDS)):
            raise ValueError("vector index directory is not an official BrowseComp-Plus snapshot")
        if not corpus["canonical_corpus"]:
            raise ValueError("official vectors require the canonical, complete BrowseComp-Plus corpus")
        if upstream["revision"] != corpus["assets"].get("index_revision"):
            raise ValueError("official vector shards must share the pinned index revision")
        for name in SHARDS:
            filename = directory / name
            if not filename.is_file():
                raise FileNotFoundError(f"missing official vector shard: {filename}")
            if verify and digest(filename) != upstream["files"][name]:
                raise ValueError(f"downloaded vector shard checksum changed: {filename}")
        return dict(version=VERSION, benchmark="browsecomp-plus", source="official", storage="parquet-and-pickle",
                    upstream=upstream, model=model_identity(model_dir), documents=corpus["documents"],
                    corpus_sha256=corpus["corpus_sha256"], dimension=DIMENSION,
                    query_prefix=QUERY_PREFIX, query_max_length=QUERY_MAX_LENGTH)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("version") != VERSION or manifest.get("benchmark") != "browsecomp-plus"
            or manifest.get("corpus_sha256") != corpus["corpus_sha256"]
            or manifest.get("documents") != corpus.get("documents", corpus.get("count"))
            or manifest.get("dimension") != DIMENSION):
        raise ValueError("vector index and SQLite corpus have different provenance or dimensions")
    if manifest["model"] != model_identity(model_dir):
        raise ValueError("query encoder snapshot does not match the vector index manifest")
    expected_revision = corpus.get("assets", {}).get("index_revision")
    if manifest.get("source") == "official" and expected_revision and manifest["upstream"]["revision"] != expected_revision:
        raise ValueError("official vector index and Lucene BM25 must share the pinned index revision")
    if manifest.get("query_prefix") != QUERY_PREFIX or manifest.get("query_max_length") != QUERY_MAX_LENGTH:
        raise ValueError("query encoding configuration differs from the vector index manifest")
    if verify:
        for filename in ("vectors.npy", "docids.json"):
            if digest(directory / filename) != manifest["files"][filename]:
                raise ValueError(f"vector index checksum changed: {filename}")
    return manifest


class DenseIndex:
    """Search official vector shards or a prepared vector matrix by inner product."""

    def __init__(self, documents, directory, model_dir, device, manifest):
        import numpy as np
        try:
            import faiss
        except ImportError as error:
            raise RuntimeError("dense retrieval requires FAISS; install requirements.browsecomp-plus.txt") from error

        self.documents = documents
        directory = Path(directory)
        direct = manifest.get("storage") == "parquet-and-pickle"
        if direct:
            blocks = official_blocks(directory, manifest["upstream"])
        else:
            vectors = np.load(directory / "vectors.npy", mmap_mode="r", allow_pickle=False)
            ids = json.loads((directory / "docids.json").read_text(encoding="utf-8"))
            blocks = [(vectors, ids)]
        self.docids = []
        self.index = faiss.IndexFlatIP(DIMENSION)
        for vectors, ids in blocks:
            if direct:
                vectors = np.ascontiguousarray(vectors, dtype=np.float32)
            if vectors.shape != (len(ids), DIMENSION) or vectors.dtype != np.float32:
                raise ValueError("invalid dense vector block shape or dtype")
            self.index.add(vectors)
            self.docids.extend(ids)
        docids = set(self.docids)
        if len(self.docids) != manifest["documents"] or len(docids) != len(self.docids):
            raise ValueError("invalid dense index shape, dtype or document IDs")
        if direct and docids != documents.by_id.keys():
            raise ValueError("official vector document IDs do not match the Parquet corpus")
        self.encoder = Encoder(model_dir, device)

    def search(self, query, topk=5):
        return [self.get(hit['docid']) for hit in self.search_with_scores(query, topk)]

    def search_with_scores(self, query, topk=5):
        import numpy as np

        if not isinstance(query, str) or not query.strip() or topk < 1:
            raise ValueError("query must be text and topk must be positive")
        vector = self.encoder.encode([QUERY_PREFIX + query], QUERY_MAX_LENGTH)
        scores, ranking = self.index.search(np.ascontiguousarray(vector, dtype=np.float32), min(topk, len(self.docids)))
        return [dict(self.get(self.docids[i]), score=float(score)) for score, i in zip(scores[0], ranking[0])]

    def get(self, docid):
        return self.documents.get(docid)
