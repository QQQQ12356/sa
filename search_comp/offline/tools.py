"""Local search/visit with bounded observations and retrieval traces."""
from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
import json
from urllib.parse import unquote
from search_comp.data.agent_prompt import build_system_prompt

SYSTEM_PROMPT = build_system_prompt(local=True)
HOTPOT_INSTRUCTION = '''For the final answer, return a JSON object inside <answer>...</answer>:
{"answer": "a short exact answer", "sp": [["Document title", 0]]}.
sp contains ONLY sentence references that support your answer, using the displayed zero-based sentence IDs.
If you cannot identify supporting sentences, use an empty sp array. Do not copy whole passages into answer.
'''
BCP_INSTRUCTION = "Give a clear final answer and cite supporting document IDs using [docid]."


class LocalTools:
    def __init__(self, index, tokenizer, topk=5, snippet_tokens=512, read_tokens=4096,
                 allow_visit=True, max_calls=200):
        self.index, self.tokenizer = index, tokenizer
        self.topk, self.snippet_tokens, self.read_tokens = topk, snippet_tokens, read_tokens
        self.allow_visit, self.max_calls = allow_visit, max_calls
        self.reset()

    def reset(self):
        self.retrieved = set()
        self.visited = set()
        self.counts = Counter()
        self.trace = []

    def clip(self, value, budget):
        tokens = self.tokenizer.encode(value, add_special_tokens=False)
        return self.tokenizer.decode(tokens[:budget], skip_special_tokens=False)

    def render(self, doc, budget):
        if doc["sentences"]:
            body = "\n".join(f"[{index}] {sentence}" for index, sentence in enumerate(doc["sentences"]))
        else:
            body = doc["text"]
        return dict(docid=doc["docid"], title=doc["title"], url=doc["url"],
                    source_url=doc["source_url"], text=self.clip(body, budget))

    def search(self, query):
        if not isinstance(query, list) or not query or any(not isinstance(q, str) or not q.strip() for q in query):
            raise ValueError("search.query must be a nonempty array of nonempty strings")
        if len(query) > 16:
            raise ValueError("at most 16 queries per search call")
        batches = []
        for value in query:
            docs = self.index.search(value, self.topk)
            docids = [doc["docid"] for doc in docs]
            self.retrieved.update(docids)
            self.trace.append(dict(query=value, ranked_docids=docids))
            batches.append(dict(query=value, results=[self.render(doc, self.snippet_tokens) for doc in docs]))
        return json.dumps(batches, ensure_ascii=False)

    def visit(self, url, goal):
        if not self.allow_visit:
            raise ValueError("visit disabled in search-only protocol")
        urls = [url] if isinstance(url, str) else url
        if not isinstance(urls, list) or not urls or len(urls) > 16 or not isinstance(goal, str):
            raise ValueError("visit requires 1..16 local URLs and a goal string")
        results = []
        for value in urls:
            if not isinstance(value, str) or not value.startswith("local://"):
                raise ValueError("visit only reads local://docid; external URLs are forbidden")
            docid = unquote(value[len("local://"):])
            doc = self.index.get(docid)
            self.visited.add(docid)
            results.append(self.render(doc, self.read_tokens))
        # goal is accepted for compatibility; no LLM summarization or answer-based filtering.
        return json.dumps(results, ensure_ascii=False)

    def execute(self, calls):
        from search_comp.evaluation.tool_agent import Registry, ToolError
        if sum(self.counts.values()) + len(calls) > self.max_calls:
            raise ToolError("local tool-call budget exhausted")
        handlers = {"search": self.search, "visit": self.visit}
        for call in calls:
            self.counts[call["name"]] += 1
        return Registry(handlers).execute(calls)


class OriginalLocalTools(LocalTools):
    """Execute export tool names without rewriting the preserved system or actions."""

    def __init__(self, *args, search_lock=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_lock = search_lock if search_lock is not None else nullcontext()

    def search_original(self, query=None, user_query=None):
        value = query if query is not None else user_query
        queries = [value] if isinstance(value, str) else value
        # The original tools expose one ranked result list for a single query.
        batches = json.loads(super().search(queries))
        return batches[0]['results'] if isinstance(value, str) else batches

    def get_document(self, docid):
        if not self.allow_visit:
            raise ValueError('document reading disabled')
        docid = str(docid)
        document = self.index.get(docid)
        self.visited.add(docid)
        return self.render(document, self.read_tokens)

    def crawl_webpage(self, url):
        if not self.allow_visit:
            raise ValueError('document reading disabled')
        documents = getattr(self.index, 'documents', self.index)
        document = documents.get_url(url)
        if document is None:
            raise ValueError('URL is not in the fixed local corpus')
        self.visited.add(document['docid'])
        return self.render(document, self.read_tokens)

    def execute(self, calls):
        from search_comp.evaluation.tool_agent import Registry, ToolError
        if sum(self.counts.values()) + len(calls) > self.max_calls:
            raise ToolError('local tool-call budget exhausted')
        handlers = {'search': self.search_original, 'web_search': self.search_original,
                    'get_document': self.get_document, 'crawl_webpage': self.crawl_webpage,
                    'visit': lambda url, goal: self.crawl_webpage(url)}
        for call in calls:
            self.counts[call['name']] += 1
        with self.search_lock:
            return Registry(handlers).execute(calls)


class OfficialLocalTools(OriginalLocalTools):
    """Official task semantics: five prefix snippets, optional full document by ID."""

    def __init__(self, *args, snippet_tokenizer, **kwargs):
        super().__init__(*args, **kwargs)
        self.snippet_tokenizer = snippet_tokenizer

    def search(self, query):
        if not isinstance(query, str) or not query.strip():
            raise ValueError('official search.query must be a nonempty string')
        search = getattr(self.index, 'search_with_scores', self.index.search)
        docs = search(query, 5)
        ids = [doc['docid'] for doc in docs]
        self.retrieved.update(ids)
        self.trace.append(dict(query=query, ranked_docids=ids))
        results = []
        for doc in docs:
            text = doc.get('raw_text', doc['text'])
            tokens = self.snippet_tokenizer.encode(text, add_special_tokens=False)
            snippet = (self.snippet_tokenizer.decode(tokens[:512], skip_special_tokens=True)
                       if len(tokens) > 512 else text)
            result = dict(docid=doc['docid'], snippet=snippet)
            if doc.get('score') is not None:
                result['score'] = float(doc['score'])
            results.append(result)
        return results

    def get_document(self, docid):
        if not self.allow_visit:
            raise ValueError('get_document disabled in official search-only profile')
        document = self.index.get(str(docid))
        self.visited.add(str(docid))
        return dict(docid=str(docid), text=document.get('raw_text', document['text']))

    def execute(self, calls):
        from search_comp.evaluation.tool_agent import Registry, ToolError
        if sum(self.counts.values()) + len(calls) > self.max_calls:
            raise ToolError('local tool-call budget exhausted')
        handlers = {'search': self.search}
        if self.allow_visit:
            handlers['get_document'] = self.get_document
        for call in calls:
            self.counts[call['name']] += 1
        with self.search_lock:
            return Registry(handlers).execute(calls)
