"""Send Qwen-native token prompts to method-owned local vLLM replicas."""

import os
from pathlib import Path
from urllib.parse import urlsplit

from search_comp.offline.assets import benchmark_request


def endpoint_urls(value):
    endpoints = []
    for item in value.split(","):
        parsed = urlsplit(item.strip().rstrip("/"))
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"} or not parsed.port
                or parsed.path != "/v1" or parsed.username is not None
                or parsed.query or parsed.fragment):
            raise ValueError("VLLM_URLS requires local http://127.0.0.1:PORT/v1 endpoints")
        url = f"http://127.0.0.1:{parsed.port}/v1"
        if url in endpoints:
            raise ValueError(f"Duplicate vLLM endpoint: {url}")
        endpoints.append(url)
    return endpoints


def served_name(origin):
    return "sa-baseline-qwen35-4b" if origin == "base-qwen35" else "sa-plain-qwen35-4b"


class VllmClient:
    def __init__(self, http, url, model_path, origin, options, max_context):
        self.http = http
        self.url = url
        self.model = served_name(origin)
        self.sampling = dict(
            temperature=options.get("temperature", 1.0) if options.get("decoding") == "sample" else 0,
            top_p=options.get("top_p", 1.0), repetition_penalty=options.get("repetition_penalty", 1.0),
            seed=options.get("seed", 42),
        )
        self.metadata = dict(backend="vllm", checkpoint=model_path, model_origin=origin,
                             tool_protocol="qwen-native", served_model=self.model,
                             **self.sampling)
        cards = self.request("GET", "models").json()["data"]
        card = next((card for card in cards if card["id"] == self.model), None)
        if not card or not card.get("root") or Path(card["root"]).resolve() != Path(model_path).resolve():
            raise ValueError(f"{url} must serve {self.model} from {model_path}")
        if card.get("max_model_len") is not None and card["max_model_len"] < max_context:
            raise ValueError(f"{url} max_model_len is below MAX_CONTEXT={max_context}")

    def request(self, method, route, **kwargs):
        key = os.environ.get("SA_VLLM_API_KEY", "EMPTY")
        with benchmark_request():
            response = self.http.request(method, f"{self.url}/{route}",
                                         headers={"Authorization": f"Bearer {key}"}, **kwargs)
        response.raise_for_status()
        return response

    def start_question(self, question):
        pass

    def __call__(self, inputs, budget, *, stop_texts):
        import httpx

        body = dict(model=self.model, prompt=inputs["input_ids"], max_tokens=budget,
                    stop=list(stop_texts), skip_special_tokens=False, **self.sampling)
        for attempt in range(3):
            try:
                result = self.request("POST", "completions", json=body).json()
                break
            except httpx.HTTPStatusError as error:
                if attempt == 2 or error.response.status_code not in {429, 500, 502, 503, 504}:
                    raise
            except httpx.TransportError:
                if attempt == 2:
                    raise
        choice = result["choices"][0]
        return choice["text"], choice["finish_reason"] != "length", result["usage"]["completion_tokens"]
