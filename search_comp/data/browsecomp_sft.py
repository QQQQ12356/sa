"""Normalize the two BrowseComp-Plus SFT exports to the Research training protocol."""
import json
import re
import copy

from .agent_prompt import DEFAULT_SYSTEM_FILE, build_system_prompt
from .openseeker import CALL_RE, normalize, parse_calls
from .research import normalize_research


SOURCE = "browsecomp-sft"
REPO = "AmanPriyanshu/tool-reasoning-sft-RESEARCH-grill-lab-browsecomp-plus-runs-data-cleaned-rectified"
SHARDS = ("data.parquet", "syn_planner.parquet")
SYSTEM = DEFAULT_SYSTEM_FILE.read_text(encoding="utf-8")
SYSTEM_PROMPT_MODES = {"normalized", "original", "official", "official-read", "by-shard"}
ORIGINAL_CALL_PROTOCOL = "research-original"


def normalize_original(messages, row, source, index, question):
    """Retain source system/actions; regroup export roles into trainable turns."""
    converted = []
    final = None
    for message in messages:
        role, content = message['role'], message['content']
        if not isinstance(content, str) or final is not None:
            raise ValueError('invalid original message or messages after final answer')
        if role in {'reasoning', 'tool_call', 'answer'}:
            if role == 'tool_call' and not parse_calls(content, ORIGINAL_CALL_PROTOCOL):
                raise ValueError('missing original tool call')
            if role == 'answer':
                match = re.fullmatch(r'\s*<answer>(.*?)</answer>\s*', content, re.S)
                final = (match[1] if match else content).strip()
                if not final:
                    raise ValueError('empty original final answer')
            if converted and converted[-1]['role'] == 'assistant':
                if parse_calls(converted[-1]['content'], ORIGINAL_CALL_PROTOCOL):
                    raise ValueError('original tool call without observation')
                converted[-1]['content'] += '\n' + content
            else:
                converted.append(dict(role='assistant', content=content))
        elif role == 'tool_output':
            converted.append(dict(role='tool', content=content))
        elif role in {'system', 'user'}:
            converted.append(dict(role=role, content=content))
        else:
            raise ValueError(f'unsupported original export role: {role}')
    if messages[-1]['role'] != 'answer' or final is None:
        raise ValueError('original export must end with answer')
    result = normalize(dict(messages=converted, question=question, answer=row.get('answer') or final,
                            call_protocol=ORIGINAL_CALL_PROTOCOL), source, index)
    result.update(training_only=True, correctness=row.get('trajectory_correctness'),
                  answer_source='reference' if row.get('answer') else 'demonstration_final_response')
    return result


def normalize_browsecomp_sft(row, source, index, *, system_prompt_mode="normalized",
                           main_prompt_mode='official', planner_prompt_mode='normalized'):
    if system_prompt_mode not in SYSTEM_PROMPT_MODES:
        raise ValueError('invalid SYSTEM_PROMPT_MODE')
    messages = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]
    if [m["role"] for m in messages[:2]] != ["system", "user"]:
        raise ValueError("expected system and user messages")
    original = messages[1]["content"]
    question = row.get("canonical_question", original)
    planner = "data_source" in row
    if system_prompt_mode == 'by-shard':
        system_prompt_mode = planner_prompt_mode if planner else main_prompt_mode
    if system_prompt_mode not in SYSTEM_PROMPT_MODES - {'by-shard'}:
        raise ValueError('invalid shard prompt mode')
    if system_prompt_mode.startswith('official') and planner:
        raise ValueError('official task alignment requires the main shard, not syn_planner')

    def convert(match):
        call = json.loads(match[1])
        name = call["name"]
        args = call.get("arguments", call.get("parameters"))
        if not isinstance(args, dict):
            raise ValueError("tool arguments must be an object")
        if name in {"search", "web_search"}:
            query = args.get("query", args.get("user_query"))
            name, args = "search", {"query": [query] if isinstance(query, str) else query}
        elif planner and name == "crawl_webpage":
            name, args = "visit", {"url": args["url"], "goal": question}
        else:
            raise ValueError(f"unsupported BrowseComp-Plus SFT tool: {name}")
        return "<tool_call>" + json.dumps(dict(name=name, arguments=args), ensure_ascii=False) + "</tool_call>"

    if system_prompt_mode.startswith('official'):
        from .browsecomp_plus_official import VERSION, task_prompt, system_prompt
        allow_read = system_prompt_mode == 'official-read'
        converted = copy.deepcopy(messages)
        converted[0]['content'] = system_prompt(allow_read)
        converted[1]['content'] = task_prompt(question, allow_read)
        for message in converted[2:]:
            if message['role'] == 'tool_call':
                calls = parse_calls(message['content'], ORIGINAL_CALL_PROTOCOL)
                if len(calls) != 1:
                    raise ValueError('official task adapter requires one tool call per turn')
                for call in calls:
                    if call['name'] == 'search':
                        query = call['arguments'].get('query', call['arguments'].get('user_query'))
                        if not isinstance(query, str) or not query.strip():
                            raise ValueError('official search requires a query string')
                        call['arguments'] = {'query': query}
                    elif call['name'] == 'get_document' and allow_read:
                        docid = call['arguments']['docid']
                        if not isinstance(docid, (str, int)) or isinstance(docid, bool) or not str(docid).strip():
                            raise ValueError('official get_document requires a nonempty docid')
                        call['arguments'] = {'docid': str(docid)}
                    else:
                        raise ValueError('tool not enabled in official task profile: ' + call['name'])
                message['content'] = '\n'.join('<tool_call>' + json.dumps(call, ensure_ascii=False) + '</tool_call>'
                                               for call in calls)
            elif message['role'] == 'answer':
                content = message['content'].strip()
                match = re.fullmatch(r'<answer>(.*?)</answer>', content, re.S)
                content = (match[1] if match else content).strip()
                if not all(re.search(r'^\s*(?:\*\*)?' + field + r'(?:\*\*)?\s*:', content, re.M | re.I)
                           for field in ('Explanation', 'Exact Answer', 'Confidence')):
                    raise ValueError('official answer requires Explanation, Exact Answer and Confidence; no fields are fabricated')
                message['content'] = content
        result = normalize_original(converted, row, source, index, question)
        result.update(experiment_protocol=VERSION, official_allow_read=allow_read)
    elif system_prompt_mode == 'original':
        result = normalize_original(messages, row, source, index, question)
    else:
        system = SYSTEM if planner else build_system_prompt(system=SYSTEM, allow_visit=False)
        converted = [dict(role="system", content=system), dict(role="user", content=question)]
        for message in messages[2:]:
            role, content = message["role"], message["content"]
            if role == "tool_call":
                content = CALL_RE.sub(convert, content)
            converted.append(dict(role=role, content=content))
        result = normalize_research(dict(row, messages=converted, question=question), source, index)
    result.update(upstream_source=row.get("data_source", row.get("category")),
                  upstream_qid=row.get("qid"), benchmark_question_id=row.get("benchmark_question_id"),
                  adapter_version="browsecomp-sft-v3", system_prompt_mode=system_prompt_mode,
                  sft_shard='planner' if planner else 'main')
    if original != question:
        result["original_question"] = original
    return result
