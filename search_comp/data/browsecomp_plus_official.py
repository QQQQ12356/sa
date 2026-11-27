"""Shared training/evaluation task contract adapted from BrowseComp-Plus Qwen.

Task wording follows search_agent/prompts.py. The tool wire format is the
project's explicit JSON/ChatML adapter, not a replacement for Qwen-Agent itself.
"""
import json


VERSION = 'browsecomp-plus-official-task-v1'
SOURCE_URL = 'https://github.com/texttron/BrowseComp-Plus/blob/046949032b0328319cc9a02663a759ec601d9402/search_agent/prompts.py'


def task_prompt(question, allow_read=False):
    tools = 'search and get_document tools' if allow_read else 'search tool'
    return (
        'You are a deep research agent. You need to answer the given question by interacting with a search engine, '
        f'using the {tools} provided. Please perform reasoning and use the '
        f'{"tools" if allow_read else "tool"} step by step, in an interleaved manner. '
        f'You may use the {tools} multiple times.\n\n'
        f'Question: {question}\n\n'
        'Your response should be in the following format:\n'
        'Explanation: {your explanation for your final answer. For this explanation section only, you should cite '
        'your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. '
        'For example, [20].}\n'
        'Exact Answer: {your succinct, final answer}\n'
        'Confidence: {your confidence score between 0% and 100% for your answer}'
    )


def system_prompt(allow_read=False):
    schemas = [dict(name='search', description='Search the document corpus; return top 5 results with docid, score and snippet.',
                    parameters=dict(type='object', properties={'query': {'type': 'string'}}, required=['query']))]
    if allow_read:
        schemas.append(dict(name='get_document', description='Retrieve the full text of a document by docid.',
                            parameters=dict(type='object', properties={'docid': {'type': 'string'}}, required=['docid'])))
    return ('You are a helpful assistant.\n\n<tools>\n'
            + '\n'.join(json.dumps(schema, ensure_ascii=False) for schema in schemas)
            + '\n</tools>\n'
            'Reason inside <think>...</think>. Call one tool per turn using '
            '<tool_call>{"name":"tool_name","arguments":{}}</tool_call>. '
            'Tool results arrive inside <tool_response>...</tool_response>. '
            'Return the requested final fields without an <answer> wrapper.')
