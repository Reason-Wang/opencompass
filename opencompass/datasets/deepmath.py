from collections import defaultdict
import json
import re
import ast
from typing import Any, Iterable, List, Optional, Union

from datasets import Dataset

JsonLike = Union[dict, list]

_CODE_FENCE_RE = re.compile(
    r"```(?:json|javascript|js|ts|python|text)?\s*(.*?)\s*```",
    re.IGNORECASE | re.DOTALL,
)

def _remove_js_comments(s: str) -> str:
    # // line comments
    s = re.sub(r"//.*?(?=\n|$)", "", s)
    # /* block comments */
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    return s

def _remove_trailing_commas(s: str) -> str:
    # Remove trailing commas before } or ]
    return re.sub(r",\s*([}\]])", r"\1", s)

def _normalize_quotes_and_literals(s: str) -> str:
    # Smart quotes → straight
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    # Python literals → JSON
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    return s

def _brace_scan_candidates(text: str) -> List[str]:
    """
    Return substrings that look like standalone JSON objects/arrays
    using balanced-brace scanning (handles nested braces and quotes).
    """
    candidates: List[str] = []
    start = None
    depth = 0
    in_str = False
    str_ch = ""
    escape = False

    for i, ch in enumerate(text):
        if start is None:
            if ch in "{[":
                start = i
                depth = 1
                in_str = False
                escape = False
                str_ch = ""
            continue

        # We are inside a potential JSON region
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == str_ch:
                in_str = False
        else:
            if ch in {'"', "'"}:
                in_str = True
                str_ch = ch
            elif ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start : i + 1])
                    start = None
    return candidates

def extract_json_candidates(text: str) -> List[str]:
    """
    Pull out possible JSON snippets from:
    - ```json ...``` (or other fenced blocks)
    - Inline brace-scanned objects/arrays
    De-duplicates while preserving order.
    """
    candidates: List[str] = []

    # 1) Code fences
    for m in _CODE_FENCE_RE.finditer(text):
        block = m.group(1).strip()
        # If the fenced block contains multiple JSONs, the brace scan will split them
        candidates.extend(_brace_scan_candidates(block) or [block])

    # 2) Fallback: scan the whole text for inline JSON
    if not candidates:
        candidates = _brace_scan_candidates(text)

    # De-duplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for c in candidates:
        key = c.strip()
        if key not in seen:
            uniq.append(key)
            seen.add(key)
    return uniq

def _try_json_load(s: str) -> Optional[JsonLike]:
    try:
        obj = json.loads(s)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        pass
    return None

def _sanitize_to_valid_json(s: str) -> str:
    s = _normalize_quotes_and_literals(s)
    s = _remove_js_comments(s)
    s = _remove_trailing_commas(s)
    return s

def _try_python_literal(s: str) -> Optional[JsonLike]:
    """
    Last-resort: parse Python-style dict/list (single quotes, etc.)
    using ast.literal_eval, then convert to JSON-compatible types.
    """
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (dict, list)):
            # Round-trip through json to ensure JSON-safe types
            return json.loads(json.dumps(obj))
    except Exception:
        pass
    return None

def load_first_json(text: str, required_keys: Optional[Iterable[str]] = None) -> JsonLike:
    """
    Extract and parse the *first* JSON block that can be loaded.
    Optionally require a set of keys to be present (for dicts).
    Raises ValueError if nothing valid is found.
    """
    required = set(required_keys or [])
    for raw in extract_json_candidates(text):
        # 1) direct load
        obj = _try_json_load(raw)
        # 2) sanitized load
        if obj is None:
            obj = _try_json_load(_sanitize_to_valid_json(raw))
        # 3) python literal fallback
        if obj is None:
            obj = _try_python_literal(raw)

        if obj is None:
            continue

        if required:
            if isinstance(obj, dict) and required.issubset(obj.keys()):
                return obj
            else:
                continue
        return obj

    return None

def load_all_json(text: str) -> List[JsonLike]:
    """
    Extract and parse *all* JSON blocks that can be loaded.
    Returns an empty list if none found.
    """
    results: List[JsonLike] = []
    for raw in extract_json_candidates(text):
        obj = (_try_json_load(raw)
               or _try_json_load(_sanitize_to_valid_json(raw))
               or _try_python_literal(raw))
        if obj is not None:
            results.append(obj)
    return results


def get_deepmath_results(output, dataset):
    """
    Calculate the accuracy for deepmath dataset
    If for each pair of questions, the model thinks both are solvable or both are impossible to solve, it is considered incorrect. Otherwise we assume it is correct.
    """
    test_set = dataset.reader.dataset['test']
    print(f"test_set: {test_set}")
    # convert output to list
    output_list = []
    for i in range(len(output)):
        output_list.append(output[str(i)])

    question_answer_category = {}
    for o, d in zip(output_list, test_set):
        print(f"d: {d}")
        if not d['question_id'] in question_answer_category:
            question_answer_category[d['question_id']] = {}
        judged_category = load_first_json(o['prediction'], required_keys=['category'])
        if not d['sub_id'] in question_answer_category[d['question_id']]:
            question_answer_category[d['question_id']][d['sub_id']] = []
        question_answer_category[d['question_id']][d['sub_id']].append(judged_category)

    results = {"accuracy": None, "accuracy_given_attempted": None, "categories": []}
    all_count = 0
    correct_count = 0
    attempted_count = 0

    print(f"question_answer_category: {question_answer_category}")
    
    for k, v in question_answer_category.items():
        all_count += 1
        if 1 in v:
            first_categories = v[1]
        else:
            first_categories = []
        if 2 in v:
            second_categories = v[2]
        else:
            second_categories = []
        
        extracted_categories = []
        for extracted_category in first_categories:
            if extracted_category is None:
                extracted_categories.append(None)
            else:
                extracted_categories.append(extracted_category['category'])
                
        first_category_compitable = all((category is None or category == "SOLUTION") for category in extracted_categories) or all((category is not None and category == "IMPOSSIBLE") for category in extracted_categories) or len(extracted_categories) == 0
        possible_first_category = next((x for x in extracted_categories if x is not None), None)
        
        extracted_categories = []
        for extracted_category in second_categories:
            if extracted_category is None:
                extracted_categories.append(None)
            else:
                extracted_categories.append(extracted_category['category'])
        second_category_compitable = all((category is None or category == "SOLUTION") for category in extracted_categories) or all((category is not None and category == "IMPOSSIBLE") for category in extracted_categories) or len(extracted_categories) == 0
        possible_second_category = next((x for x in extracted_categories if x is not None), None)
        if first_category_compitable and second_category_compitable and possible_first_category != possible_second_category:
            correct_count += 1
            v["correct"] = True
        else:
            v["correct"] = False
        
        if (all(first_category is not None for first_category in first_categories) or len(first_categories) == 0) and (all(second_category is not None for second_category in second_categories) or len(second_categories) == 0):
            attempted_count += 1

        results['categories'].append(v)

    results['accuracy'] = correct_count / all_count
    results['accuracy_given_attempted'] = correct_count / attempted_count

    return results

def deepmath_postprocess(
    output: dict,
    output_path: str,
    dataset: Dataset,
) -> dict:
    results = get_deepmath_results(output, dataset)
    results['details'] = output
    return results