import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import ollama

from openai import OpenAI
from google import genai
from google.genai import types

from dashboard import run as run_dashboard
from parse_hiad import build_event_record, read_enriched_events

FILE_PATH = "HIAD.xlsx"
EVENTS_TO_INCLUDE = 2000
ANSWERS_PER_MODEL = 2
DEFAULT_MAX_WORKERS = 8
OPENROUTER_DEFAULT_MAX_WORKERS = 4

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = ""
OPENROUTER_REASONING_EFFORT = "high"
OPENROUTER_TIMEOUT_SECONDS = 40
OPENROUTER_MAX_ATTEMPTS = 4
OPENROUTER_RETRY_BACKOFF_SECONDS = 20
GEMINI_API_KEY = ""

LLMS = [
    {
        "name": "GPT 5.4 Nano",
        "model": "openai/gpt-5.4-nano",
        "provider": "openrouter",
    },
    {
        "name": "Gemini 3.1 Flash Lite",
        "model": "google/gemini-3.1-flash-lite-preview",
        "provider": "openrouter",
    },
    {
        "name": "Grok 4.1 Fast",
        "model": "x-ai/grok-4.1-fast",
        "provider": "openrouter",
    },
    {
        "name": "Mistral Small 4",
        "model": "mistralai/mistral-small-2603",
        "provider": "openrouter",
    },
    {
        "name": "Qwen3.5 397B",
        "model": "qwen/qwen3.5-397b-a17b",
        "provider": "openrouter",
    },
    #{"name": "Qwen3.5-4B", "model": "qwen3.5:4b", "provider": "ollama"},
    #{"name": "Gemma3 1B", "model": "gemma3:1b", "provider": "ollama"},
    #{"name": "Ministral 3 Reasoning 14B", "model": "seamon67/Ministral-3-Reasoning:14b", "provider": "ollama"},
    #{"name": "Qwen3.5 35B", "model": "qwen3.5:35b", "provider": "ollama"},
    #{"name": "gpt-oss-20b", "model": "gpt-oss:20b", "provider": "ollama"},
    #{"name": "GLM 4.7 Flash", "model": "glm-4.7-flash", "provider": "ollama"},
]

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "continuous_release": {"type": "integer", "minimum": 0, "maximum": 10},
        "immediate_ignition": {"type": "integer", "minimum": 0, "maximum": 10},
        "barrier_stopped_immediate_ignition": {"type": "integer", "minimum": 0, "maximum": 10},
        "delayed_ignition": {"type": "integer", "minimum": 0, "maximum": 10},
        "barrier_stopped_delayed_ignition": {"type": "integer", "minimum": 0, "maximum": 10},
        "confined_space": {"type": "integer", "minimum": 0, "maximum": 10},
        "pure_h2": {"type": "integer", "minimum": 0, "maximum": 10},
        "gaseous_h2": {"type": "integer", "minimum": 0, "maximum": 10},
        "loss_of_containment": {"type": "integer", "minimum": 0, "maximum": 10},
    },
    "required": [
        "continuous_release",
        "immediate_ignition",
        "barrier_stopped_immediate_ignition",
        "delayed_ignition",
        "barrier_stopped_delayed_ignition",
        "confined_space",
        "pure_h2",
        "gaseous_h2",
        "loss_of_containment",
    ],
}
ANSWER_FIELDS = tuple(RESPONSE_SCHEMA["properties"].keys())

RESPONSE_SCHEMA_JSON = json.dumps(RESPONSE_SCHEMA, ensure_ascii=True, indent=2)

OPENROUTER_SCHEMA = {
    "name": "event_tree_output",
    "schema": {
        **RESPONSE_SCHEMA,
        "additionalProperties": False,
    },
    "strict": True,
}

SYSTEM_PROMPT = """Fill every field in the JSON schema with a single integer from 0-10. Use this scale for every field:
- 10 = yes with full certainty
- 9-6 = yes, with decreasing certainty
- 5 = no information either way / genuinely ambiguous from the provided evidence
- 4-1 = no, with increasing certainty
- 0 = no with full certainty

Return only raw JSON. Do not wrap it in markdown or code fences.

Schema: {continuous_release:int, immediate_ignition:int, barrier_stopped_immediate_ignition:int, delayed_ignition:int, barrier_stopped_delayed_ignition:int, confined_space:int, pure_h2:int, gaseous_h2:int, loss_of_containment:int}. Use the provided event details to decide.

Continuous release rubric: Score high if hydrogen release persisted over time rather than a single brief discharge.
Immediate ignition rubric: Score high if ignition occurred at the moment of release or within seconds without delay or hydrogen accumulation in the surrounding environment.
Delayed ignition rubric: Score high if a flammable cloud or explosive mixture formed and ignited after a noticeable delay from the release.
Barrier (immediate) rubric: Score high only when immediate ignition did not occur and a barrier meaningfully prevented immediate ignition (e.g., ESD systems, isolation valves, emergency shutdowns). If immediate ignition clearly occurred, barrier_stopped_immediate_ignition should be near 0.
Barrier (delayed) rubric: Score high only when delayed ignition did not occur and a barrier meaningfully prevented delayed ignition (e.g., ESD, inerting, venting, isolation). If delayed ignition clearly occurred, barrier_stopped_delayed_ignition should be near 0.
Confined space rubric: Score high if the release occurred in an enclosed or poorly ventilated area that limits dispersion.
Pure H2 rubric: Score high if the released substance is pure or essentially pure gaseous hydrogen; score low if it is a hydrogen mixture with significant non-hydrogen components or is not primarily hydrogen.
Gaseous H2 rubric: Score high if the released hydrogen was gaseous; score low if the release was liquid/solid hydrogen or otherwise not gaseous hydrogen.
Loss of containment rubric: Score high if any amount of hydrogen actually leaked or was released; score low if no hydrogen release occurred.
"""


def _build_genai_response_schema():
    properties = {}
    for name, spec in RESPONSE_SCHEMA["properties"].items():
        schema_type = spec.get("type")
        if schema_type == "boolean":
            properties[name] = types.Schema(type=types.Type.BOOLEAN)
        elif schema_type == "integer":
            properties[name] = types.Schema(
                type=types.Type.INTEGER,
                minimum=spec.get("minimum"),
                maximum=spec.get("maximum"),
            )
        else:
            raise ValueError(f"Unsupported schema type: {schema_type}")
    return types.Schema(
        type=types.Type.OBJECT,
        properties=properties,
        required=RESPONSE_SCHEMA.get("required", []),
    )


def _parse_genai_response(response):
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if hasattr(parsed, "model_dump"):
            return (
                _validate_response_payload(parsed.model_dump(), json.dumps(parsed.model_dump(), ensure_ascii=True)),
                "",
                False,
            )
        if isinstance(parsed, dict):
            return _validate_response_payload(parsed, json.dumps(parsed, ensure_ascii=True)), "", False
        raise ValueError(f"LLM returned unsupported parsed content: {parsed}")

    candidates = getattr(response, "candidates", None) or []
    parts = []
    if candidates:
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []

    text_parts = []
    reasoning_parts = []
    for part in parts:
        text = getattr(part, "text", None)
        if not text:
            continue
        if getattr(part, "thought", False):
            reasoning_parts.append(text)
        else:
            text_parts.append(text)

    text = "".join(text_parts).strip()
    reasoning = "\n".join(reasoning_parts).strip()
    if not text:
        raise ValueError("LLM returned no text content.")
    parsed = _parse_json_response(text)
    return parsed, reasoning, False


def _serialize_reasoning(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=True, indent=2)
    except TypeError:
        return str(value)


def _normalize_reasoning_trace(value):
    shortened = False

    def normalize_item(item):
        nonlocal shortened
        if item is None:
            return ""
        if isinstance(item, str):
            text = item.strip()
            return text
        if isinstance(item, list):
            parts = [normalize_item(part) for part in item]
            return "\n\n".join(part for part in parts if part)
        if isinstance(item, dict):
            item_type = item.get("type")
            if item_type == "reasoning.encrypted":
                shortened = True
                return ""
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
            if item_type == "reasoning.summary":
                summary = item.get("summary")
                if isinstance(summary, str) and summary.strip():
                    return summary.strip()
                if isinstance(summary, list):
                    parts = [normalize_item(part) for part in summary]
                    return "\n\n".join(part for part in parts if part)
            return ""
        return str(item).strip()

    return normalize_item(value).strip(), shortened


def _validate_response_payload(parsed, raw_text: str):
    if not isinstance(parsed, dict):
        raise ValueError(f"LLM returned non-object JSON content: {raw_text}")

    expected_fields = set(ANSWER_FIELDS)
    actual_fields = set(parsed.keys())
    missing_fields = expected_fields - actual_fields
    extra_fields = actual_fields - expected_fields
    if missing_fields or extra_fields:
        problems = []
        if missing_fields:
            problems.append(f"missing fields: {sorted(missing_fields)}")
        if extra_fields:
            problems.append(f"unexpected fields: {sorted(extra_fields)}")
        raise ValueError(f"LLM returned invalid JSON schema ({'; '.join(problems)}): {raw_text}")

    for field in ANSWER_FIELDS:
        value = parsed[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"LLM returned non-integer value for {field}: {raw_text}")
        if not 0 <= value <= 10:
            raise ValueError(f"LLM returned out-of-range value for {field}: {raw_text}")

    return parsed


def _parse_json_response(content):
    if not isinstance(content, str):
        raise ValueError(f"LLM returned non-text content: {content}")

    text = content.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.fullmatch(r"```(?i:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if not match:
            raise ValueError(f"LLM returned non-JSON content: {content}")
        fenced_text = match.group(1).strip()
        try:
            parsed = json.loads(fenced_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned non-JSON content: {content}") from exc

    return _validate_response_payload(parsed, text)


def _ask_ollama(prompt: str, system_prompt: str, model: str, seed: int | None = None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    request_kwargs = {
        "model": model,
        "think": True,
        "format": RESPONSE_SCHEMA,
        "messages": messages,
    }
    if seed is not None:
        request_kwargs["options"] = {"seed": seed}

    response = ollama.chat(
        **request_kwargs,
    )

    message = response.get("message", {})
    content = message.get("content", "")
    reasoning = (
        message.get("thinking")
        or message.get("reasoning")
        or message.get("metadata", {}).get("thinking")
        or response.get("thinking")
        or response.get("metadata", {}).get("thinking")
        or ""
    )
    reasoning = _serialize_reasoning(reasoning)

    parsed = _parse_json_response(content)
    return parsed, reasoning, False


def _ask_openrouter(prompt: str, system_prompt: str, model_config: dict, seed: int | None = None):
    model = model_config["model"]
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    reasoning_effort = model_config.get("reasoning_effort") or OPENROUTER_REASONING_EFFORT
    request_timeout_seconds = model_config.get("timeout_seconds") or OPENROUTER_TIMEOUT_SECONDS
    max_attempts = max(1, int(model_config.get("max_attempts") or OPENROUTER_MAX_ATTEMPTS))
    retry_backoff_seconds = model_config.get("retry_backoff_seconds") or OPENROUTER_RETRY_BACKOFF_SECONDS

    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        timeout=request_timeout_seconds,
        max_retries=0,
    )
    request_kwargs = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_schema", "json_schema": OPENROUTER_SCHEMA},
        "extra_body": {
            "plugins": [{"id": "response-healing"}],
            "include_reasoning": True,
            "reasoning": {"effort": reasoning_effort},
        },
    }
    if seed is not None:
        request_kwargs["seed"] = seed

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                **request_kwargs,
            )
            message = response.choices[0].message

            if hasattr(message, "model_dump"):
                payload = message.model_dump()
                content = payload.get("content", "")
                reasoning_details = payload.get("reasoning_details")
                reasoning_text = payload.get("reasoning")
            elif isinstance(message, dict):
                content = message.get("content", "")
                reasoning_details = message.get("reasoning_details")
                reasoning_text = message.get("reasoning")
            else:
                content = getattr(message, "content", "")
                reasoning_details = getattr(message, "reasoning_details", None)
                reasoning_text = getattr(message, "reasoning", None)

            reasoning_from_details, reasoning_is_shortened = _normalize_reasoning_trace(reasoning_details)
            reasoning_from_text, shortened_from_text = _normalize_reasoning_trace(reasoning_text)
            reasoning = reasoning_from_details or reasoning_from_text
            reasoning_is_shortened = reasoning_is_shortened or shortened_from_text

            parsed = _parse_json_response(content)
            return parsed, reasoning, reasoning_is_shortened
        except ValueError as exc:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"OpenRouter returned invalid structured output for {model} after {max_attempts} attempts: {exc}"
                ) from exc
            time.sleep(retry_backoff_seconds * attempt)
        except Exception as exc:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"OpenRouter request failed for {model} after {max_attempts} attempts: {exc}"
                ) from exc
            time.sleep(retry_backoff_seconds * attempt)


def _ask_google_ai_studio(prompt: str, system_prompt: str, model: str, seed: int | None = None):
    if not GEMINI_API_KEY:
        raise ValueError("Missing GEMINI_API_KEY for Google AI Studio.")

    client = genai.Client(api_key=GEMINI_API_KEY)
    config = types.GenerateContentConfig(
        systemInstruction=system_prompt or None,
        thinkingConfig=types.ThinkingConfig(thinkingLevel=types.ThinkingLevel.HIGH),
        responseMimeType="application/json",
        responseSchema=_build_genai_response_schema(),
        seed=seed,
    )
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    ]

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    return _parse_genai_response(response)


def ask(prompt: str, system_prompt: str, model_config: dict, seed: int | None = None):
    provider = model_config.get("provider", "ollama")
    model = model_config.get("model")
    if not model:
        raise ValueError(f"Invalid model config: {model_config!r}")
    if provider == "openrouter":
        return _ask_openrouter(prompt=prompt, system_prompt=system_prompt, model_config=model_config, seed=seed)
    if provider == "google-ai-studio":
        return _ask_google_ai_studio(prompt=prompt, system_prompt=system_prompt, model=model, seed=seed)
    if provider == "ollama":
        return _ask_ollama(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            seed=seed,
        )
    raise ValueError(f"Unsupported provider: {provider}")

def _run_model_request(record: dict, model_config: dict, realization_index: int):
    result, reasoning, reasoning_is_shortened = ask(
        prompt=record["user_prompt"],
        system_prompt=record["system_prompt"],
        model_config=model_config,
        seed=realization_index + 1,
    )
    return realization_index, result, reasoning, reasoning_is_shortened


def _initialize_event_result(record: dict):
    result = {
        **record,
        "reasoning": ["" for _ in range(ANSWERS_PER_MODEL)],
        "reasoning_is_shortened": [False for _ in range(ANSWERS_PER_MODEL)],
    }
    for field in ANSWER_FIELDS:
        result[field] = [None for _ in range(ANSWERS_PER_MODEL)]
    return result


def _store_realization(
    event_result: dict,
    realization_index: int,
    result: dict,
    reasoning: str,
    reasoning_is_shortened: bool,
):
    event_result["reasoning"][realization_index] = reasoning
    event_result["reasoning_is_shortened"][realization_index] = reasoning_is_shortened
    for field in ANSWER_FIELDS:
        event_result[field][realization_index] = result.get(field)


def _store_failed_realization(event_result: dict, realization_index: int, error: Exception):
    event_result["reasoning"][realization_index] = f"ERROR: {error}"
    event_result["reasoning_is_shortened"][realization_index] = False


def process_events(model_config, save_path=None):
    events = read_enriched_events(FILE_PATH, EVENTS_TO_INCLUDE)
    results = []
    if not isinstance(model_config, dict) or not model_config.get("model"):
        raise ValueError(f"Invalid model config: {model_config!r}")

    model_label = model_config.get("name") or model_config.get("model") or "model"
    provider = model_config.get("provider", "ollama")
    if provider in ("openrouter", "google-ai-studio"):
        if provider == "openrouter":
            max_workers = model_config.get("max_workers") or OPENROUTER_DEFAULT_MAX_WORKERS
        else:
            max_workers = model_config.get("max_workers") or DEFAULT_MAX_WORKERS
        results_by_index = [None] * len(events)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for idx, (_, row) in enumerate(events.iterrows()):
                record = build_event_record(row, SYSTEM_PROMPT, model_config, RESPONSE_SCHEMA_JSON)
                results_by_index[idx] = _initialize_event_result(record)
                for realization_index in range(ANSWERS_PER_MODEL):
                    futures[
                        executor.submit(
                            _run_model_request,
                            record,
                            model_config,
                            realization_index,
                        )
                    ] = (idx, realization_index)

            with tqdm(
                total=len(events) * ANSWERS_PER_MODEL,
                desc=f"Processing events ({model_label})",
            ) as progress:
                for future in as_completed(futures):
                    idx, realization_index = futures[future]
                    try:
                        _, result, reasoning, reasoning_is_shortened = future.result()
                    except Exception as exc:
                        tqdm.write(
                            f"{model_label}: event {idx + 1}, realization {realization_index + 1} failed: {exc}"
                        )
                        _store_failed_realization(results_by_index[idx], realization_index, exc)
                    else:
                        _store_realization(
                            results_by_index[idx],
                            realization_index,
                            result,
                            reasoning,
                            reasoning_is_shortened,
                        )
                    progress.update(1)
        results = results_by_index
    else:
        event_progress = tqdm(
            total=len(events) * ANSWERS_PER_MODEL,
            desc=f"Processing events ({model_label})",
        )
        for _, row in events.iterrows():
            record = build_event_record(row, SYSTEM_PROMPT, model_config, RESPONSE_SCHEMA_JSON)
            event_result = _initialize_event_result(record)
            for realization_index in range(ANSWERS_PER_MODEL):
                _, result, reasoning, reasoning_is_shortened = _run_model_request(
                    record, model_config, realization_index
                )
                _store_realization(
                    event_result,
                    realization_index,
                    result,
                    reasoning,
                    reasoning_is_shortened,
                )
                event_progress.update(1)
            results.append(event_result)
        event_progress.close()

    if save_path is not None:
        save_events(results, path=save_path)

    return results


def save_events(events, path = "events.json"):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(events, handle, ensure_ascii=True, indent=2)


def load_events(path = "events.json"):
    with open(path, "r", encoding="utf-8") as handle:
        events = json.load(handle)
    return events


def save_events_manifest(model_configs, default_model=None, path="events-manifest.json"):
    models = [
        {
            "name": model_config.get("name"),
            "events_path": f"events-{_model_slug(model_config['model'])}.json",
            "model": model_config.get("model"),
        }
        for model_config in model_configs
    ]
    payload = {"models": models, "answers_per_model": ANSWERS_PER_MODEL}
    if default_model is None and model_configs:
        default_model = model_configs[0].get("model")
    if default_model is not None:
        payload["default_model"] = default_model
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def _model_slug(model: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in model)


if __name__ == "__main__":
    gen = 1
    port = 4000

    if gen:
        for model_config in LLMS:
            events_path = f"events-{_model_slug(model_config['model'])}.json"
            process_events(model_config=model_config, save_path=events_path)

        save_events_manifest(LLMS)

    run_dashboard(port=port)
