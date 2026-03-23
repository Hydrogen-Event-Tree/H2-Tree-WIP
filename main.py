import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import ollama

from openai import OpenAI
from google import genai
from google.genai import types

from dashboard import run as run_dashboard
from parse_hiad import build_event_record, read_enriched_events

FILE_PATH = "HIAD.xlsx"
COUNT = 2
DEFAULT_MAX_WORKERS = 8

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = ""
GEMINI_API_KEY = ""

LLMS = [
    #{
    #    "name": "Gemini 3 Pro",
    #    "model": "gemini-3-pro-preview",
    #    "provider": "google-ai-studio",
    #},
    #{
    #    "name": "Gemini 3 Flash",
    #    "model": "gemini-3-flash-preview",
    #    "provider": "google-ai-studio",
    #},
    #{"name": "Qwen3.5-4B", "model": "qwen3.5:4b", "provider": "ollama"},
    {"name": "gpt-oss-20b", "model": "gpt-oss:20b", "provider": "ollama"},
    {"name": "Gemma3-1B", "model": "gemma3:1b", "provider": "ollama"},
]

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "continuous_release": {"type": "boolean"},
        "continuous_release_confidence": {"type": "integer", "minimum": 0, "maximum": 10},
        "immediate_ignition": {"type": "boolean"},
        "immediate_ignition_confidence": {"type": "integer", "minimum": 0, "maximum": 10},
        "barrier_stopped_immediate_ignition": {"type": "boolean"},
        "delayed_ignition": {"type": "boolean"},
        "delayed_ignition_confidence": {"type": "integer", "minimum": 0, "maximum": 10},
        "barrier_stopped_delayed_ignition": {"type": "boolean"},
        "confined_space": {"type": "boolean"},
        "confined_space_confidence": {"type": "integer", "minimum": 0, "maximum": 10},
        "exclude_not_pure_h2": {"type": "boolean"},
        "exclude_not_gaseous_h2": {"type": "boolean"},
        "exclude_no_loc": {"type": "boolean"},
    },
    "required": [
        "continuous_release",
        "continuous_release_confidence",
        "immediate_ignition",
        "immediate_ignition_confidence",
        "barrier_stopped_immediate_ignition",
        "delayed_ignition",
        "delayed_ignition_confidence",
        "barrier_stopped_delayed_ignition",
        "confined_space",
        "confined_space_confidence",
        "exclude_not_pure_h2",
        "exclude_not_gaseous_h2",
        "exclude_no_loc",
    ],
}

RESPONSE_SCHEMA_JSON = json.dumps(RESPONSE_SCHEMA, ensure_ascii=True, indent=2)

OPENROUTER_SCHEMA = {
    "name": "event_tree_output",
    "schema": {
        **RESPONSE_SCHEMA,
        "additionalProperties": False,
    },
    "strict": True,
}

SYSTEM_PROMPT = """Fill every field in the JSON schema. For each question not about barriers or exclusion, provide a boolean answer and an integer confidence from 0-10 (0 = no information in the description to decide; 10 = the description makes the chosen answer unquestionably clear). Schema: {continuous_release:boolean, continuous_release_confidence:int, immediate_ignition:boolean, immediate_ignition_confidence:int, barrier_stopped_immediate_ignition:boolean, delayed_ignition:boolean, delayed_ignition_confidence:int, barrier_stopped_delayed_ignition:boolean, confined_space:boolean, confined_space_confidence:int, exclude_not_pure_h2:boolean, exclude_not_gaseous_h2:boolean, exclude_no_loc:boolean}. Use the provided event details to decide.

Continuous release rubric: Mark true if hydrogen release persisted over time rather than a single brief discharge.
Immediate ignition rubric: Mark true if ignition occurred at the moment of release or within seconds without delay or hydrogen accumulation in the surrounding environment.
Delayed ignition rubric: Mark true if a flammable cloud or explosive mixture formed and ignited after a noticeable delay from the release.
Barrier (immediate) rubric: If immediate_ignition is true, barrier_stopped_immediate_ignition must be false. If immediate_ignition is false, set barrier_stopped_immediate_ignition to true only when a barrier meaningfully prevented immediate ignition (e.g., ESD systems, isolation valves, emergency shutdowns); otherwise set it to false.
Barrier (delayed) rubric: If delayed_ignition is true, barrier_stopped_delayed_ignition must be false. If delayed_ignition is false, set barrier_stopped_delayed_ignition to true only when a barrier meaningfully prevented delayed ignition (e.g., ESD, inerting, venting, isolation); otherwise set it to false.
Confined space rubric: Mark true if the release occurred in an enclosed or poorly ventilated area that limits dispersion.
Exclude not pure H2 rubric: Mark true if the release substance is a hydrogen mixture with significant non-hydrogen components or is not primarily hydrogen.
Exclude not gaseous H2 rubric: Mark true if hydrogen was released in a non-gaseous state (e.g., liquid or solid hydrogen) or the release medium is not gaseous H2.
Exclude no LOC rubric: Mark true when no hydrogen was released; set false if any amount of hydrogen actually leaked.
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
            return parsed.model_dump(), ""
        if isinstance(parsed, dict):
            return parsed, ""
        return parsed, ""

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
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned non-JSON content: {text}") from exc
    return parsed, reasoning


def _serialize_reasoning(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=True, indent=2)
    except TypeError:
        return str(value)


def _ask_ollama(prompt: str, system_prompt: str, model: str):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = ollama.chat(
        model=model,
        think=False,
        format=RESPONSE_SCHEMA,
        messages=messages,
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

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned non-JSON content: {content}") from exc
    return parsed, reasoning


def _ask_openrouter(prompt: str, system_prompt: str, model: str):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_schema", "json_schema": OPENROUTER_SCHEMA},
        extra_body={
            "plugins": [{"id": "response-healing"}],
            "include_reasoning": True,
            "reasoning": {"enabled": True},
        }
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

    reasoning = reasoning_details if reasoning_details is not None else reasoning_text or ""
    reasoning = _serialize_reasoning(reasoning)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned non-JSON content: {content}") from exc
    return parsed, reasoning


def _ask_google_ai_studio(prompt: str, system_prompt: str, model: str):
    if not GEMINI_API_KEY:
        raise ValueError("Missing GEMINI_API_KEY for Google AI Studio.")

    client = genai.Client(api_key=GEMINI_API_KEY)
    config = types.GenerateContentConfig(
        systemInstruction=system_prompt or None,
        thinkingConfig=types.ThinkingConfig(thinkingLevel=types.ThinkingLevel.HIGH),
        responseMimeType="application/json",
        responseSchema=_build_genai_response_schema(),
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


def ask(prompt: str, system_prompt: str, model_config: dict):
    provider = model_config.get("provider", "ollama")
    model = model_config.get("model")
    if not model:
        raise ValueError(f"Invalid model config: {model_config!r}")
    if provider == "openrouter":
        return _ask_openrouter(prompt=prompt, system_prompt=system_prompt, model=model)
    if provider == "google-ai-studio":
        return _ask_google_ai_studio(prompt=prompt, system_prompt=system_prompt, model=model)
    if provider == "ollama":
        return _ask_ollama(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
        )
    raise ValueError(f"Unsupported provider: {provider}")

def _run_model_request(record: dict, model_config: dict):
    result, reasoning = ask(
        prompt=record["user_prompt"],
        system_prompt=record["system_prompt"],
        model_config=model_config,
    )
    record["reasoning"] = reasoning
    record.update(result)
    return record, result


def process_events(model_config, save_path=None):
    events = read_enriched_events(FILE_PATH, COUNT)
    results = []
    if not isinstance(model_config, dict) or not model_config.get("model"):
        raise ValueError(f"Invalid model config: {model_config!r}")

    model_label = model_config.get("name") or model_config.get("model") or "model"
    provider = model_config.get("provider", "ollama")
    if provider in ("openrouter", "google-ai-studio"):
        max_workers = model_config.get("max_workers") or DEFAULT_MAX_WORKERS
        results_by_index = [None] * len(events)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for idx, (_, row) in enumerate(events.iterrows()):
                record = build_event_record(row, SYSTEM_PROMPT, model_config, RESPONSE_SCHEMA_JSON)
                futures[executor.submit(_run_model_request, record, model_config)] = idx

            with tqdm(
                total=len(events),
                desc=f"Processing events ({model_label})",
            ) as progress:
                for future in as_completed(futures):
                    idx = futures[future]
                    record, _ = future.result()
                    results_by_index[idx] = record
                    progress.update(1)
        results = results_by_index
    else:
        for _, row in tqdm(
            events.iterrows(),
            total=len(events),
            desc=f"Processing events ({model_label})",
        ):
            record = build_event_record(row, SYSTEM_PROMPT, model_config, RESPONSE_SCHEMA_JSON)
            record, _ = _run_model_request(record, model_config)
            results.append(record)

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
    payload = {"models": models}
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
