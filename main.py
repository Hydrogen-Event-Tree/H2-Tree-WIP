import json
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from pandas import Series
from tqdm import tqdm
import ollama

from openai import OpenAI
from google import genai
from google.genai import types

from dashboard import run as run_dashboard

FILE_PATH = "HIAD.xlsx"
COUNT = 20
DEFAULT_MAX_WORKERS = 8

MERGE_SHEETS = [
    "EVENTS",
    "FACILITY",
    "CONSEQUENCES",
    "LESSONS LEARNT",
    "EVENT NATURE",
    "REFERENCES",
]

PROMPT_IGNORED_COLUMNS = {
    "Event Title",
    "Event full description",
    "rocket",
}

PROMPT_SECTIONS = [
    (
        "Event Overview",
        [
            ("Event ID", "Event ID"),
            ("Q", "Q"),
            ("Event Initiating system", "Event Initiating system"),
            ("Classification of the physical effects", "Classification of the physical effects"),
            ("Nature of the consequences", "Nature of the consequences"),
            ("Macro-region", "Macro-region"),
            ("Country", "Country"),
            ("Date", "Date"),
            ("Date entry in HIAD", "Date entry in HIAD"),
        ],
    ),
    (
        "Cause Analysis",
        [
            ("Summary root causes", "Summary root causes"),
            ("Root CAUSE analysis", "Root CAUSE analysis"),
            ("System design error", "System design error"),
            ("Material/manufacturing error", "Material/ manufacturing error"),
            ("Installation error", "Installation error"),
            ("Job factors", "Job factors "),
            ("Human factors", "Human factors"),
            ("Management factors", "Management factors"),
            ("Environment", "Environment"),
            ("Unknown", "Unknown"),
        ],
    ),
    (
        "Facility And Process",
        [
            ("Application", "Application"),
            ("Sub-application", "Sub-application"),
            ("Hydrogen supply chain stage", "Hydrogen supply chain stage"),
            ("Other components involved", "Other components involved"),
            ("Storage/process medium", "Storage/process medium"),
            ("Storage/process quantity [kg]", "Storage/process quantity [kg]"),
            ("Actual pressure [MPa]", "Actual pressure\n[MPa]"),
            ("Design pressure [MPa]", "Design pressure\n[MPa]"),
            ("Location type", "Location type"),
            ("Location description", "Location description"),
            ("Operational condition", "Operational condition"),
            ("Pre-event occurrences", "Pre-event occurrences"),
            ("Description of the facility/unit/process/substances", "Description of the facility/unit/process/substances"),
        ],
    ),
    (
        "Consequences",
        [
            ("Number of injured persons", "Number of injured persons"),
            ("Number of fatalities", "Number of fatalities"),
            ("Environmental damage", "Environmental damage"),
            ("Currency", "Currency"),
            ("Property loss (onsite)", "Property loss (onsite)"),
            ("Property loss (offsite)", "Property loss (offsite)"),
            ("Post-event summary", "Post-event summary"),
            ("Official legal action", "Official legal action"),
            ("Investigation comments", "Investigation comments"),
        ],
    ),
    (
        "Lessons Learnt",
        [
            ("Lesson learnt", "Lesson Learnt"),
            ("Corrective measures", "Corrective Measures"),
        ],
    ),
    (
        "Event Nature",
        [
            ("Emergency action", "Emergency action"),
            ("Emergency evaluation", "Emergency evaluation"),
            ("Release type", "Release type"),
            ("Involved substances", "Involved substances"),
            ("Concentration [% vol]", "[% vol]"),
            ("Probable ignition source", "Probable IGNITION SOURCE"),
        ],
    ),
]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = ""
GEMINI_API_KEY = ""

LLMS = [
    #{
    #    "id": "gemini-3-pro",
    #    "name": "Gemini 3 Pro",
    #    "model": "gemini-3-pro-preview",
    #    "provider": "google-ai-studio",
    #},
    #{
    #    "id": "gemini-3-flash",
    #    "name": "Gemini 3 Flash",
    #    "model": "gemini-3-flash-preview",
    #    "provider": "google-ai-studio",
    #},
    {"id": "qwen3.5-4b", "name": "Qwen3.5-4B", "model": "qwen3.5:4b", "provider": "ollama"},
    {"id": "gemma3-1b", "name": "Gemma3-1B", "model": "gemma3.5:1b", "provider": "ollama"},
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


def _suppress_openpyxl_warnings():
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="openpyxl",
        message="Conditional Formatting extension is not supported",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="openpyxl",
        message="Data Validation extension is not supported",
    )


def _clean_value(value):
    """Normalize cell values and drop empty content."""
    if pd.isna(value):
        return None
    text = str(value).replace("_x000D_", "\n").strip()
    return text or None


def _serialize_reasoning(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=True, indent=2)
    except TypeError:
        return str(value)


def _extract_ollama_message_value(message, field: str):
    if isinstance(message, dict):
        return message.get(field) or message.get("metadata", {}).get(field)
    return getattr(message, field, None) or getattr(getattr(message, "metadata", None), field, None)


def _ask_ollama(prompt: str, system_prompt: str, model: str, stream_reasoning: bool = False):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    if stream_reasoning:
        content_parts = []
        reasoning_parts = []
        printed_reasoning = False

        response = ollama.chat(
            model=model,
            think=False,
            stream=True,
            format=RESPONSE_SCHEMA,
            messages=messages,
        )

        for chunk in response:
            message = chunk.get("message", {}) if isinstance(chunk, dict) else getattr(chunk, "message", {})
            content_delta = _extract_ollama_message_value(message, "content") or ""
            reasoning_delta = (
                _extract_ollama_message_value(message, "thinking")
                or _extract_ollama_message_value(message, "reasoning")
                or ""
            )

            if content_delta:
                content_parts.append(content_delta)

            if reasoning_delta:
                reasoning_parts.append(reasoning_delta)
                if not printed_reasoning:
                    tqdm.write("Thinking:")
                    printed_reasoning = True
                sys.stdout.write(reasoning_delta)
                sys.stdout.flush()

        if printed_reasoning:
            sys.stdout.write("\n")
            sys.stdout.flush()

        content = "".join(content_parts)
        reasoning = _serialize_reasoning("".join(reasoning_parts))
    else:
        response = ollama.chat(
            model=model,
            think=True,
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


def ask(prompt: str, system_prompt: str, model_config: dict, stream_reasoning: bool = False):
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
            stream_reasoning=stream_reasoning,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def read_enriched_events(path: str, n: int):
    _suppress_openpyxl_warnings()
    workbook = pd.ExcelFile(path)
    available_sheets = set(workbook.sheet_names)

    merged = None
    merged_cols: set[str] = set()

    for sheet in MERGE_SHEETS:
        if sheet not in available_sheets:
            continue
        df = pd.read_excel(workbook, sheet_name=sheet)
        if "Event ID" not in df.columns:
            continue  # Skip sheets that cannot be merged.
        if merged is None:
            merged = df
        else:
            # Drop duplicate columns (keep the first occurrence seen).
            dup_cols = [col for col in df.columns if col in merged_cols and col != "Event ID"]
            df = df.drop(columns=dup_cols)
            merged = merged.merge(df, on="Event ID", how="left")
        merged_cols = set(merged.columns)

    if merged is None:
        raise ValueError("No sheets loaded from Excel file.")

    merged = merged[merged["Event full description"].notna()].head(n)
    return merged


def build_event_markdown(row: Series) -> str:
    title = _clean_value(row.get("Event Title")) or "Untitled Event"
    description = _clean_value(row.get("Event full description")) or "No description available."
    used_columns = set()
    sections = [f"# {title}", ""]

    for section_title, field_specs in PROMPT_SECTIONS:
        lines = []
        for label, column in field_specs:
            value = _clean_value(row.get(column))
            if not value:
                continue
            lines.append(f"- **{label}:** {value}")
            used_columns.add(column)

        if lines:
            sections.append(f"## {section_title}")
            sections.extend(lines)
            sections.append("")

    additional_fields = []
    for col in row.index:
        if col in PROMPT_IGNORED_COLUMNS or col in used_columns:
            continue
        cleaned = _clean_value(row.get(col))
        if cleaned:
            additional_fields.append(f"- **{col}:** {cleaned}")

    if additional_fields:
        sections.append("## Additional HIAD Fields")
        sections.extend(additional_fields)
        sections.append("")

    sections.append("## Description")
    sections.append(description)
    return "\n".join(sections)


def _prepare_event_record(row: Series, system_prompt: str, model_config: dict):
    markdown = build_event_markdown(row)
    description = _clean_value(row.get("Event full description")) or "No description available."
    prompt = f"""Use the HIAD event record below to determine the answers for each question.
Treat the structured descriptors as the primary evidence and use the narrative description and references to resolve ambiguity.

Event details:

{markdown}
"""
    record = {
        "event_id": _clean_value(row.get("Event ID")),
        "title": _clean_value(row.get("Event Title")) or "Untitled Event",
        "description": description,
        "user_prompt": prompt,
        "system_prompt": system_prompt,
        "model": model_config["model"],
        "model_id": model_config.get("id"),
        "model_name": model_config.get("name"),
        "reasoning": "",
    }
    return record, markdown


def _run_model_request(record: dict, model_config: dict, stream_reasoning: bool = False):
    result, reasoning = ask(
        prompt=record["user_prompt"],
        system_prompt=record["system_prompt"],
        model_config=model_config,
        stream_reasoning=stream_reasoning,
    )
    record["reasoning"] = reasoning
    record.update(result)
    return record, result


def process_events(model_config, log=False, save_path=None):
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
                record, markdown = _prepare_event_record(row, SYSTEM_PROMPT, model_config)
                if log:
                    tqdm.write(markdown + "\n")
                    tqdm.write("==========================================")
                futures[executor.submit(_run_model_request, record, model_config)] = idx

            with tqdm(
                total=len(events),
                desc=f"Processing events ({model_label})",
            ) as progress:
                for future in as_completed(futures):
                    idx = futures[future]
                    record, result = future.result()
                    results_by_index[idx] = record
                    if log:
                        tqdm.write(str(result))
                        tqdm.write("==========================================")
                    progress.update(1)
        results = results_by_index
    else:
        for _, row in tqdm(
            events.iterrows(),
            total=len(events),
            desc=f"Processing events ({model_label})",
        ):
            record, markdown = _prepare_event_record(row, SYSTEM_PROMPT, model_config)
            if log:
                tqdm.write(markdown + "\n")
                tqdm.write("==========================================")
            record, result = _run_model_request(
                record,
                model_config,
                stream_reasoning=log and provider == "ollama",
            )
            results.append(record)

            if log:
                tqdm.write(str(result))
                tqdm.write("==========================================")

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


def save_events_manifest(models, default_model_id=None, path="events-manifest.json"):
    payload = {"models": models}
    if default_model_id is not None:
        payload["default_model_id"] = default_model_id
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    gen = 1
    port = 4000

    manifest = [
        {
            "id": model_config.get("id"),
            "name": model_config.get("name"),
            "events_path": f"events-{model_config['id']}.json",
            "model": model_config.get("model"),
        }
        for model_config in LLMS
    ]

    if gen:
        for model_config in LLMS:
            events_path = f"events-{model_config['id']}.json"
            process_events(model_config=model_config, log=True, save_path=events_path)

    save_events_manifest(manifest, default_model_id=LLMS[0]["id"] if LLMS else None)

    run_dashboard(port=port)
