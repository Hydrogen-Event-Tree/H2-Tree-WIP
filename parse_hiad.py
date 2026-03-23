import warnings

import pandas as pd

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

MERGE_SHEETS = [
    "EVENTS",
    "FACILITY",
    "CONSEQUENCES",
    "LESSONS LEARNT",
    "EVENT NATURE",
    "REFERENCES",
]


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
            continue
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


def clean_value(value):
    """Normalize cell values and drop empty content."""
    if pd.isna(value):
        return None
    text = str(value).replace("_x000D_", "\n").strip()
    return text or None


def build_event_markdown(row) -> str:
    title = clean_value(row.get("Event Title")) or "Untitled Event"
    description = clean_value(row.get("Event full description")) or "No description available."
    used_columns = set()
    sections = [f"# {title}", ""]

    for section_title, field_specs in PROMPT_SECTIONS:
        lines = []
        for label, column in field_specs:
            value = clean_value(row.get(column))
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
        cleaned = clean_value(row.get(col))
        if cleaned:
            additional_fields.append(f"- **{col}:** {cleaned}")

    if additional_fields:
        sections.append("## Additional HIAD Fields")
        sections.extend(additional_fields)
        sections.append("")

    sections.append("## Description")
    sections.append(description)
    return "\n".join(sections)


def build_event_record(row, system_prompt: str, model_config: dict, response_schema_json: str):
    markdown = build_event_markdown(row)
    description = clean_value(row.get("Event full description")) or "No description available."
    prompt = f"""Use the HIAD event record below to determine the answers for each question.
Treat the structured descriptors as the primary evidence and use the narrative description and references to resolve ambiguity.

Event details:

{markdown}

JSON schema:

{response_schema_json}
"""
    return {
        "event_id": clean_value(row.get("Event ID")),
        "title": clean_value(row.get("Event Title")) or "Untitled Event",
        "description": description,
        "user_prompt": prompt,
        "system_prompt": system_prompt,
        "model": model_config["model"],
        "model_name": model_config.get("name"),
        "reasoning": "",
    }
