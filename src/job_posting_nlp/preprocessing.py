import html
import re

import pandas as pd


_URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_URL_TRAILING_PUNCTUATION = ".,;:!?)]}"
_OBSERVED_INVISIBLE_CHARACTERS = re.compile(r"[\u200D\uFEFF]")
_WHITESPACE_PATTERN = re.compile(r"\s+")

_PREPARED_COLUMNS = [
    "job_id",
    "title",
    "role_category",
    "company_id",
    "company_name",
    "description",
    "cleaned_description",
]


def _replace_url(match: re.Match[str]) -> str:
    url = match.group(0)
    url_without_punctuation = url.rstrip(_URL_TRAILING_PUNCTUATION)
    trailing_punctuation = url[len(url_without_punctuation):]

    # The URL match can include the punctuation ending the sentence, so add it back.
    return f" [URL]{trailing_punctuation} "


def clean_description(description: str) -> str:
    """Apply the conservative shared cleaning used before modelling."""
    # Convert entities back to their normal characters
    cleaned = html.unescape(description)

    # This broken apostrophe was the only encoding issue found in the audit.
    cleaned = cleaned.replace("â\x80\x99", "'")
    cleaned = _URL_PATTERN.sub(_replace_url, cleaned) # replaces URLs with [URL] token

    # Spaces avoid accidentally joining words that were separated by hidden characters.
    cleaned = _OBSERVED_INVISIBLE_CHARACTERS.sub(" ", cleaned)

    return _WHITESPACE_PATTERN.sub(" ", cleaned).strip()


def prepare_modeling_rows(postings: pd.DataFrame) -> pd.DataFrame:
    """Clean descriptions and keep one row per cleaned description and label."""
    # Exclude URL-only rows first, otherwise the URL becomes a normal looking token.
    url_only_mask = postings["description"].map(
        lambda description: _URL_PATTERN.fullmatch(description.strip()) is not None
    )

    prepared = postings.loc[~url_only_mask].copy()
    prepared["cleaned_description"] = prepared["description"].map(
        clean_description
    )

    # Include the label because the same description can have different observed labels.
    prepared = prepared.drop_duplicates(
        subset=["cleaned_description", "role_category"],
        keep="first",
    )

    return prepared[_PREPARED_COLUMNS].reset_index(drop=True)
