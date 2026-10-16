"""
firebase_client.py
------------------
Initialises Firebase Admin SDK (once) and exposes simple Firestore
CRUD helpers used across FlowSight AI.

Usage
-----
    from firebase_client import get_db, save_document, get_collection

Credentials
-----------
The SDK is initialised from the service-account JSON placed at the
project root.  The path is resolved relative to this file so it works
regardless of where the process is launched from.
"""

import os
import logging
from typing import Any

import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
_DEFAULT_KEY_PATH = os.path.join(_PROJECT_ROOT, "firebase-service-account.json")


# ---------------------------------------------------------------------------
# Initialisation (singleton)
# ---------------------------------------------------------------------------

def _init_firebase() -> None:
    """Initialise the Firebase app exactly once per process."""
    if firebase_admin._apps:
        return  # Already initialised

    # 1. Try Streamlit secrets (preferred in cloud deployments)
    try:
        secret = dict(st.secrets["firebase"])
        cred = credentials.Certificate(secret)
        firebase_admin.initialize_app(cred)
        logger.info("Firebase initialised from Streamlit secrets.")
        return
    except (KeyError, FileNotFoundError):
        pass

    # 2. Fall back to the service-account JSON file on disk
    key_path = os.environ.get("FIREBASE_KEY_PATH", _DEFAULT_KEY_PATH)
    if os.path.exists(key_path):
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)
        logger.info("Firebase initialised from key file: %s", key_path)
        return

    raise FileNotFoundError(
        "Firebase service account key not found.\n"
        f"Expected at: {key_path}\n"
        "Set the FIREBASE_KEY_PATH env var or add credentials to "
        ".streamlit/secrets.toml"
    )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_db():
    """Return a cached Firestore client (initialises Firebase if needed)."""
    _init_firebase()
    return firestore.client()


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def save_document(collection: str, doc_id: str, data: dict) -> None:
    """Write (or overwrite) a document."""
    get_db().collection(collection).document(doc_id).set(data)
    logger.debug("Saved %s/%s", collection, doc_id)


def update_document(collection: str, doc_id: str, data: dict) -> None:
    """Merge-update an existing document (only provided fields are changed)."""
    get_db().collection(collection).document(doc_id).set(data, merge=True)
    logger.debug("Updated %s/%s", collection, doc_id)


def get_document(collection: str, doc_id: str):
    """Fetch a single document. Returns None if it doesn't exist."""
    doc = get_db().collection(collection).document(doc_id).get()
    return doc.to_dict() if doc.exists else None


def get_collection(collection: str, limit: int = 500) -> list:
    """
    Fetch all documents in a collection (up to *limit*).
    Each dict includes a special '_id' key with the Firestore document ID.
    """
    docs = get_db().collection(collection).limit(limit).stream()
    results = []
    for doc in docs:
        entry = doc.to_dict()
        entry["_id"] = doc.id
        results.append(entry)
    return results


def delete_document(collection: str, doc_id: str) -> None:
    """Delete a document from Firestore."""
    get_db().collection(collection).document(doc_id).delete()
    logger.debug("Deleted %s/%s", collection, doc_id)


def query_collection(collection: str, field: str, operator: str, value: Any, limit: int = 500) -> list:
    """
    Simple filter query.

    Parameters
    ----------
    collection : Firestore collection name (e.g. "predictions")
    field      : Field to filter on (e.g. "department")
    operator   : "==", "<", "<=", ">", ">=", "array_contains", "in"
    value      : Value to compare against
    limit      : Max results to return

    Example
    -------
        rows = query_collection("predictions", "department", "==", "Cardiology")
    """
    docs = (
        get_db()
        .collection(collection)
        .where(field, operator, value)
        .limit(limit)
        .stream()
    )
    results = []
    for doc in docs:
        entry = doc.to_dict()
        entry["_id"] = doc.id
        results.append(entry)
    return results
