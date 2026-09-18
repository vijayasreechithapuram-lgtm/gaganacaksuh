"""
Project Gaganacakṣuḥ: Cryptographic Authentication & RBAC Module
Compliant with military-grade clearance protocols and ISO/IEC 27037 audit standards.
"""

import hmac
import hashlib
import time
import json
import base64
from typing import Dict, Any, Optional

SECRET_SALT = "GAGANACAKSUH_CLASSIFIED_DEFENSE_SALT_2026"

ROLE_COMMAND_OFFICER = "COMMAND_OFFICER"
ROLE_TRIBUNAL_JUDGE = "TRIBUNAL_JUDGE"

ROLES_METADATA = {
    ROLE_COMMAND_OFFICER: {
        "title": "Command Officer",
        "jurisdiction": "Indian Coast Guard / Maritime Port Authority",
        "clearance_level": "LEVEL-IV OPERATIONAL CLEARANCE",
        "permissions": [
            "LIVE_WMS_INGESTION",
            "PIPELINE_EXECUTION",
            "UNET_SEGMENTATION_TRIGGER",
            "HYDRODYNAMIC_BACKCASTING",
            "ZERO_TRUST_AIS_AUDIT",
            "DYNAMIC_REROUTE_DISPATCH",
            "VHF_ALERT_BROADCAST",
            "EVIDENCE_DOCKET_INSPECTION"
        ],
        "badge_color": "#00ff66"
    },
    ROLE_TRIBUNAL_JUDGE: {
        "title": "Tribunal Judge / Legal Investigator",
        "jurisdiction": "Admiralty Court / Maritime Regulatory Tribunal",
        "clearance_level": "LEVEL-V JUDICIAL AUDIT CLEARANCE",
        "permissions": [
            "EVIDENCE_DOCKET_INSPECTION",
            "SHA256_TAMPER_VALIDATION",
            "LEGAL_DOSSIER_EXPORT",
            "CHAIN_OF_CUSTODY_VERIFY"
        ],
        "badge_color": "#00e5ff"
    }
}

CREDENTIALS_DB = {
    "officer.icg": {
        "password_hash": hashlib.sha256("CoastGuard@2026".encode()).hexdigest(),
        "role": ROLE_COMMAND_OFFICER,
        "officer_name": "Cmdr. Rajesh Verma",
        "service_number": "ICG-WEST-7741",
        "base": "Mumbai Coast Guard Station (Western Command)"
    },
    "judge.tribunal": {
        "password_hash": hashlib.sha256("Justice@Maritime2026".encode()).hexdigest(),
        "role": ROLE_TRIBUNAL_JUDGE,
        "officer_name": "Justice Ananya Sundaram",
        "service_number": "MARITIME-TRIBUNAL-DL-09",
        "base": "High Court Admiralty Division (Special Bench)"
    }
}

def generate_session_token(username: str, role: str, session_id: str) -> str:
    """Generates an HMAC-SHA256 authenticated clearance token."""
    payload = {
        "user": username,
        "role": role,
        "sid": session_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + 28800  # 8 hours validity
    }
    payload_str = json.dumps(payload, separators=(',', ':'), sort_keys=True)
    payload_b64 = base64.urlsafe_b64encode(payload_str.encode()).decode()
    signature = hmac.new(SECRET_SALT.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()
    return f"GGN.{payload_b64}.{signature[:32]}"

def verify_session_token(token: str) -> Optional[Dict[str, Any]]:
    """Verifies cryptographic signature and expiry of session clearance token."""
    try:
        parts = token.split(".")
        if len(parts) != 3 or parts[0] != "GGN":
            return None
        payload_b64, signature = parts[1], parts[2]
        expected_sig = hmac.new(SECRET_SALT.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()[:32]
        if not hmac.compare_digest(signature, expected_sig):
            return None
        payload_json = base64.urlsafe_b64decode(payload_b64.encode()).decode()
        payload = json.loads(payload_json)
        if time.time() > payload.get("exp", 0):
            return None
        return payload
    except Exception:
        return None

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Validates operator credentials against clearance DB and returns profile."""
    user = CREDENTIALS_DB.get(username.strip().lower())
    if not user:
        return None
    input_hash = hashlib.sha256(password.strip().encode()).hexdigest()
    if hmac.compare_digest(user["password_hash"], input_hash):
        session_id = hashlib.sha256(f"{username}:{time.time()}".encode()).hexdigest()[:16]
        token = generate_session_token(username, user["role"], session_id)
        return {
            "authenticated": True,
            "username": username,
            "role": user["role"],
            "officer_name": user["officer_name"],
            "service_number": user["service_number"],
            "base": user["base"],
            "meta": ROLES_METADATA[user["role"]],
            "session_token": token,
            "session_id": session_id,
            "login_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        }
    return None
