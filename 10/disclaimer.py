import sys
import os

disclaimer_text = """⚠️  WARNING: EDUCATIONAL CONTENT ONLY ⚠️

This notebook demonstrates AI Red Teaming. Executing these cells may produce content that is offensive or unsafe.
By proceeding, you acknowledge this is for didactic purposes only.
"""
disclaimer_env = {"key": "DISCLAIMER_ACCEPTED", "value": "I AGREE"}

def require_consent(prompt: str = f"Type '{disclaimer_env['value']}' to accept these terms and proceed: ") -> bool:
    if is_consent_provided():
        return True
    print(disclaimer_text)
    response = input(prompt)
    if response.strip().upper() == disclaimer_env["value"].upper():
        os.environ[disclaimer_env["key"]] = disclaimer_env["value"]
        print("✅ Disclaimer accepted. You may proceed.")
        return True
    sys.exit("❌ Disclaimer not accepted. Execution stopped.")

def is_consent_provided(env_var: str = disclaimer_env["key"]) -> bool:
    return os.environ.get(env_var, "").strip().upper() == disclaimer_env["value"].upper()