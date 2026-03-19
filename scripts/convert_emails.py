import email
import json
import os
import base64
from pathlib import Path

def eml_to_json(eml_path):
    with open(eml_path, 'rb') as f:
        msg = email.message_from_binary_file(f)

    data = {
        "subject": msg["subject"],
        "from": msg["from"],
        "to": msg["to"],
        "date": msg["date"],
        "body": "",
        "attachments": []
    }

    for part in msg.walk():
        content_type = part.get_content_type()
        disposition = str(part.get("Content-Disposition"))

        # extract plain text body
        if content_type == "text/plain" and "attachment" not in disposition:
            data["body"] = part.get_payload(decode=True).decode("utf-8", errors="ignore")

        # extract attachments
        elif "attachment" in disposition:
            filename = part.get_filename()
            payload = part.get_payload(decode=True)
            data["attachments"].append({
                "filename": filename,
                "content_type": content_type,
                "data": base64.b64encode(payload).decode("utf-8") if payload else None
            })

    return data


def convert_folder(eml_dir, output_file):
    results = []
    for eml_file in Path(eml_dir).glob("*.eml"):
        try:
            results.append(eml_to_json(eml_file))
            print(f"✓ Converted: {eml_file.name}")
        except Exception as e:
            print(f"✗ Failed {eml_file.name}: {e}")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} emails to {output_file}")


convert_folder("data/emails/raw", "data/emails/real_emails.json")