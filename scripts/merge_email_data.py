import json

with open('data/emails/real_emails.json') as f:
    real = json.load(f)

with open('data/emails/sample_emails.json') as f:
    synthetic = json.load(f)

combined = real + synthetic

with open('data/emails/all_emails.json', 'w') as f:
    json.dump(combined, f, indent=2)

print(f"Total: {len(combined)} emails ({len(real)} real + {len(synthetic)} synthetic)")