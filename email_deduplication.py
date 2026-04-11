import pandas as pd
from datasketch import MinHash, MinHashLSH


def get_minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for word in text.lower().split():
        m.update(word.encode("utf8"))
    return m


def apply_deduplication(df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """
    Scans all emails in the DataFrame for near-duplicates.
    Sets the 'duplicate' column to 1 for any email that is
    a near-duplicate of a previously seen email.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    seen_ids = set()

    for idx, row in df.iterrows():
        eid  = row["id"]
        m    = get_minhash(row["text"])
        hits = lsh.query(m)

        if hits or eid in seen_ids:
            # This email is a near-duplicate of something already seen
            df.at[idx, "duplicate"] = 1
        else:
            lsh.insert(eid, m)
            seen_ids.add(eid)

    return df