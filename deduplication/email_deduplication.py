from datasketch import MinHash, MinHashLSH

lsh = MinHashLSH(threshold=0.85, num_perm=128)
email_store = {}  # id -> MinHash

def get_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=128)
    for word in text.lower().split():
        m.update(word.encode("utf8"))
    return m

def is_duplicate(email_id: str, text: str) -> bool:
    m = get_minhash(text)
    result = lsh.query(m)
    if result:
        return True
    lsh.insert(email_id, m)
    email_store[email_id] = m
    return False