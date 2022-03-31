from encryption_utils import encrypt_by_text_key, decrypt_by_text_key
import json

def test_encrypt_decrypt():
    key = """
    A: Bla bal
    B: no
    A: but Bla Bla
    B: ok
    """

    data = {"Text": "This Data will be encrypted"}

    encryped = encrypt_by_text_key(key, json.dumps(data).encode('utf-8'))
    data_dec = json.loads(decrypt_by_text_key(key, encryped).decode('utf-8'))

    assert data['Text'] == data_dec['Text']