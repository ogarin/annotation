from encryption_utils import decrypt_using_sample, encrypt_using_sample
import json

def test_encrypt_decrypt():
    key = """
    A: Bla bal
    B: no
    A: but Bla Bla
    B: ok
    """

    data = {"Text": "This Data will be encrypted"}

    encryped = encrypt_using_sample(key, json.dumps(data).encode('utf-8'))
    data_dec = json.loads(decrypt_using_sample(key, encryped).decode('utf-8'))

    assert data['Text'] == data_dec['Text']