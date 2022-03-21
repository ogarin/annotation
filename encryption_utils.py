from Crypto.Cipher import AES
from Crypto.Hash import SHA256
from Crypto import Random
import json
import base64

SALT_SIZE = 12


def encrypt_by_sample(smp, doc):
    key = dervice_encrpytion_from_sample(smp)
    return encrypt_by_text_key(key, json.dumps(doc).encode("utf-8"))


def decrypt_by_sample(smp, enc_data):
    key = dervice_encrpytion_from_sample(smp)
    return json.loads(decrypt_by_text_key(key, enc_data).decode("utf-8"))


def encrypt_by_text_key(big_key: str, data_to_encrypt: bytes) -> dict:
    salt = Random.new().read(SALT_SIZE)
    key = SHA256.new((big_key).encode("utf-8") + salt).digest()
    IV = Random.new().read(AES.block_size)
    encryptor = AES.new(key, AES.MODE_CBC, IV)
    padding = (
        AES.block_size - len(data_to_encrypt) % AES.block_size
    )  # calculate needed padding
    data_to_encrypt += (
        bytes([padding]) * padding
    )  # Python 2.x: source += chr(padding) * padding
    data = (
        IV + salt + encryptor.encrypt(data_to_encrypt)
    )  # store the IV at the beginning and encrypt
    return base64.b64encode(data).decode("latin-1")


def decrypt_by_text_key(big_key: str, encrypted_data: bytes):
    source = base64.b64decode(encrypted_data.encode("latin-1"))
    IV = source[: AES.block_size]  # extract the IV from the beginning
    salt = source[
        AES.block_size : AES.block_size + SALT_SIZE
    ]  # extract the IV from the beginning
    key = SHA256.new(
        (big_key).encode("utf-8") + salt
    ).digest()  # use SHA-256 over our key to get a proper-sized AES key
    decryptor = AES.new(key, AES.MODE_CBC, IV)
    data = decryptor.decrypt(source[(AES.block_size + SALT_SIZE) :])  # decrypt
    padding = data[-1]  # pick the padding value from the end; Python 2.x: ord(data[-1])
    if (
        data[-padding:] != bytes([padding]) * padding
    ):  # Python 2.x: chr(padding) * padding
        raise ValueError("Invalid padding...")
    return data[:-padding]  # remove the padding


def dervice_encrpytion_from_sample(chat) -> str:
    return " ".join([turn["content"] for turn in chat["turns"]])
