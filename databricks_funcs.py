
def _read_chat_file(chat_file):
    import json
    import fsspec
    with fsspec.open(chat_file) as fp:
        for line in fp.readlines():
            yield json.loads(line)


def read_chats(spec):
    chat_file, uids = spec
    return [
        chat
        for chat in _read_chat_file(chat_file)
        if chat["uid"] in uids
    ]


def read_chats_metadata(chat_file):
    return [
        {
            'chat_file': chat_file,
            'uid': chat['uid'],
            'n_words': len(chat['chat_text'].split()),
            'n_turns': len(chat['chat_text'].splitlines())
        }
        for chat in _read_chat_file(chat_file)
    ]
