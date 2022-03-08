def read_raw_lct(lct_file):
    from crm_parser.entities import livechattranscript as lct_parser
    live_transcripts = []
    for lct in lct_parser.parse_avro_chats(lct_file):
        live_transcripts.append({
            "turns": [
                {
                    "speaker_name": ce.speaker_name,
                    "speaker_role": ce.speaker_role,
                    "content": ce.content
                }
                for ce in lct.livechat
                if ce.piece_label == 'utterance'
            ],
            'uid': lct.uid
        })

    return live_transcripts


def _read_processed_chat_file(chat_file):
    import json
    import fsspec
    with fsspec.open(chat_file) as fp:
        for line in fp.readlines():
            chat = json.loads(line)
            yield dict(chat, chat_text="\n".join(
                "%s: %s" % (turn["speaker_name"], turn["content"])
                for turn in chat["turns"]
            ))


def read_chats(spec):
    chat_file, uids = spec
    return [
        chat
        for chat in _read_processed_chat_file(chat_file)
        if chat["uid"] in uids
    ]


def read_chats_metadata(chat_file):
    return [
        {
            'chat_file': chat_file,
            'uid': chat['uid'],
            'n_words': len(chat['chat_text'].split()),
            'n_turns': len(chat['turns'])
        }
        for chat in _read_processed_chat_file(chat_file)
    ]
