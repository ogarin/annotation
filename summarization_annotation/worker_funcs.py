def read_raw_cases(case_file):
    from crm_parser import normalize
    return list(normalize.yield_avro_records(case_file))

def read_raw_lcts(lct_file):
    from crm_parser.entities import livechattranscript as lct_parser

    case_additional_fields = {}
    for record in normalize.yield_avro_records(lct_file.path):
        case_additional_fields[record['Id']] = {'case_id': record['CaseId'], 'owner_id': record['OwnerId']}

    return [
        {
            "turns": [
                {
                    "speaker_name": turn.speaker_name,
                    "speaker_role": turn.speaker_role,
                    "content": turn.content,
                    "is_bot_chat": is_live_chat == 0
                }
                for is_live_chat, turns in enumerate([lct.bot, lct.livechat])
                for turn in turns
                if turn.piece_label == "utterance"
            ],
            "uid": lct.uid,
            **case_additional_fields[lct.uid]
        }
        for lct in lct_parser.parse_avro_chats(lct_file)
    ]


def _read_processed_chat_file(chat_file):
    import json
    import fsspec

    with fsspec.open(chat_file) as fp:
        for line in fp.readlines():
            chat = json.loads(line)
            yield dict(
                chat,
                chat_text="\n".join(
                    "%s: %s" % (turn["speaker_name"], turn["content"])
                    for turn in chat["turns"]
                    if not turn["is_bot_chat"] or turn["speaker_role"] == "Customer"
                ),
            )


def read_chats(spec):
    chat_file, uids = spec
    return [
        chat for chat in _read_processed_chat_file(chat_file) if chat["uid"] in uids
    ]


def read_chats_metadata(chat_file):
    return [
        {
            "chat_file": chat_file,
            "uid": chat["uid"],
            "case_id": chat["case"]["Id"],
            "case_origin": chat["case"]["Origin"],
            "n_words": len(chat["chat_text"].split()),
            "n_turns": len(chat["turns"]),
            "n_case_description_words": len(
                (chat["case"].get("Description") or "").split()
            ),
        }
        for chat in _read_processed_chat_file(chat_file)
    ]
