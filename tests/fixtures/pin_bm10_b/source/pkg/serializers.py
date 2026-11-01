def serialize_one(value):
    return {"value": value}


def serialize_batch(results):
    return [
        {"index": r.index, "item": r.item, "ok": r.ok, "value": r.value, "error": r.error}
        for r in results
    ]
