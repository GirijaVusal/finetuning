from datasets import Dataset, DatasetDict
from uuid import uuid4

def make_dual_nepali_to_eng_dataset(data):
    """
    Converts each entry into two rows:
      - One with Devanagari Nepali as source
      - One with Romanized Nepali as source
    data: list of dicts like:
      {"nepali": "...", "romanized_nep": "...", "eng": "..."}
    Returns: DatasetDict with 'train' split
    """
    rows = []
    for ex in data:
        eng = ex["eng"]
        nep_devanagari = ex["nepali"]
        nep_romanized = ex["romanized_nep"]

        # rows.append({
        #     "id": str(uuid4()),
        #     "translation": {
        #         "nep": nep_devanagari,
        #         "eng": eng
        #     }
        # })

        rows.append({
            "id": str(uuid4()),
            "translation": {
                "nep": nep_romanized,
                "eng": eng
            }
        })

    ds = Dataset.from_list(rows)
    return DatasetDict({"train": ds})

def load_nep_eng_ds():
    from dataset_builder.data import data
    return make_dual_nepali_to_eng_dataset(data)

if __name__ == "__main__":
    data = [
        {
            "nepali": "तिमी कस्तो छौ?",
            "romanized_nep": "timi kasto chau?",
            "eng": "How are you?"
        }
    ]

    dataset = make_dual_nepali_to_eng_dataset(data)
    for ex in dataset["train"]:
        print(ex)
