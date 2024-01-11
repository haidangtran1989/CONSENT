import gzip


def load_documents(document_file_path):
    id_to_document = dict()
    with gzip.open(document_file_path, "rt", encoding="utf-8") as document_file:
        for line in document_file:
            parts = line.strip().split("\t")
            if len(parts) != 5:
                continue
            id_to_document[parts[0]] = line
    print("Loaded " + document_file_path)
    return id_to_document


id_to_document = load_documents("../data/resource/news-en-documents-2017-20181120.tsv.gz") | \
        load_documents("../data/resource/wiki-en-documents-20170920.tsv.gz")


def get_document(document_id):
    return id_to_document.get(document_id, None)


if __name__ == "__main__":
    print(get_document("news_5"))
