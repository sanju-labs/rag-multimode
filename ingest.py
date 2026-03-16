# ingest.py
# Run this script to index a document into the vector store.
#
# Usage:
#   python ingest.py "Apple 2025.pdf"
#   python ingest.py "chart.png"
#   python ingest.py "relationships.json"

import sys
from ingestor import load_documents, split_documents
from vectorstore import build_vectorstore


def main():
    # Check a filename was provided
    if len(sys.argv) < 2:
        print("Please provide a file to ingest.")
        print('Usage: python ingest.py "Apple 2025.pdf"')
        sys.exit(1)

    file_path = sys.argv[1]

    print(f"\nLoading '{file_path}'...")
    docs = load_documents(file_path)
    print(f"Loaded {len(docs)} page(s).")

    print("Splitting into chunks...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("Building vector index...")
    build_vectorstore(chunks)

    print("\nAll done! Now run:")
    print('  python query.py "What was the revenue in Q3?"')


if __name__ == "__main__":
    main()
