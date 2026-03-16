# query.py
# Usage:
#   python query.py "What was Apple's revenue?"
#   python query.py "Compare Q1 and Q3 margins and explain the trend"

import sys
import time
from rag import ask_stream


def main():
    if len(sys.argv) < 2:
        print("Please provide a question.")
        print('Usage: python query.py "What was Apple\'s revenue?"')
        sys.exit(1)

    question = sys.argv[1]

    print(f"\nQuestion: {question}\n")

    start = time.time()
    result = ask_stream(question)
    end = time.time()

    print(f"\nSource         : {result['sources']}")
    print(f"Chunks checked : {result['chunks_checked']}")
    print(f"Routed to      : {result['routed_to']}")
    print(f"Time taken     : {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
