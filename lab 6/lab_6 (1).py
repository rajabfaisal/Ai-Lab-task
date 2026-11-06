import csv
import re
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DOCS = {
    "late": "Assessment Rules for Undergraduate Courses. All coursework must be submitted through the learning portal before the stated deadline. Late submission is normally penalized by ten percent for each day after the deadline. Illness extensions require medical evidence before the deadline. The evidence must be signed by Dr. Khan or another registered doctor and must show the dates of illness. Students who cannot get a signed note in time should inform the course coordinator by email and then send the note within three days. Extensions are not given for travel, work shifts or computer problems unless the coordinator agrees in writing. Group projects follow the same rule, and every member must send their own evidence if they ask for an extension. The coordinator replies to extension requests within two working days and keeps a record of every decision for later checks by the department. Students may appeal a decision by writing to the head of department within one week.",
    "late_pg": "Assessment Rules for Postgraduate Courses. Postgraduate students submit coursework through the learning portal and must follow the deadline given in the course outline. Late submission is not accepted without prior approval from the supervisor. Illness extensions require medical evidence within seven days after the deadline. The evidence must show the dates of illness and must be sent to the supervisor and the graduate office. Research thesis chapters follow a different timetable that is agreed at the start of each semester, and the supervisor may change it when the project plan changes. Students who miss a deadline without approval receive a mark of zero for that piece of work, but they may request a review through the graduate office. The graduate office replies within five working days and keeps all decisions in the student record for the final examination board. Students should keep a copy of every email and form they send, because the office may ask for them during a review.",
    "password": "IT Help Desk Guide for Accounts. Reset a forgotten password using the recovery link. The link is sent to the registered student email and stays active for thirty minutes only. Never share recovery codes with anyone, including staff who contact you by phone or chat, because the help desk will never ask for them. If the recovery email does not arrive, check the spam folder first and then ask the help desk to confirm the email address stored in the account, which they can do only after checking your student card at the service counter during working hours. Passwords must have at least twelve characters and must not repeat any of the last five passwords. Accounts are locked after five wrong attempts, and the lock is removed automatically after fifteen minutes. Students should also turn on two step login in the account settings page to protect their files and email. The help desk never changes a password by phone.",
    "assignment": "Assignment Submission Guide. Upload assignments through the learning portal. Submit the final file as a PDF unless the course outline asks for a different format. Name the file with your registration number and the course code so that the marker can find it quickly. Each upload page shows the deadline and the maximum file size, which is twenty megabytes for most courses. Students can upload a new version any number of times before the deadline, and the portal keeps only the latest version for marking. After the deadline the upload page closes, and the course coordinator must open it again for any late work that was approved. A receipt email is sent after every successful upload, and students should keep it as proof of submission. If the portal is down near the deadline, send the file to the coordinator by email and attach a screenshot of the error message. Students should always keep a backup copy of the file.",
    "contractor": "Remote Work Policy. Employees may request remote work with manager approval. Employee rules do not apply to contractors. Contractors can work remotely only if the signed contract permits it. A contractor who wants a change must ask the project owner, who will check the contract and the security rules before replying. Remote work days must be recorded in the attendance system by the end of each week. Staff working from home must use the company laptop and the approved virtual private network, and they must not store company files on personal devices. Managers review remote work requests every three months and may cancel an arrangement when team needs change. Employees who work remotely for more than two weeks in a row must send a short report to human resources. Contractors paid by the hour must also record their remote hours in the same system so the invoice can be checked. Questions about this policy go to human resources.",
    "printer": "Equipment Troubleshooting Guide. For blank printer pages, check ink levels and run the cleaning utility. Print a test page afterwards to see if the problem is fixed. If the test page is still blank, remove the ink cartridge, check the protective tape and place it again firmly. Warning: switch off the printer and unplug it before opening the cover, because the inside parts can stay hot for several minutes. For a blank scanner image, check document placement and select the correct scanning source. The scanner glass should be cleaned with a soft dry cloth only. Paper jams are cleared by pulling the sheet slowly in the direction of the paper path. Do not use sharp tools inside the machine. Devices that still fail after these steps must be sent to the repair desk with the fault form, which is available on the help desk page of the intranet. Always record the printer model and the error code before calling the repair desk.",
}

EVIDENCE = {
    "late": "Illness extensions require medical evidence before the deadline.",
    "password": "Reset a forgotten password using the recovery link.",
    "assignment": "Upload assignments through the learning portal.",
    "contractor": "Contractors can work remotely only if the signed contract permits it.",
    "printer": "For blank printer pages, check ink levels and run the cleaning utility.",
}

QUERIES = [
    ("What is required for illness extensions?", "late", "dev"),
    ("How do I reset a forgotten password?", "password", "dev"),
    ("Where do I upload assignments?", "assignment", "dev"),
    ("Can contractors work remotely?", "contractor", "dev"),
    ("What do I do for blank printer pages?", "printer", "test"),
    ("How can I recover my login access?", "password", "test"),
    ("How should I hand in my coursework?", "assignment", "test"),
    ("What proof is needed when I was ill?", "late", "test"),
    ("What are tomorrow's flight prices?", None, "test"),
    ("Who won the football match?", None, "test"),
]

FAQ = [
    {"id": "faq:password", "question": "How do I reset my password?",
     "answer": "Use the recovery link on the login page.", "source": "it_help.txt"},
    {"id": "faq:submit", "question": "Where do I submit my assignment?",
     "answer": "Use the assignment upload page in the learning portal.", "source": "assessment.txt"},
    {"id": "faq:late", "question": "Can illness justify a late submission?",
     "answer": "Request an extension with medical evidence before the deadline.", "source": "handbook.txt"},
    {"id": "faq:library", "question": "What are the library timings?",
     "answer": "The library is open from 8 am to 8 pm on weekdays.", "source": "library.txt"},
    {"id": "faq:wifi", "question": "How do I connect to campus WiFi?",
     "answer": "Choose the campus network and log in with your student account.", "source": "it_help.txt"},
]

class TfidfSearch:
    def __init__(self, records):
        self.records = records
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([r["text"] for r in records])

    def search(self, query, k=3):
        if query.strip() == "":
            return []
        q = self.vectorizer.transform([query])
        if q.nnz == 0:
            return []
        scores = (q @ self.matrix.T).toarray()[0]
        return top_results(self.records, scores, k)

class EmbeddingSearch:
    def __init__(self, records, model):
        self.records = records
        self.model = model
        texts = [r["text"] for r in records]
        self.matrix = model.encode(texts, normalize_embeddings=True)

    def search(self, query, k=3):
        if query.strip() == "":
            return []
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.matrix @ q
        return top_results(self.records, scores, k)

def top_results(records, scores, k):
    order = np.argsort(-scores, kind="stable")[:k]
    return [(records[i], float(scores[i])) for i in order]

def make_search(records, method, model):
    if method == "tfidf":
        return TfidfSearch(records)
    return EmbeddingSearch(records, model)

def load_model():
    try:
        from sentence_transformers import SentenceTransformer
        start = time.time()
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        print("Model load seconds:", round(time.time() - start, 2))
        return model
    except Exception as error:
        print("Embedding model not available:", type(error).__name__)
        return None

def windows(text, size=40, overlap=10, offset=0):
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("size must be positive and overlap must be smaller than size")
    words = list(re.finditer(r"\S+", text))
    result = []
    i = 0
    while i < len(words):
        j = min(i + size, len(words))
        result.append((offset + words[i].start(), offset + words[j - 1].end()))
        if j == len(words):
            break
        i = i + size - overlap
    return result

def sentence_chunks(text, size=40):
    cuts = [0] + [m.end() for m in re.finditer(r"[.!?]\s+", text)] + [len(text)]
    chunks = []
    start = None
    end = 0
    count = 0
    for a, b in zip(cuts, cuts[1:]):
        n = len(text[a:b].split())
        if n == 0:
            continue
        if n > size:
            if start is not None:
                chunks.append((start, end))
                start = None
                count = 0
            chunks.extend(windows(text[a:b], size, 0, a))
        elif start is not None and count + n <= size:
            end = b
            count = count + n
        else:
            if start is not None:
                chunks.append((start, end))
            start = a
            end = b
            count = n
    if start is not None:
        chunks.append((start, end))
    return chunks

def make_chunks(docs, strategy):
    result = []
    for doc_id, text in docs.items():
        if strategy == "sentence":
            spans = sentence_chunks(text)
        else:
            spans = windows(text, 40, int(strategy))
        for i in range(len(spans)):
            a, b = spans[i]
            result.append({"id": doc_id + ":" + str(i), "doc": doc_id,
                           "start": a, "end": b, "text": text[a:b]})
    return result

def show_tokens(model):
    print("TASK 1: TOKEN COMPARISON")
    samples = ["AI, search!", "I can't log in.", "help@example.com",
               "CS4944 late!", "", "یہ ایک مثال ہے"]
    for text in samples:
        parts = text.split()
        words = re.findall(r"\b\w+\b", text)
        print("Original:", repr(text))
        print("  Whitespace:", parts, len(parts))
        print("  Word-like:", words, len(words))
        if model is not None and text in samples[:2]:
            pieces = model.tokenizer.tokenize(text)
            print("  Model subwords:", pieces, len(pieces))
    print()

def reply(engine, query):
    if query.strip() == "":
        return "Please enter a question."
    for item in FAQ:
        if query.strip().lower() == item["question"].lower():
            return item["answer"] + " [source: " + item["source"] + "]"
    hits = engine.search(query)
    if len(hits) == 0:
        return "No matching FAQ. Please contact the help desk."
    lines = ["Possible matches; choose or rephrase your question:"]
    for r, score in hits:
        lines.append(r["question"] + " (similarity " + str(round(score, 3)) + ")")
    return "\n".join(lines)

def chatbot_demo():
    print("TASK 1: FAQ CHATBOT (TF-IDF)")
    records = []
    for item in FAQ:
        r = dict(item)
        r["text"] = item["question"] + " " + item["answer"]
        records.append(r)
    engine = TfidfSearch(records)
    for query in ["How do I reset my password?", "password recovery",
                  "library", "", "volcano geology"]:
        print("You:", repr(query))
        print("Bot:", reply(engine, query))
    print()

def chunk_demo():
    print("TASK 2: CHUNKING TESTS")
    try:
        windows("a b", 2, 2)
    except ValueError as error:
        print("Invalid overlap rejected:", error)
    long_sentence = " ".join(["word"] * 95) + "."
    parts = sentence_chunks(long_sentence)
    print("Empty text chunks:", len(windows("")))
    print("Short text chunks:", len(windows("short text")))
    print("95-word sentence chunks:", len(parts),
          [len(long_sentence[a:b].split()) for a, b in parts])
    print("Abbreviation test (Dr. Khan) sentence chunks:", len(sentence_chunks(DOCS["late"])))
    print()
    print("TASK 2: CHUNK COMPARISON")
    for name, text in DOCS.items():
        print(name, "words:", len(text.split()))
    total_words = sum(len(t.split()) for t in DOCS.values())
    for strategy in ["0", "10", "sentence"]:
        chunks = make_chunks(DOCS, strategy)
        sizes = [len(c["text"].split()) for c in chunks]
        kept = 0
        for doc, evidence in EVIDENCE.items():
            for c in chunks:
                if c["doc"] == doc and evidence in c["text"]:
                    kept += 1
                    break
        print("strategy =", strategy, "| chunks =", len(chunks),
              "| avg words =", round(np.mean(sizes), 1), "| max words =", max(sizes),
              "| duplicate words =", sum(sizes) - total_words,
              "| evidence kept =", kept, "/", len(EVIDENCE))
    print()

def find_rank(hits, doc, evidence):
    rank = 1
    for r, score in hits:
        if r["doc"] == doc and evidence in r["text"]:
            return rank
        rank += 1
    return 0

def get_metrics(ranks):
    if len(ranks) == 0:
        return 0, 0
    hit = sum(1 for r in ranks if r > 0) / len(ranks)
    mrr = sum(1 / r for r in ranks if r > 0) / len(ranks)
    return round(hit, 3), round(mrr, 3)

def evaluate_all(model):
    print("TASK 3: RETRIEVAL EVALUATION")
    methods = ["tfidf"]
    if model is not None:
        methods.append("embedding")
    results = {}
    failures = []
    log = []
    unanswerable = []
    for strategy in ["0", "10", "sentence"]:
        chunks = make_chunks(DOCS, strategy)
        for method in methods:
            start = time.time()
            engine = make_search(chunks, method, model)
            build_time = time.time() - start
            dev_ranks = []
            test_ranks = []
            start = time.time()
            for query, doc, split in QUERIES:
                hits = engine.search(query)
                for i in range(len(hits)):
                    r, score = hits[i]
                    log.append([strategy, method, query, split, i + 1,
                                round(score, 4), r["id"], r["doc"], r["start"], r["end"]])
                if doc is None:
                    if len(hits) > 0:
                        unanswerable.append((strategy, method, query, hits[0][0]["id"], round(hits[0][1], 3)))
                    else:
                        unanswerable.append((strategy, method, query, "none", 0))
                    continue
                rank = find_rank(hits, doc, EVIDENCE[doc])
                if split == "dev":
                    dev_ranks.append(rank)
                else:
                    test_ranks.append(rank)
                if rank != 1:
                    got = hits[0][0]["id"] if len(hits) > 0 else "none"
                    failures.append((strategy, method, query, rank, got))
            search_time = time.time() - start
            dev = get_metrics(dev_ranks)
            test = get_metrics(test_ranks)
            results[(strategy, method)] = (dev, test)
            print("strategy =", strategy, "| method =", method, "| chunks =", len(chunks),
                  "| dev (Hit, MRR) =", dev, "| test (Hit, MRR) =", test,
                  "| build =", round(build_time, 4), "s | search =", round(search_time, 4), "s")
    with open("retrieval_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "method", "query", "split", "rank", "score",
                         "chunk_id", "doc", "start", "end"])
        writer.writerows(log)
    print("Saved retrieval_log.csv with", len(log), "rows")
    print()
    print("FAILURES (answer not at rank 1; rank 0 means not in top 3)")
    for item in failures:
        print(item)
    print()
    print("UNANSWERABLE QUERIES (top result is still returned)")
    for item in unanswerable:
        if item[0] == "sentence":
            print(item)
    print()
    best = None
    for key in results:
        if best is None:
            best = key
        else:
            dev_best = results[best][0]
            dev_now = results[key][0]
            if (dev_now[1], dev_now[0]) > (dev_best[1], dev_best[0]):
                best = key
    print("Best on development data:", best, "dev:", results[best][0], "test:", results[best][1])
    print()
    return best

def evidence_demo(model, best):
    print("DOCUMENT EVIDENCE ASSISTANT")
    strategy, method = best
    engine = make_search(make_chunks(DOCS, strategy), method, model)
    for query in ["How do I reset a forgotten password?", "How can I recover my login access?", ""]:
        print("Question:", repr(query))
        hits = engine.search(query)
        if len(hits) == 0:
            print("No candidate evidence; rephrase or contact a human.")
            continue
        print("Candidate evidence; relevance must be checked:")
        number = 1
        for r, score in hits:
            print(number, r["id"], round(score, 3), "characters", r["start"], r["end"])
            number += 1
        r, score = hits[0]
        print("User chose 1")
        print("Retrieved passage:", r["text"])
        print("Source:", r["doc"], "characters", r["start"], r["end"])
    print()

def embedding_demo(model):
    print("EMBEDDING CHECK")
    if model is None:
        print("Skipped because the model is not available in this run.")
        print()
        return
    passages = ["Reset a forgotten password using the recovery link.",
                "Submit assignments through the learning portal.",
                "Check ink levels when a printer produces blank pages."]
    C = model.encode(passages, normalize_embeddings=True)
    Q = model.encode(["How can I recover my login?"], normalize_embeddings=True)
    scores = (Q @ C.T)[0]
    print("Embedding shape:", C.shape)
    for i in np.argsort(-scores):
        print(i, round(float(scores[i]), 3), passages[i])
    print()

model = load_model()
print()
show_tokens(model)
chatbot_demo()
chunk_demo()
embedding_demo(model)
best = evaluate_all(model)
evidence_demo(model, best)
