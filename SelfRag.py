def self_rag(question, documents):
    # Step 1: is any document relevant?
    relevant = [d for d in documents if is_relevant(d, question)]

    if not relevant:
        return "No good information found. Need to search again."

    # Step 2: build an answer from relevant docs
    answer = make_answer(relevant)

    print("Step 2 - Answer generated:", answer)

    # Step 3: self-check -- is the answer supported?
    if not is_supported(answer, relevant):
        return "Answer not supported. Try again."
    
    # print("Step 3 - Answer generated:", answer)
    return answer


# Simple helper functions (just for demonstration)
def is_relevant(doc, question):
    variable = any(word in doc.lower() for word in question.lower().split())
    print("is_relevant - result for '", doc, "':", variable)
    return variable

def make_answer(docs):
    return "Based on the data: " + docs[1]

def is_supported(answer, docs):
    return True   # in a real app, the AI would judge this


# Try it
docs = ["Python is a programming language.", "Python is an animal."]
print(self_rag("What is python?", docs))