def corrective_rag(question, documents):
    grade = grade_documents(question, documents)

    if grade == "correct":
        data = documents
    elif grade == "ambiguous":
        data = documents + web_search(question)
    else:  # incorrect
        data = web_search(question)   # ignore bad docs, use web

    return "Answer using: " + str(data)


# Simple helper functions (just for demonstration)
def grade_documents(question, documents):
    # pretend the docs are not good enough
    return "incorrect"

def web_search(question):
    return ["(fresh result from the web about: " + question + ")"]


print(corrective_rag("Latest news about AI?", ["old unrelated text"]))