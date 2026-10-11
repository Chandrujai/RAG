# Results from two different question versions
# (each list is ordered: best match first)
search_1 = ["docA", "docB", "docC"]
search_2 = ["docB", "docA", "docD"]

def reciprocal_rank_fusion(searches, k=60):
    scores = {}
    for result_list in searches:
        for rank, doc in enumerate(result_list):
            # rank starts at 0, so we use rank+1
            scores[doc] = scores.get(doc, 0) + 1 / (rank + 1 + k)
    # sort documents by total score (highest first)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

final = reciprocal_rank_fusion([search_1, search_2])

print("Final combined ranking:")
for doc, score in final:
    print(doc, "-> score:", round(score, 4))