import math
def map_score(search_result_relevances: list[int], cut_off=10) -> float:
    
    ranked_lists = [search_result_relevances]
    ap_sum = 0
    
    for ranked_list in ranked_lists:
        missing = ranked_list[cut_off:].count(1)
        precision_sum = 0
        relevant_docs = 0
        for i, doc in enumerate(ranked_list[0:cut_off]):
            if doc == 1:  # doc is relevant
                relevant_docs += 1
                precision_sum += relevant_docs / (i + 1)
        if relevant_docs == 0:
            ap_sum += 0
        else:
            ap_sum += precision_sum / (relevant_docs + missing)
    map = ap_sum / len(ranked_lists)
    
    return map


def ndcg_score(search_result_relevances: list[float], 
               ideal_relevance_score_ordering: list[float], cut_off=10):
    """
    Calculates the normalized discounted cumulative gain (NDCG) given a lists of relevance scores.
    Relevance scores can be ints or floats, depending on how the data was labeled for relevance.

    Args:
        search_result_relevances: 
            A list of relevance scores for the results returned by your ranking function in the
            order in which they were returned. These are the human-derived document relevance scores,
            *not* the model generated scores.
            
        ideal_relevance_score_ordering: The list of relevance scores for results for a query, sorted by relevance score in descending order.
            Use this list to calculate IDCG (Ideal DCG).

        cut_off: The default cut-off is 10.

    Returns:
        The NDCG score
    """
    # TODO: Implement NDCG
    # DCG = r1 + r2 /log22 + r3 /log23 + ... rn/log2n
    # NCDG = DCG/IDCG
    # IDCG sorted ideal
    if len(search_result_relevances) < cut_off:
        cut_off = len(search_result_relevances)
    DCG = 0
    for i in range(cut_off):
        if i == 0:
            DCG += search_result_relevances[i]
        else:
            DCG += search_result_relevances[i]/math.log2(i+1)
    IDCG = 0
    for i in range(cut_off):
        if i == 0:
            IDCG += ideal_relevance_score_ordering[i]
        else:
            IDCG += ideal_relevance_score_ordering[i]/math.log2(i+1)
    if IDCG == 0:
        return 0
    else:
        return DCG/IDCG




def run_relevance_tests(relevance_data_filename: str, ranker) -> dict[str, float]:
    # TODO: Implement running relevance test for the search system for multiple queries.
    """
    Measures the performance of the IR system using metrics, such as MAP and NDCG.
    
    Args:
        relevance_data_filename [str]: The filename containing the relevance data to be loaded

        ranker: A ranker configured with a particular scoring function to search through the document collection.
                This is probably either a Ranker or a L2RRanker object, but something that has a query() method

    Returns:
        A dictionary containing both MAP and NDCG scores
    """

    # TODO: Run each of the dataset's queries through your ranking function.

    # TODO: For each query's result, calculate the MAP and NDCG for every single query and average them out.

    # NOTE: MAP requires using binary judgments of relevant (1) or not (0). You should use relevance 
    # scores of (1,2,3) as not-relevant, and (4,5) as relevant.

    # NOTE: NDCG can use any scoring range, so no conversion is needed in the human relevance scores.

    # TODO: Compute the average MAP and NDCG across all queries and return the scores. 

    return {'map': 0, 'ndcg': 0}