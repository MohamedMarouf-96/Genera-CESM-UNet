import numpy as np
def fn_based_dcg(y_true,y_pred,y_score, k= None, log_base=2,):
    ranking = np.argsort(y_pred)[::-1]
    y_score = y_score * y_true
    y_true_sorted = y_true[ranking]
    discount = (np.arange(len(y_true)) + 1 - np.cumsum(1 - y_true_sorted))  / (np.arange(len(y_true)) + 1 )
    # discount = 1 / (np.log(np.cumsum(1 - y_true_sorted) + 2)) / np.log(log_base)
    if k is not None:
        discount[k:] = 0
    ranked_y_score = y_score[ranking]
    cumulative_gains = discount[:,np.newaxis].dot(ranked_y_score[np.newaxis,:])
    return np.average(cumulative_gains)
def dcg_sample_scores(y_true, y_score, k=None, log_base=2, ignore_ties = False) :
    """Compute Discounted Cumulative Gain.
    Sum the true scores ranked in the order induced by the predicted scores,
    after applying a logarithmic discount.
    This ranking metric yields a high value if true labels are ranked high by
    ``y_score``.
    Parameters
    ----------
    y_true : ndarray of shape (n_samples, n_labels)
        True targets of multilabel classification, or true scores of entities
        to be ranked.
    y_score : ndarray of shape (n_samples, n_labels)
        Target scores, can either be probability estimates, confidence values,
        or non-thresholded measure of decisions (as returned by
        "decision_function" on some classifiers).
    k : int, default=None
        Only consider the highest k scores in the ranking. If `None`, use all
        outputs.
    log_base : float, default=2
        Base of the logarithm used for the discount. A low value means a
        sharper discount (top results are more important).
    ignore_ties : bool, default=False
        Assume that there are no ties in y_score (which is likely to be the
        case if y_score is continuous) for efficiency gains.
    Returns
    -------
    discounted_cumulative_gain : ndarray of shape (n_samples,)
        The DCG score for each sample.
    See Also
    --------
    ndcg_score : The Discounted Cumulative Gain divided by the Ideal Discounted
        Cumulative Gain (the DCG obtained for a perfect ranking), in order to
        have a score between 0 and 1.
    """
    discount = 1 / (np.log(np.arange(y_true.shape[1]) + 2) / np.log(log_base))
    if k is not None:
        discount[k:] = 0
    if ignore_ties:
        ranking = np.argsort(y_score)[:, ::-1]
        ranked = y_true[np.arange(ranking.shape[0])[:, np.newaxis], ranking]
        cumulative_gains = discount.dot(ranked.T)
    else:
        discount_cumsum = np.cumsum(discount)
        cumulative_gains = [
            _tie_averaged_dcg(y_t, y_s, discount_cumsum)
            for y_t, y_s in zip(y_true, y_score)
        ]
        cumulative_gains = np.asarray(cumulative_gains)
    return cumulative_gains


def _tie_averaged_dcg(y_true, y_score, discount_cumsum):
    """
    Compute DCG by averaging over possible permutations of ties.
    The gain (`y_true`) of an index falling inside a tied group (in the order
    induced by `y_score`) is replaced by the average gain within this group.
    The discounted gain for a tied group is then the average `y_true` within
    this group times the sum of discounts of the corresponding ranks.
    This amounts to averaging scores for all possible orderings of the tied
    groups.
    (note in the case of dcg@k the discount is 0 after index k)
    Parameters
    ----------
    y_true : ndarray
        The true relevance scores.
    y_score : ndarray
        Predicted scores.
    discount_cumsum : ndarray
        Precomputed cumulative sum of the discounts.
    Returns
    -------
    discounted_cumulative_gain : float
        The discounted cumulative gain.
    References
    ----------
    McSherry, F., & Najork, M. (2008, March). Computing information retrieval
    performance measures efficiently in the presence of tied scores. In
    European conference on information retrieval (pp. 414-421). Springer,
    Berlin, Heidelberg.
    """
    _, inv, counts = np.unique(-y_score, return_inverse=True, return_counts=True)
    ranked = np.zeros(len(counts))
    np.add.at(ranked, inv, y_true)
    ranked /= counts
    groups = np.cumsum(counts) - 1
    discount_sums = np.empty(len(counts))
    discount_sums[0] = discount_cumsum[groups[0]]
    discount_sums[1:] = np.diff(discount_cumsum[groups])
    return (ranked * discount_sums).sum()