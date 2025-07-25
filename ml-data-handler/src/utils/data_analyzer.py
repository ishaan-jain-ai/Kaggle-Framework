def analyze_data(data):
    """
    Analyzes the given data and returns various statistics.

    Parameters:
    data (list or array-like): The input data to analyze.

    Returns:
    dict: A dictionary containing mean, median, and mode of the data.
    """
    import numpy as np
    from scipy import stats

    mean = np.mean(data)
    median = np.median(data)
    mode = stats.mode(data).mode[0]

    return {
        'mean': mean,
        'median': median,
        'mode': mode
    }