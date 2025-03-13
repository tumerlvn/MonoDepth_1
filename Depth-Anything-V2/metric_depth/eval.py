from metric_depth.util.metric import eval_depth


def eval_model(preds, targets):
    # Initialize accumulators for each metric
    metrics = {
        'd1': 0.0,
        'd2': 0.0,
        'd3': 0.0,
        'abs_rel': 0.0,
        'sq_rel': 0.0,
        'rmse': 0.0,
        'rmse_log': 0.0,
        'log10': 0.0,
        'silog': 0.0,
    }
    
    # Number of pairs
    num_pairs = len(preds)
    
    # Iterate over each prediction-target pair
    for pred, target in zip(preds, targets):
        result = eval_depth(pred, target)
        
        # Accumulate results for each metric
        for key in metrics:
            metrics[key] += result[key]
    
    # Compute the average for each metric
    for key in metrics:
        metrics[key] /= num_pairs
    
    return metrics
