import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import cohen_kappa_score
import numpy as np

# Load the Excel file
file_path = './data/saber文本性打分汇总.xlsx'  # 修改为你的文件路径
data = pd.read_excel(file_path)


# Function to calculate Concordance Correlation Coefficient (CCC)
def concordance_correlation_coefficient(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    covariance = np.cov(y_true, y_pred)[0][1]
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    return ccc


# Calculate Pearson, Spearman, Kendall, CCC for each judge and the average score
judges = ['评委1', '评委2', '评委3', '评委4', '评委5', '评委6', '评委平均分']
model_scores = data['大模型']

results = []

for judge in judges:
    judge_scores = data[judge]

    # Pearson Correlation
    pearson_corr, _ = pearsonr(judge_scores, model_scores)

    # Spearman Correlation
    spearman_corr, _ = spearmanr(judge_scores, model_scores)

    # Kendall's Tau
    kendall_corr, _ = kendalltau(judge_scores, model_scores)

    # CCC
    ccc = concordance_correlation_coefficient(judge_scores, model_scores)

    # Store results
    results.append({
        'Judge': judge,
        'Pearson': round(pearson_corr, 2),
        'Spearman': round(spearman_corr, 2),
        'Kendall': round(kendall_corr, 2),
        'CCC': round(ccc, 2)
    })


# Function to calculate Quadratic Weighted Kappa (QWK)
def quadratic_weighted_kappa(y_true, y_pred, min_rating=None, max_rating=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if min_rating is None:
        min_rating = min(y_true.min(), y_pred.min())
    if max_rating is None:
        max_rating = max(y_true.max(), y_pred.max())

    num_ratings = int(max_rating - min_rating + 1)
    hist_true = np.zeros(num_ratings)
    hist_pred = np.zeros(num_ratings)

    for rating in range(min_rating, max_rating + 1):
        hist_true[rating - min_rating] = np.sum(y_true == rating)
        hist_pred[rating - min_rating] = np.sum(y_pred == rating)

    weight_matrix = np.zeros((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            weight_matrix[i][j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2)

    conf_matrix = np.zeros((num_ratings, num_ratings))
    for true_rating, pred_rating in zip(y_true, y_pred):
        conf_matrix[true_rating - min_rating][pred_rating - min_rating] += 1

    expected_matrix = np.outer(hist_true, hist_pred) / len(y_true)
    kappa = 1 - (np.sum(weight_matrix * conf_matrix) / np.sum(weight_matrix * expected_matrix))

    return kappa


# Calculate QWK for each judge
for i, judge in enumerate(judges):
    judge_scores = data[judge]

    # Ensure scores are integers for QWK calculation
    judge_scores = judge_scores.round().astype(int)
    model_scores_int = model_scores.round().astype(int)

    qwk = quadratic_weighted_kappa(judge_scores, model_scores_int)
    results[i]['QWK'] = round(qwk, 2)

# Create a DataFrame from results
results_df = pd.DataFrame(results)

# Save the results to an Excel file
output_file = 'model_judge_consistency_analysis_saber.xlsx'
results_df.to_excel(output_file, index=False)

print(f"Results saved to {output_file}")
