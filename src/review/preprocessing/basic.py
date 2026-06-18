
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def count_predictors(df, ):

    pred_count = []
    for category, group in df[-df['nested']].groupby('predictors.category'):
        scales = []
        for _, g in group.groupby('model_id'):
            preds = list(set([p for p in g['predictors.scale'] if p != 'nan' and isinstance(p, str)]))
            scales.extend(preds)
        scales = pd.Series(scales)
        scales_df = pd.DataFrame(scales, columns=['scale'])
        scales_df = scales_df.groupby('scale').size().reset_index(name='count')
        scales_df['category'] = category
        pred_count.append(scales_df)
        print(f"\n\nPredictors for {category}:\n {scales.value_counts().index}")

    pred_count = pd.concat(pred_count)
    pred_count = pred_count.sort_values(by=['category', 'count'], ascending=False).reset_index(drop=True)

    unique_predictors = pred_count['scale'].unique().tolist()

    return pred_count, unique_predictors


def from_long_to_wide(data):
    # Convert from long format to wide format for predictors
    predictors = []
    for model_id, group in data.groupby('model_id'):
        info = {k: ', '.join(str(value) for value in v.tolist()) for k, v in group.loc[:, 'predictors.name': 'predictors.acquisition_time'].items()}
        info['model_id'] = model_id
        info['notes'] = group['notes'].unique()[0]
        predictors.append(info)
    # create a dataframe from the list of dictionaries
    predictors_df = pd.DataFrame(predictors)
    data = data.drop(columns=data.loc[:, 'predictors.name': 'predictors.acquisition_time'].columns).drop(columns='notes')
    data = data.drop_duplicates()
    data['duplicate_model'] = data['model_id'].duplicated(keep=False)
    check_duplicated = data[data['duplicate_model']].copy()
    if not check_duplicated.empty:
        logger.error("Duplicated models found:")
        logger.error(check_duplicated[['Key', 'first_author_year', 'model_id', 'duplicate_model']])
        raise ValueError

    # merge predictors_df with data on model_id
    data = pd.merge(data, predictors_df, on='model_id', how='inner')

    return data