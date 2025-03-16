# Build a voting selector
import warnings
import pandas as pd

from logger import logger
from feature_selection.data_transformation import DataProcessing
from feature_selection.feature_selection_methods import BestFeatureSelector


warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'


class GetBestMLFeatures:
    def __init__(self):
        pass

    def selection_criteria(self, mdf, selection_option=None, n_feature=None):
        if selection_option == 'aggregate':
            # method-1: aggregated sum on ranks
            mdf.reset_index(drop=True, inplace=True)
            agg_sum = mdf.groupby(['feature'])['ranks'].sum()
            agg_sum.sort_values(ascending=False, inplace=True)
            agg_sum = agg_sum.to_frame()
            agg_sum.reset_index(drop=False, inplace=True)
            agg_sum = agg_sum.head(n_feature)
            best_features = agg_sum['feature'].tolist()
        else:
            # method-2: frequency/voting count of features
            frq_count = mdf['feature'].value_counts()
            frq_count = frq_count.to_frame()
            frq_count.reset_index(drop=False, inplace=True)
            frq_count = frq_count.head(n_feature)
            # best_features = frq_count['feature'].tolist()
            best_features = frq_count['index'].tolist()

        return best_features

    def find_best_features(self, **kwargs):
        use_columns = kwargs.get('selected_columns')
        label_id_column = kwargs.get('label_id_column')
        label_name_colum = kwargs.get('label_name_column')
        drop_columns = kwargs.get('drop_columns')
        n_feature = kwargs.get('n_feature_to_select')
        df = pd.read_csv(kwargs.get('data_path'))
        df = df[use_columns]
        dp = DataProcessing()
        X, y = dp.processing(df, label_id_column, label_name_colum, drop_columns=drop_columns)
        bf = BestFeatureSelector()
        mdf = bf.select(X, y, **kwargs)
        logger.info(f"Completed running best feature selector methods")

        final_features = {}
        for selection_option in kwargs.get('feature_selection_option'):
            logger.info(f"Finding best feature with: {selection_option}")
            best_features = self.selection_criteria(mdf, selection_option=selection_option, n_feature=n_feature)
            final_features[selection_option] = best_features

        return final_features


if __name__ == '__main__':
    args = {
        "selected_columns": '',
        "label_id_column": '',
        "label_name_column": '',
        "drop_columns": '',
        "n_feature_to_select": 5,
        "data_path": '',
        "feature_selection_option": ["aggregate", "voting"],
        "supervised_type": "Classification"
    }
    ml_feat = GetBestMLFeatures()
    final_features = ml_feat.find_best_features(**args)
    logger.info(f'Best features: {final_features}')
