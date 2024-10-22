# Build a voting selector
import warnings
import pandas as pd

from logger import logger
from feature_selection.config_entity import DataIngestionConfig, MLFeatureSelectorConfig
from feature_selection.components.data_transformation import DataProcessing
from feature_selection.components.feature_selection_methods import BestFeatureSelector


warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'


class GetBestMLFeatures:
    def __init__(self, diconfig: DataIngestionConfig, mlfsconfig: MLFeatureSelectorConfig):
        self.diconfig = diconfig
        self.mlfsconfig = mlfsconfig

        self.data_file = self.diconfig.data_file
        self.supervised_type = self.mlfsconfig.supervised_type
        self.label_id_columns = self.mlfsconfig.label_id_columns
        self.label_name_columns = self.mlfsconfig.label_name_columns
        self.drop_columns = self.mlfsconfig.drop_columns
        self.feature_selection_option = self.mlfsconfig.feature_selection_option
        self.n_feature_to_select = self.mlfsconfig.n_feature_to_select

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
            best_features = frq_count['feature'].tolist()

        return best_features

    def find_best_features(self):
        kwargs = {
            'supervised_type': self.supervised_type,
            'data_path': self.data_file,
            'drop_columns': self.drop_columns,
            'feature_selection_option': self.feature_selection_option,
            'n_feature_to_select': self.n_feature_to_select
        }
        df = pd.read_csv(kwargs.get('data_path'))
        dp = DataProcessing()
        X, y = dp.processing(df, self.label_id_columns,
                             self.label_name_columns, drop_columns=kwargs.get('drop_columns'))
        bf = BestFeatureSelector()
        mdf = bf.select(X, y, **kwargs)
        logger.info(f"Completed running best feature selector methods")

        final_features = {}
        for selection_option in kwargs.get('feature_selection_option'):
            logger.info(f"Finding best feature with: {selection_option}")
            best_features = self.selection_criteria(
                mdf, selection_option=selection_option, n_feature=kwargs.get('n_feature_to_select'))
            final_features[selection_option] = best_features

        return final_features
