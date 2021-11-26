from dataclasses import dataclass
from itertools import chain
from typing import Dict, Generator, List

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer


ICE_CAT_IRRELEVANT_COLUMN_NAMES = [
    'id', 'supplier_id', 'category_id', 'category_name', 'title',
    'model_name', 'description_short', 'description_middle', 'description_long',
    'summary_short', 'summary_long', 'warranty', 'is_limited', 'on_market', 'quality',
    'url_details', 'url_manual', 'url_pdf', 'created_at', 'updated_at', 'released_on',
    'end_of_life_on', 'ean', 'n_variants', 'countries',
    # 'supplier_name',
]

ICE_CAT_ID_LIKE_COLUMN_NAMES = [
    'Customs product code (TARIC)',
    'Master (outer) case GTIN (EAN/UPC)',
    'Bundled software',
    'Shipping (inner) case GTIN (EAN/UPC)',
    'Dimensions (W x D x H) (imperial)',
    'Dimensions printer compartment (W x D x H)',
    'Package dimensions (W x D x H)',
    'Interior dimensions (W x D x H)',
    'Package dimensions (WxDxH)',
    'Dimensions (WxDxH)',
    'Exterior dimensions (WxDxH)',
    'Weight (imperial)',
    'Board size (W x D)',
    'Case or master carton dimensions (W x D x H)',
    'Pallet GTIN (EAN/UPC)',
    'Pallet dimensions (W x D x H)',
    'Pallet dimensions (W x D x H) (imperial)',
    'Pole dimensions (W x D x H)',
    'Slot size (LxWxH)',
]


class MultiHotEncoder(OneHotEncoder):
    def __init__(self, handle_unknown='ignore'):
        super().__init__(
            drop=None,
            sparse=False,
            dtype=np.float64,
            handle_unknown=handle_unknown
        )
        self.classes = []

    def fit(self, X, y=None):
        X_list, n_samples, n_features = self._check_X(X)
        self.classes = []
        for i in range(n_features):
            Xi = X_list[i]
            
            # Assume that each cell value type is a list -- containing one or more values
            cats = np.unique(list(chain.from_iterable(Xi)))
            
            self.classes.append(cats)
        return self
    
    def transform(self, X, y=None):
        X_list, n_samples, n_features = self._check_X(X)
        
        if n_features != len(self.classes):
            raise ValueError(
                "The number of features in X is different to the number of "
                "features of the fitted data. The fitted data had {} features "
                "and the X has {} features."
                .format(len(self.classes,), n_features)
            )
        
        output = []
        for i in range(n_features):
            Xi = X_list[i]
            classes_i = self.classes[i]
            binarizer = MultiLabelBinarizer(classes=classes_i)
            output.append(binarizer.fit_transform(Xi))
            
        return np.concatenate(output, axis=1)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)



class TypeInference:
    def __init__(self, max_cat_value_count: int=1000) -> None:
        self._max_cat_value_count = max_cat_value_count

    def run(self, dataframe: pd.DataFrame, col_name: str) -> np.dtype:
        is_datetime_col = dataframe[col_name].str.match('(\d{2,4}(-|\/|\\|\.| )\d{2}(-|\/|\\|\.| )\d{2,4})+').all()
        if is_datetime_col:
            return 'datetime'
        
        is_int32 = dataframe[col_name].str.match('\d{1,6}$').all()
        if is_int32:
            return 'int32'
        
        is_float = dataframe[col_name].str.match(r'\d{1,6}(\.\d{1,5})?$').all()
        if is_float:
            return 'float'
        
        unique_vals = dataframe[col_name].unique()
        n_unique = unique_vals.shape[0]

        if n_unique == 2 or n_unique == 3:
            bool_vals = np.array(['(N/A)', 'N', 'Y'], dtype='str')
            possible_bool_vals = np.array(pd.DataFrame(unique_vals).fillna('(N/A)')[0])
            if np.isin(possible_bool_vals, bool_vals).all():
                return 'bool'
        
        if n_unique >= 2 and n_unique < self._max_cat_value_count:
            unique_val_lengths = self._get_unique_value_lengths(dataframe, col_name)
            if np.max(unique_val_lengths) > 300:
                # print(f'{col_name} -> {np.max(unique_val_lengths)}')
                return 'object'

            # Detect multi-label categories
            unique_vals_with_seperator = [str(v) for v in unique_vals if ',' in str(v)]
            unique_vals_no_sep = list(set(unique_vals) - set(unique_vals_with_seperator))
            unique_vals_multi = list(set(self._split_and_flatten(unique_vals_with_seperator)))
            if len(set(unique_vals_no_sep).intersection(set(unique_vals_multi))) > 1:
                return 'multi-category'

            return 'category'

        return 'object'

    def _get_unique_value_lengths(self, dataframe: pd.DataFrame, col_name: str):
        unique_vals = map(lambda r: str(r), dataframe[col_name].unique())
        unique_vals = filter(lambda r: r != 'nan', unique_vals)
        unique_vals = map(lambda r: len(r), unique_vals)
        unique_vals = list(unique_vals)
        return np.array(unique_vals)


    def _split_and_flatten(self, arr: list) -> List[str]:
        arr_splitted = [str(v).split(',') for v in arr]
        arr_flattened = [v.strip() for v in chain.from_iterable(arr_splitted)]
        return arr_flattened


@dataclass
class FeatureInfo:
    name: str
    dtype: str
    synthentic: bool = False
    proxy_name: str = None

    @staticmethod
    def get_name_for_invalid_feature(name: str) -> str:
        return f"{name} Invalid"

    @staticmethod
    def get_name_for_not_available_feature(name: str) -> str:
        return f"{name} Not Available"


class FeatureCollection:
    def __init__(self) -> None:
        self._features: List[FeatureInfo] = []

    def add(self, name: str, dtype: str):
        """Add a feature to the collection."""
        self._features.append(
            FeatureInfo(name=name, dtype=dtype, synthentic=False)
        )
    
    def add_for_invalid(self, name: str) -> None:
        """Add a binary feature for indicating when a NaN value for a feature `name` is invalid for a given category."""
        self._features.append(
            FeatureInfo(
                name=FeatureInfo.get_name_for_invalid_feature(name),
                dtype="binary",
                synthentic=True,
                proxy_name=name
            )
        )
    
    def add_for_not_available(self, name: str) -> None:
        """Add a binary feature for indicating when a NaN value for a feature `name` is valid but not available."""
        self._features.append(
            FeatureInfo(
                name=FeatureInfo.get_name_for_not_available_feature(name),
                dtype="binary",
                synthentic=True,
                proxy_name=name
            )
        )

    @property
    def non_synthetic(self) -> Generator[FeatureInfo, None, None]:
        for feature in self._features:
            if not feature.synthentic:
                yield feature

    @property
    def synthetic(self) -> Generator[FeatureInfo, None, None]:
        for feature in self._features:
            if feature.synthentic:
                yield feature

    @property
    def multi_categorical(self) -> Generator[FeatureInfo, None, None]:
        for feature in self._features:
            if feature.dtype == "multi-category":
                yield feature

    @property
    def numeric(self) -> Generator[FeatureInfo, None, None]:
        for feature in self._features:
            if feature.dtype in ["int32", "flot"]:
                yield feature

    @property
    def categorical(self) -> Generator[FeatureInfo, None, None]:
        for feature in self._features:
            if feature.dtype in ["category"]:
                yield feature

    @property
    def binary(self) -> Generator[FeatureInfo, None, None]:
        for feature in self._features:
            if feature.dtype in ["binary"]:
                yield feature

class IceCatFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        self._min_count_per_feature = 10
        self._valid_feature_names = []
        self._infer_type = TypeInference(max_cat_value_count=1000).run
        self._features = FeatureCollection()

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()

        # Only consider product features that can be used for comparing distances between products.
        valid_column_names = self._get_feature_columns(X)

        fill_ratios = self._get_fill_ratios_for_product_features_per_category(
            dataframe=X,
            col_product_features=valid_column_names,
        )
        self._valid_cats_per_feat = self._get_valid_categories_per_product_feature(
            fill_ratio=fill_ratios
        )

        feature_names = self._filter_out_noisy_columns(
            dataframe=X, 
            valid_column_names=valid_column_names
        )

        self._features : FeatureCollection = self._create_feature_collection(
            dataframe=X,
            feature_names=feature_names
        )

        numeric_features = [f.name for f in self._features.numeric]
        categorical_features = [f.name for f in self._features.categorical]
        multi_categorical_features = [f.name for f in self._features.multi_categorical]
        binary_features = [f.name for f in self._features.binary]

        # Sanity check!
        all_features = numeric_features + categorical_features + multi_categorical_features + binary_features
        assert 'category_name' not in all_features, 'Product category variable should not be present!'

        # Create encoders/scalers for different types of features.
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        multi_categorical_transformer = MultiHotEncoder()

        # Combine all transformers into a preprocessor
        self._preprocessor = ColumnTransformer(
            transformers=[
                ('numerical', numeric_transformer, numeric_features),
                ('categorical', categorical_transformer, categorical_features),
                ('multi-categorical', multi_categorical_transformer, multi_categorical_features),
                ('binary', 'passthrough', binary_features)
            ]
        )

        # Clean up the values before call the preprocessing them.
        self._clean_data_values(df=X)

        self._preprocessor.fit(X)

        return self

    def transform(self, X: pd.DataFrame, y=None):
        # Clean up the values before call the preprocessing them.
        self._clean_data_values(df=X)

    def _create_feature_collection(self, dataframe: pd.DataFrame, feature_names: List[str]) -> FeatureCollection:
        features = FeatureCollection()
        for feat in set(feature_names):
            dtype = self._infer_type(dataframe, feat)
            features.add(name=feat, dtype=dtype)

            if dtype in ['float', 'int32']:
                # We want distinguish between NaN values for which the feature is invalid or N/A.
                # Therefore, we create two binary features to track what is what.
                # Invalid means that that particular feature is does not exist for a given product category.
                # N/A means that the feature is valid for a given category but no value is provided.
                # These two new features are filled out during the transform() step.
                features.add_for_invalid(feat)
                features.add_for_not_available(feat)

        return features

    def _fill_missing_values(self, df: pd.DataFrame):
        # Fill missing values for non-synthetic features
        for feature in self._features.non_synthetic:
            feat = feature.name
            dtype = feature.dtype

            valid_product_categories = self._valid_cats_per_feat[feature.name]
            filter_cat_not_valid = ~df.category_name.isin(valid_product_categories)
            
            if dtype in ['category', 'multi-category']:
                # Distinguish between values for which the feature is invalid or N/A.
                # Mark the categories for which the given product feature is not valid for as Invalid.
                # If the category is valid, mark NULL values as Not Available. 
                df.loc[filter_cat_not_valid & df[feat].isnull(), feat] = '(Invalid)'
                df.loc[df[feat].isnull(), feat] = '(N/A)'
                df[feat] = df[feat].astype('category' if dtype == 'category' else 'str') 

            elif dtype in ['float', 'int32']:
                if (filter_cat_not_valid & df[feat].isnull()).sum() > 0:
                    # Create a new binary feature that tracks that the current numerical
                    # product feature is invalid for certain product categories.
                    feat_invalid = FeatureInfo.get_name_for_invalid_feature(feat)  # Name of the new feature 
                    df[feat_invalid] = 0  # Default value is False
                    df[feat_invalid] = df[feat_invalid].astype(int)

                    # Mark invalid if the product feature is not valid for the
                    # categories and value is NULL.
                    df.loc[filter_cat_not_valid & df[feat].isnull(), feat_invalid] = 1

                if df[feat].isnull().sum() > 0:
                    # Create a binary feature that tracks when the current numerical
                    # feature is valid for certain product categories but value is
                    # not available for a given product.
                    feat_na = FeatureInfo.get_name_for_not_available_feature(feat)  # Name of the new feature 
                    df[feat_na] = 0  # Default Value
                    df[feat_na] = df[feat_na].astype(int)

                    # Mark valid but unassigned values.
                    df.loc[df[feat].isnull(), feat_na] = 1

                # NaN values for numerical features are set to 0.
                df[feat].fillna(0.0, inplace=True)
                df[feat] = df[feat].astype(dtype)

        # Fill missing values for synthetic features with NaN values
        for feature in self._features.synthetic:
            feat = feature.name
            dtype = feature.dtype

            if dtype == "binary":
                if feat in df.columns:
                    df[feat].fillna(0, inplace=True)
                    df[feat] = df[feat].astype(int)
                else:
                    df[feat] = 0
                    df[feat] = df[feat].astype(int)
            else:
                raise ValueError(f"Unable to fill missing values for synthetic features of dtype {dtype}")

    def _convert_multi_categorical_feature_values(self, df: pd.DataFrame):
        """Convert multi-categorical features to list of values as MultiHotEncoder expects it."""
        for f in self._features.multi_categorical:
            df[f.name] = df[f.name].astype('str').apply(lambda r: [v.strip() for v in r.split(',')])

    def _clean_data_values(self, df: pd.DataFrame):
        self._fill_missing_values(df=df)
        self._convert_multi_categorical_feature_values(df=df)

    def _filter_out_noisy_columns(self, dataframe: pd.DataFrame, valid_column_names: List[str]):
        # Find columns that have too few specified values
        n_rows = dataframe.shape[0]
        excluded_cols = []
        for col in valid_column_names:
            n_filled = n_rows - dataframe[col].isna().sum()
            if n_filled < self._min_count_per_feature:
                excluded_cols.append(col)
        
        # Manually exclude columns if they were not caught by the
        # heuristic above.
        excluded_cols += [
            'category_name',  # This should not be present in the X.
            'Customs product code (TARIC)',
            'Master (outer) case GTIN (EAN/UPC)',
            'Bundled software',
            'Shipping (inner) case GTIN (EAN/UPC)',
            'Dimensions (W x D x H) (imperial)',
            'Dimensions printer compartment (W x D x H)',
            'Package dimensions (W x D x H)',
            'Interior dimensions (W x D x H)',
            'Package dimensions (WxDxH)',
            'Dimensions (WxDxH)',
            'Exterior dimensions (WxDxH)',
            'Weight (imperial)',
            'supplier_name',
        ]

        # Find columns that have enough values
        product_features_to_use = [
            col 
            for col in valid_column_names
            if col not in excluded_cols
        ]

        return product_features_to_use


    def _get_feature_columns(self, dataframe: pd.DataFrame) -> List[str]:
        """Get the names of the columns that define the features of a product.
        """
        all_cols = set(dataframe.columns)
        fixed_cols = set(ICE_CAT_IRRELEVANT_COLUMN_NAMES)
        id_like_cols = set(ICE_CAT_ID_LIKE_COLUMN_NAMES)
        disjoint_set = all_cols - fixed_cols - id_like_cols
        return list(sorted(disjoint_set))
    
    def _get_fill_ratios_for_product_features_per_category(self, dataframe: pd.DataFrame, col_product_features: List[str]):
        # Find the number of filled values for each product feature and each product category.
        df_fill_per_category = dataframe[col_product_features].groupby(dataframe.category_name).count()
        df_fill_per_category = df_fill_per_category.transpose()

        # For each product feature, sum the number of filled values across categories.
        df_fill_per_category['total_filled'] = df_fill_per_category.sum(axis=1)

        # Fetch all categories.
        categories = dataframe.category_name.unique()

        # Prepare the result map
        category_feature_map = dict()

        for category in categories:
            # Find number of products in the given category
            n_rows_for_category = dataframe[dataframe.category_name == category].shape[0]

            # Find valid product features for the given category by filtering out
            # the product features that do not have any filled value.
            df_cat_features = df_fill_per_category[df_fill_per_category[category] > 0][category]
            category_feature_names = df_cat_features.index.tolist()
            
            # Compute fill ratio for each product feature in the given category.
            fill_ratios = (df_fill_per_category[df_fill_per_category[category] > 0][category] / n_rows_for_category).tolist()
            
            # Store the data in the map
            category_feature_map[category] = {k:v for k, v in zip(category_feature_names, fill_ratios)}

        return category_feature_map

    def _get_valid_categories_per_product_feature(self, fill_ratio: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
        valid_cats_per_feat = dict()
        for category, feats_map in fill_ratio.items():
            for feat in feats_map.keys():
                if not feat in valid_cats_per_feat:
                    valid_cats_per_feat[feat] = []
                valid_cats_per_feat[feat].append(category)
        return valid_cats_per_feat
