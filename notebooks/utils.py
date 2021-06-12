from itertools import chain
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer


FIXED_COLUMN_NAMES = [
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


def get_feature_columns(dataframe: pd.DataFrame, remove_id_like_columns: bool=False) -> List[str]:
    all_cols = set(dataframe.columns)
    fixed_cols = set(FIXED_COLUMN_NAMES)
    id_like_cols = set(ICE_CAT_ID_LIKE_COLUMN_NAMES) if remove_id_like_columns else set()
    disjoint_set = all_cols - fixed_cols - id_like_cols
    return list(sorted(disjoint_set))


def get_unique_value_lengths(dataframe: pd.DataFrame, col_name: str):
    unique_vals = map(lambda r: str(r), dataframe[col_name].unique())
    unique_vals = filter(lambda r: r != 'nan', unique_vals)
    unique_vals = map(lambda r: len(r), unique_vals)
    unique_vals = list(unique_vals)
    return np.array(unique_vals)


def split_and_flatten(arr: list) -> List[str]:
    arr_splitted = [str(v).split(',') for v in arr]
    arr_flattened = [v.strip() for v in chain.from_iterable(arr_splitted)]
    return arr_flattened


def inferred_type(dataframe: pd.DataFrame, col_name: str, max_cat_value_count: int=1000) -> np.dtype:
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
    
    if n_unique >= 2 and n_unique < max_cat_value_count:
        unique_val_lengths = get_unique_value_lengths(dataframe, col_name)
        if np.max(unique_val_lengths) > 300:
            # print(f'{col_name} -> {np.max(unique_val_lengths)}')
            return 'object'

        # Detect multi-label categories
        unique_vals_with_seperator = [str(v) for v in unique_vals if ',' in str(v)]
        unique_vals_no_sep = list(set(unique_vals) - set(unique_vals_with_seperator))
        unique_vals_multi = list(set(split_and_flatten(unique_vals_with_seperator)))
        if len(set(unique_vals_no_sep).intersection(set(unique_vals_multi))) > 1:
            return 'multi-category'

        return 'category'

    return 'object'


def filter_rows(dataframe: pd.DataFrame, min_count_per_category: int=50):
    # Count number of rows per category
    df_count_by_category = dataframe.groupby('category_name').agg({'id': 'count'}).rename(columns={'id': 'n_rows'})

    # Find categories with at least N amount of rows
    categories = list(df_count_by_category[df_count_by_category.n_rows > min_count_per_category].index)

    # Delete rows with few that N amount of rows per category
    return dataframe[dataframe.category_name.isin(categories)]


def get_product_feature_columns_for_training(dataframe: pd.DataFrame, min_count_per_feature:int=10):
    # Get columns that specify features of the products
    product_feature_columns = get_feature_columns(dataframe, remove_id_like_columns=True)

    # Find columns that have too few specified values
    n_rows = dataframe.shape[0]
    excluded_cols = []
    for col in product_feature_columns:
        n_filled = n_rows - dataframe[col].isna().sum()
        if n_filled < min_count_per_feature:
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
    ]

    # Find columns that have enough values
    product_features_to_use = [
        col 
        for col in product_feature_columns
        if col not in excluded_cols
    ]

    return product_features_to_use


def detect_and_fix_column_types(dataframe: pd.DataFrame):
    df_cleaned = dataframe.copy()
    # Use proper dtypes
    for col in df_cleaned.columns:
        dtype = inferred_type(df_cleaned, col)
        if dtype == 'int32':
            df_cleaned[col].fillna(0, inplace=True)
        elif dtype == 'float':
            df_cleaned[col].fillna(0.0, inplace=True)
        elif dtype == 'bool':
            df_cleaned[col].fillna('N', inplace=True)
            df_cleaned[col] = df_cleaned[col].str.replace('N', '0')
            df_cleaned[col] = df_cleaned[col].str.replace('Y', '1')
            df_cleaned[col] = df_cleaned[col].astype('int')
        elif dtype == 'category':
            df_cleaned[col].fillna('(N/A)', inplace=True)
        df_cleaned[col] = df_cleaned[col].astype(dtype)
    return df_cleaned


def split_train_test(df):
    train = df.sample(frac=.8, random_state=42)
    test = df.loc[~df.index.isin(train.index)]
    return train, test


def compute_column_stats(dataframe: pd.DataFrame):
    dmap = {
        'column': [],
        'suggested_type': [],
        'n_unique': [],
        'len_total': [],
        'len_min': [],
        'len_max': [],
        'len_avg': [],
        'values': [],
        'n_filled': [],
    }

    n_rows = dataframe.shape[0]

    for col in dataframe.columns:
        dmap['column'].append(col)
        dmap['suggested_type'].append(inferred_type(dataframe, col))

        dmap['n_filled'].append( n_rows - dataframe[col].isna().sum() )

        dmap['n_unique'].append(dataframe[col].unique().shape[0])
        unique_val_lengths = get_unique_value_lengths(dataframe, col)
        dmap['len_total'].append(len(unique_val_lengths))
        dmap['len_min'].append(np.min(unique_val_lengths))
        dmap['len_max'].append(np.max(unique_val_lengths))
        dmap['len_avg'].append(np.mean(unique_val_lengths))

        vals = ' | '.join([str(s) for s in list(dataframe[col].unique())[0:5]])
        dmap['values'].append(vals)

    return pd.DataFrame(dmap)


def create_variance_scree_plot(variance_ratios):
    n_items = len(variance_ratios)
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(np.arange(1, n_items + 1), np.cumsum(variance_ratios))
    ax.bar(np.arange(1, n_items + 1), variance_ratios)
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance');


def find_top_n_categories(dataframe: pd.DataFrame, top_n: int=10):
    return list(dataframe
        .groupby(['category_name'])
        .count()[['id']]
        .sort_values('id', ascending=False)
        .head(top_n)
        .index
    )


def get_fill_ratios_for_product_features_per_category(dataframe: pd.DataFrame):
    # Only consider product features that can be used for comparing distances between products.
    col_product_features = get_feature_columns(dataframe, remove_id_like_columns=True)

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


def get_valid_categories_per_product_feature(fill_ratio: Dict[str, Dict[str, float]]):
    valid_cats_per_feat = dict()
    for category, feats_map in fill_ratio.items():
        for feat in feats_map.keys():
            if not feat in valid_cats_per_feat:
                valid_cats_per_feat[feat] = []
            valid_cats_per_feat[feat].append(category)
    return valid_cats_per_feat


def prepare_for_preprocessing(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    df = dataframe.copy()
    fill_ratios = get_fill_ratios_for_product_features_per_category(df)
    valid_cats_per_feat = get_valid_categories_per_product_feature(fill_ratios)
    product_feature_columns = get_product_feature_columns_for_training(df)

    feature_type_map = {
        'binary': [],
        'category': [],
        'multi-category': [],
        'int32': [],
        'float': [],
    }
    
    for feat in set(product_feature_columns):
        valid_product_categories = valid_cats_per_feat[feat]
        dtype = inferred_type(df, feat)
        filter_cat_not_valid = ~df.category_name.isin(valid_product_categories)
        
        if dtype in ['category', 'multi-category']:
            # Distinguish between values for which the feature is invalid or N/A.
            # Mark the categories for which the given product feature is not valid for as Invalid.
            # If the category is valid, mark NULL values as Not Available. 
            df.loc[filter_cat_not_valid & df[feat].isnull(), feat] = '(Invalid)'
            df.loc[df[feat].isnull(), feat] = '(N/A)'
            df[feat] = df[feat].astype('category' if dtype == 'category' else 'str') 
            feature_type_map[dtype].append(feat)
        
        elif dtype in ['float', 'int32']:
            if (filter_cat_not_valid & df[feat].isnull()).sum() > 0:
                # Create a new binary feature that tracks that the current numerical
                # product feature is invalid for certain product categories.
                feat_invalid = feat + ' Invalid'  # Name of the new feature 
                df[feat_invalid] = 0  # Default value is False
                df[feat_invalid] = df[feat_invalid].astype(int)

                # Ensure that the newly created feature is used.
                feature_type_map['binary'].append(feat_invalid)

                # Mark invalid if the product feature is not valid for the
                # categories and value is NULL.
                df.loc[filter_cat_not_valid & df[feat].isnull(), feat_invalid] = 1

            if df[feat].isnull().sum() > 0:
                # Create a binary feature that tracks when the current numerical
                # feature is valid for certain product categories but value is
                # not available for a given product.
                feat_na = feat + ' Not Available'  # Name of the new feature 
                df[feat_na] = 0  # Default Value
                df[feat_na] = df[feat_na].astype(int)

                # Ensure that the newly created feature is used downstream.
                feature_type_map['binary'].append(feat_na)

                # Mark valid but unassigned values.
                df.loc[df[feat].isnull(), feat_na] = 1

            df[feat].fillna(0.0, inplace=True)
            df[feat] = df[feat].astype(dtype)
            feature_type_map[dtype].append(feat)
                
    sorted_cols = list(sorted(df.columns.tolist()))
    return df[sorted_cols], feature_type_map
