def get_processing_pipeline(variables, code = '111'):
    
    num_attribs_mtb = [var for var in variables if 'mtb' in var]
    num_attribs_nonmtb = [var for var in variables if ('mtb' not in var) and (var in var_num)]
    cat_attribs = [var for var in variables if var in var_cat + var_cat_ordinal]
    num_attribs_skewed_mtb = [var for var in var_num if var in mtbs_skewed]
    
    
    log_transformer = ColumnTransformer(
        [('log_transformer', FunctionTransformer(np.log1p, validate=True), num_attribs_skewed_mtb)], 
        remainder = 'passthrough'
    )    
    
    # Numerical pipeline
    num_pipeline_mtb = Pipeline(
        [
            ("selector", DataFrameSelector(num_attribs_mtb)),
            ("imputer", SimpleImputer(strategy="constant", fill_value = 1e-200, add_indicator=True)),
            ('log_transformation', log_transformer), 
            ("std_scaler", StandardScaler()),
        ]
    )
    
    num_pipeline_nonmtb = Pipeline(
        [
            ("selector", DataFrameSelector(num_attribs_nonmtb)),
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
#             ("KNNImputer", KNNImputer()),
            ("std_scaler", StandardScaler()),
        ]
    )
    
    # Categorical pipeline
    cat_pipeline = Pipeline(
        [
            ("selector", DataFrameSelector(cat_attribs)),
            ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
            ("ohe", OneHotEncoder(sparse=False, handle_unknown="ignore"))
        ]
    )        
    
    transformer_list=[
            ("num_pipeline_mtb", num_pipeline_mtb),
            ("num_pipeline_nonmtb", num_pipeline_nonmtb),
            ("cat_pipeline", cat_pipeline)
        ]
    
    transformer_list_cp = transformer_list.copy()
    for i, c in enumerate(code):
        if not int(c):
            transformer_list.remove(transformer_list_cp[i])
          
    data_prep_pipeline = FeatureUnion(
        transformer_list = transformer_list
    )
    
    return data_prep_pipeline