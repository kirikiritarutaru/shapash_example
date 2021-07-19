import pandas as pd
from category_encoders import OrdinalEncoder
from shapash.data.data_loader import data_loading
from shapash.explainer.smart_explainer import SmartExplainer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def main():
    house_df, house_dict = data_loading('house_prices')
    print(house_df.head(3))
    print(f'house_dict: {house_dict}')

    y_df = house_df['SalePrice'].to_frame()
    X_df = house_df[house_df.columns.difference(['SalePrice'])]
    categorical_features = [
        col for col in X_df.columns if X_df[col].dtype == 'object'
    ]
    encoder = OrdinalEncoder(cols=categorical_features).fit(X_df)
    X_df = encoder.transform(X_df)

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X_df,
        y_df,
        train_size=0.75
    )
    reg = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2
    ).fit(Xtrain, ytrain)

    y_pred = pd.DataFrame(
        reg.predict(Xtest),
        columns=['pred'],
        index=Xtest.index
    )

    # Initialize a SmartExplainer Object
    # features_dict: DataFrameの各列の説明を指定するdict
    xpl = SmartExplainer(features_dict=house_dict)

    # Compile
    xpl.compile(
        x=Xtest,
        model=reg,
        preprocessing=encoder,
        y_pred=y_pred
    )

    app = xpl.run_app()
    # 記載のurlを開く


if __name__ == '__main__':
    main()
