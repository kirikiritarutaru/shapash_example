import pandas as pd
from category_encoders import OrdinalEncoder
from shapash.data.data_loader import data_loading
from shapash.explainer.smart_explainer import SmartExplainer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def main():
    house_df, house_dict = data_loading('house_prices')
    print_data = False
    if print_data:
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
    # urlを開く
    # (左上) Features Importance: 各特徴の重要度のプロット
    # (右下) Contribution plot: FIで選択した特徴の予測への(局所的な)寄与度合い
    # (右上) Selection Table: データのテーブル
    # (右下) Local plot: テーブルで選択したデータにおいて、どの特徴が予測値に最も貢献するかを表示

    # 終了する場合
    # app.kill()

    # 選択したデータの部分集合の特徴の重要度を表示
    disp_on_jupyter = False
    if disp_on_jupyter:
        subset = [168, 54, 995, 799, 310, 322, 1374,
                  1106, 232, 645, 1170, 1229, 703, 66,
                  886, 160, 191, 1183, 1037, 991, 482,
                  725, 410, 59, 28, 719, 337, 36]
        xpl.plot.features_importance(selection=subset)
        xpl.plot.contribution_plot("OverallQual")
        xpl.filter(max_contrib=8, threshold=100)
        xpl.plot.local_plot(index=560)
        xpl.filter(max_contrib=3, threshold=1000)
        summary_df = xpl.to_pandas()
        print(summary_df.head())
        xpl.plot.compare_plot(row_num=[0, 1, 2, 3, 4], max_features=8)


if __name__ == '__main__':
    main()
