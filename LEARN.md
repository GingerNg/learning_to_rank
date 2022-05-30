ModuleNotFoundError: No module named 'sklearn.externals.six'
Module sklearn.externals.six was removed in the scikit-learn version 0.23. To use it you have to downgrade to version 0.22


https://weirping.github.io/blog/xgboost-rank-in-sklearn.html
https://github.com/Weirping/xgbranker_sklearn

groups = train_data.groupby('id').size().to_frame('size')['size'].to_numpy()

