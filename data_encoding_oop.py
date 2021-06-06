from sklearn.datasets import load_iris
from sklearn import metrics,preprocessing
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

iris = load_iris()

x = iris['data']
y = iris['target']

col_name = iris['feature_names']

x_df = pd.DataFrame(x, columns = col_name)
y_df = pd.DataFrame(y,columns= ['target'])

iris_df = pd.concat([x_df,y_df],axis=1)

print(x_df.head(10))
print(x_df.info())
print(y_df)
print(iris_df.head(4))


class lucas_preprocess:
  def cont(cont_cols):
    min_max_scaler = preprocessing.MinMaxScaler()
    cont_cols_result = min_max_scaler.fit_transform(cont_cols)
    print(cont_cols_result)
  def cate(cate_cols):
    enc = OneHotEncoder(handle_unknown = 'ignore')
    enc.fit(cate_cols)
    cate_cols_result = enc.transform(cate_cols).toarray()
    print(cate_cols_result)
    
lucas_preprocess.cont(x_df)
lucas_preprocess.cate(y_df)
