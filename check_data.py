import pandas as pd


if __name__ == '__main__':
    #data = pd.read_csv('/home/dante0shy/PycharmProjects/ToxicComments/data/train.csv',index_col='id')
    data = pd.read_csv('./data/train.csv', index_col='id')
    print(data.columns)
    print(data.head())
    pass
