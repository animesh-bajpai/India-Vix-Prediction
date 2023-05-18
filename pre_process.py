import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def read_data(filename):
    # Load the data from the CSV file
    df = pd.read_csv(filename)

    # Drop unnecessary columns and set datetime column as index
    df.set_index('Date', inplace = True)
    df.index = pd.to_datetime(df.index)
    df.drop(['High', 'Low', 'Change', 'Open'], axis = 1, inplace=True)

    # Shuffle the columns to get output variable as first column
    df = df[['Close', 'PrevClose', 'Pchange']]

    return df


def scale_split_data(dataFrame):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataFrame)
    train = scaled_data[:, 0]
    test = scaled_data[:, 1:]
    trainY, testY, trainX, testX = train_test_split(train, test, test_size=0.25, shuffle=False)
    trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
    testX = testX.reshape((testX.shape[0], 1, testX.shape[1]))

    return scaler, trainX, trainY, testX, testY