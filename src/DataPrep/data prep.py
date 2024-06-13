import os
import pandas as pd


def main():
    # Constants
    file_path = r"../../data/metadata/validated.tsv"
    data_dir = r"../../data/clips/"
    upvote_threshold = 3
    num_accents = 10
    additional_accents = []

    # get the data
    df = get_data(file_path)

    # filter the data
    filtered_df = filter_data(df, upvote_threshold, num_accents, additional_accents)

    # split the data
    train_df, val_df, test_df = split_data(filtered_df)

    print(f"Train size: {train_df.shape[0]} samples")
    print(f"Val size: {val_df.shape[0]} samples")
    print(f"Test size: {test_df.shape[0]} samples")

    # save the data
    train_df.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(data_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(data_dir, 'test.csv'), index=False)


def get_data(file_path):
    # get the data from the csv file
    df = pd.read_csv(file_path, sep='\t', usecols=['path', 'accents', 'up_votes', 'down_votes'])

    # drop rows with nan values
    non_nan_df = df.dropna()

    return non_nan_df


def filter_data(df, upvote_threshold=8, num_accents=10, additional_accents=None):
    if additional_accents is None:
        additional_accents = []

    # drop rows with accents other than the ones in top num_accents and additional accents
    top_accents = df['accents'].value_counts().head(num_accents).index
    top_df = df[df['accents'].isin(top_accents) | df['accents'].isin(additional_accents)]

    # drop rows with diffrence between upvotes and downvotes less than threshold
    top_df = top_df[(top_df['up_votes'] - top_df['down_votes']) >= upvote_threshold]

    # drop up_votes and down_votes columns and add new column 'accent' with values as integers
    top_df.drop(columns=['up_votes', 'down_votes'], inplace=True)
    top_df['accent'] = top_df['accents'].astype('category').cat.codes

    return top_df


def split_data(dataframe, train_size=0.9, val_size=0.3):
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()

    for accent in dataframe['accent'].unique():
        accent_df = dataframe[dataframe['accent'] == accent]

        accent_train_df = accent_df.sample(frac=train_size, random_state=42)
        accent_val_test_df = accent_df.drop(accent_train_df.index)
        accent_val_df = accent_val_test_df.sample(frac=val_size, random_state=42)
        accent_test_df = accent_val_test_df.drop(accent_val_df.index)

        train_df = pd.concat([train_df, accent_train_df])
        val_df = pd.concat([val_df, accent_val_df])
        test_df = pd.concat([test_df, accent_test_df])

    return train_df, val_df, test_df


if __name__ == '__main__':
    main()
