import pandas as pd
import tensorflow as tf

if __name__ == '__main__':
    df = pd.read_parquet('data/output_v2_10000_rows.parquet')
    x_cols = df.columns.tolist()
    x_cols.remove('move_id')
    one_hot_df = (pd.get_dummies(df[x_cols].astype(str))
                  .astype('int8').join(df['move_id']))
    x_cols = one_hot_df.columns.tolist()
    x_cols.remove('move_id')

    n_features = one_hot_df.shape[1] - 1
    n_outputs = df['move_id'].nunique()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(n_features, activation='relu'),
        tf.keras.layers.Dense(n_features * 2, activation='relu'),
        tf.keras.layers.Dense(n_features, activation='relu'),
        tf.keras.layers.Dense(n_features / 2, activation='relu'),
        tf.keras.layers.Dense(n_outputs, activation='relu')])

    # When all labels will be in the dataset: remove 'sparse_' and get_dummies
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(one_hot_df[x_cols],
              pd.get_dummies(df['move_id'].astype(str)).astype('int8'),
              epochs=5)
