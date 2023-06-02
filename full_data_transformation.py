import pandas as pd
from tqdm import tqdm


def remove_lists(value):
    return value[0] if value else 0


def txt_to_parquet(batch_size, num_rows=22932488):
    whole_df_list = []
    with open('data/output_v2.txt') as input_file:
        for _ in tqdm(range(num_rows // batch_size)):
            rows = ''
            for j in range(batch_size):
                rows += next(input_file)
            df = pd.read_json(rows, lines=True)

            columns_df = pd.DataFrame(df['features'].tolist())
            result = pd.DataFrame(columns_df[0].tolist(),
                                  columns=[f'0_{i}' for i in range(9)])

            for i in range(1, 11):
                result = result.join(pd.DataFrame(
                    columns_df[i].tolist(),
                    columns=[f'{i}_{j}' for j in range(9)]))

            result = result.applymap(remove_lists)

            result = result.join(df['category'])
            result.rename(columns={'category': 'move_id'}, inplace=True)
            dtypes_dict = {}
            for col in result.columns.tolist():
                if col == 'move_id':
                    dtypes_dict[col] = 'int16'
                else:
                    dtypes_dict[col] = 'int8'
            whole_df_list.append(result.astype(dtypes_dict))
    whole_df = pd.concat(whole_df_list).reset_index(drop=True)
    whole_df.to_parquet(f'data/output_v2_{num_rows}.parquet')


if __name__ == '__main__':
    txt_to_parquet(1000)
