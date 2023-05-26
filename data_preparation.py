import pandas as pd
import time
import os

pd.options.display.max_columns = 120
pd.options.display.max_rows = 100


def get_n_rows_in_nwe_file(n_rows):
    with open('data/output_v2.txt') as input_file:
        with open(f'data/output_v2_{n_rows}_rows.txt', 'w') as output:
            for i in range(n_rows):
                output.write(next(input_file))


def remove_lists(value):
    return value[0] if value else 100


def txt_to_parquet(n_rows):

    if not os.path.exists(f'data/output_v2_{n_rows}_rows.txt'):
        start_time = time.time_ns()
        get_n_rows_in_nwe_file(n_rows)
        print(f'Smaller file generation: '
              f'{(time.time_ns() - start_time) / 100000000} second(s).')

    start_time = time.time_ns()
    df = pd.read_json(f'data/output_v2_{n_rows}_rows.txt', lines=True)
    print(f'File reading: '
          f'{(time.time_ns() - start_time) / 100000000} second(s).')

    start_time = time.time_ns()
    columns_df = pd.DataFrame(df['features'].tolist())
    result = pd.DataFrame(columns_df[0].tolist(),
                          columns=[f'0_{i}' for i in range(9)])
    print(f'Columns and first row explode: '
          f'{(time.time_ns() - start_time) / 100000000} second(s).')

    start_time = time.time_ns()
    for i in range(1, 11):
        result = result.join(pd.DataFrame(
            columns_df[i].tolist(), columns=[f'{i}_{j}' for j in range(9)]))
    print(f'Rows explode: '
          f'{(time.time_ns() - start_time) / 100000000} second(s).')

    start_time = time.time_ns()
    result = result.applymap(remove_lists)
    print(f'Lists removing: '
          f'{(time.time_ns() - start_time) / 100000000} second(s).')

    start_time = time.time_ns()
    result = result.join(df['category'])
    result.rename(columns={'category': 'move_id'}, inplace=True)
    dtypes_dict = {}
    for col in result.columns.tolist():
        if col == 'move_id':
            dtypes_dict[col] = 'int16'
        else:
            dtypes_dict[col] = 'int8'
    result = result.astype(dtypes_dict)
    result.to_parquet(f'data/output_v2_{n_rows}_rows.parquet')
    print(f'Minor transforms and saving: '
          f'{(time.time_ns() - start_time) / 100000000} second(s).')


if __name__ == '__main__':
    start_time = time.time_ns()
    txt_to_parquet(500000)
    print(f'Total time: {(time.time_ns() - start_time) / 100000000}'
          f' second(s).')
