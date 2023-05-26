import pandas as pd
import time

if __name__ == '__main__':
    start_time = time.time_ns()
    df = pd.read_parquet('data/output_v2_500000_rows.parquet')
    print(f'Total loading time: {(time.time_ns() - start_time) / 100000000}'
          f' second(s).')
    print(f'Dataframe memory usage:'
          f' {df.memory_usage(deep=True).sum() / 1000000} Mbytes.')
