import pandas as pd
import time

start_time = time.time_ns()
df = pd.read_parquet('data/output_v2_500000_rows.parquet')
print(f'Total loading time: {(time.time_ns() - start_time) / 100000000}'
      f' second(s).')
print(f'Dataframe memory usage: {df.memory_usage(deep=True).sum()/1000000}'
      f' Mbytes.')
