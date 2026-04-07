import os
import json
import pandas as pd
import sys

def main(folder_path):
    json_directory = f'./{folder_path}'
    csv_directory = f'./{folder_path}_csv'
    merged_csv_path = f'./{csv_directory}/merged_output.csv'

    fields = ['date',
              'model_id',
              'num_prompts',
              'request_rate',
              'duration',
              'completed',
              'total_input_tokens',
              'total_output_tokens',
              'output_throughput',
              #'median_ttft_ms',
              'p50_ttft_ms',
              'p95_ttft_ms',
              #'median_tpot_ms',
              'p50_tpot_ms',
              'p95_tpot_ms',
              #'median_itl_ms',
              'p50_itl_ms',
              'p95_itl_ms',
              #'median_e2el_ms',
              'p50_e2el_ms',
              'p95_e2el_ms'
              ]
    os.makedirs(csv_directory, exist_ok=True)

    csv_files=[]
    for filename in os.listdir(json_directory):
        if filename.endswith('.json'):
            json_path = os.path.join(json_directory, filename)
            with open(json_path, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                    filtered_data = {field: data.get(field, None) for field in fields}
                    df = pd.DataFrame([filtered_data])
                    csv_filename = f"{os.path.splitext(filename)[0]}.csv"
                    csv_path = os.path.join(csv_directory, csv_filename)
                    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                    csv_files.append(csv_path)
                except json.JSONDecodeError:
                    print("error")
    merged_df = pd.concat([pd.read_csv(csv_file) for csv_file in csv_files], ignore_index=True)
    merged_df_sorted = merged_df.sort_values(by='date')
    merged_df_sorted.to_csv(merged_csv_path, index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    main(sys.argv[1])

