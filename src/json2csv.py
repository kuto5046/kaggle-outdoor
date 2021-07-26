import pandas as pd
import json
import sys
import os
import glob
import numpy as np
from pandas.io.json import json_normalize

def main():
    output_dir = "../google-sdc-corrections/osr/csv/"
    os.makedirs(output_dir, exist_ok=True)

    file_paths = glob.glob("../google-sdc-corrections/osr/json/*.json")
    print(len(file_paths))
    for path in file_paths:
        df = pd.read_json(path, lines=True)
        df.drop(["payload"], axis=1, inplace=True)
        file_name = path.split("/")[-1].split(".")[0]

        df_json = json_normalize(df["common"], sep='_')
        df.to_csv(output_dir + f"{file_name}.csv", encoding='utf-8')
        break


if __name__ =='__main__':
    main()