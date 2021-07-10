import sys
import random
import simplekml
import pandas as pd

"""
How to use
```
python3 src/df2kml.py exp/exp023/exp023_submission.csv
```
"""
def main():
    path = sys.argv[1]
    print(path)
    df = pd.read_csv(path)
    names = df["phone"].unique()
    kml = simplekml.Kml(open=1)
    for name_ in names:
        temp  = df[df["phone"] == name_]
        temp_ = []
        for ele in temp[["lngDeg","latDeg"]].values.tolist():
            temp_.append(tuple(ele))
        linestring = kml.newlinestring(name = name_)
        linestring.style.linestyle.width = 30
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        linestring.style.linestyle.color = simplekml.Color.rgb(r,g,b)
        # linestring.style.linestyle.color = simplekml.ColorMode.random
        linestring.coords = temp_
    save_name = path.split('.')[0]
    kml.savekmz(f'{save_name}.kmz')
    print(f'save Done to {save_name}.kmz')


if __name__ =='__main__':
    main()