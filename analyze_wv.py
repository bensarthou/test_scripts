import numpy as np
import glob
# print(glob.glob("/volatile/bsarthou/datas/wv_benchmark/loop_*"))

for f in glob.glob("/volatile/bsarthou/datas/wv_benchmark/loop_*"):
    dic = np.load(f).item()
    print(dic)
    print('\n')
