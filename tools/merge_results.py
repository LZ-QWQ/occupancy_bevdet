import numpy as np
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

parser = argparse.ArgumentParser(description="merge results")
parser.add_argument("--merge_sub_path", type=str, help="加载子路径")
parser.add_argument("--submission_prefix", type=str, help="保存子路径")
args = parser.parse_args()

path = "/media/data4/lizheng/SurroundOcc/BEVDet/results"
merge_path = os.path.join(path, args.merge_sub_path)  # merge_logit

results = os.listdir(merge_path)

submission_prefix = os.path.join(path, args.submission_prefix)
os.makedirs(submission_prefix, exist_ok=True)

files = os.listdir(os.path.join(merge_path, results[0]))

num_class = 18


def thread_merge(file):
    sum = np.zeros((200, 200, 16, num_class)).astype("int64")
    for result in results:
        label_path = os.path.join(merge_path, result, file)
        res = np.load(label_path)
        res = res["arr_0"]
        # if res.shape == (200, 200, 16):
        #     res = res.astype('uint8')
        #     res = np.eye(num_class)[res].astype('uint8')
        sum += res
    sum = np.argmax(sum, -1)
    return sum, file


def get_result(future):
    sum, file = future.result()
    save_path = os.path.join(submission_prefix, "{}".format(file))
    np.savez_compressed(save_path, sum.astype(np.uint8))
    print(file)


executor = ProcessPoolExecutor(10)
future_list = []

for i, file in enumerate(files):
    future_list.append(executor.submit(thread_merge, file))

for future in as_completed(future_list):
    future.add_done_callback(get_result)
