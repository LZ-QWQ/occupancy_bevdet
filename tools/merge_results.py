import numpy as np
import os    
import sys
from concurrent.futures import ProcessPoolExecutor,as_completed

path = '/media/data3/caiwb/BEVDet/results'
merge_path = os.path.join(path, 'merge')

results = os.listdir(merge_path)

submission_prefix =os.path.join(path, 'occ_submission') 
if not os.path.exists(submission_prefix):
    os.mkdir(submission_prefix)

files = os.listdir( os.path.join(merge_path, results[0]) )

num_class = 18


def thread_merge(file):
    sum = np.zeros((200, 200, 16, num_class))
    for result in results:
        label_path = os.path.join(merge_path, result, file)
        res = np.load(label_path)
        res = res['arr_0']
        res = np.eye(num_class)[res]
        sum += res
    sum = np.argmax(sum, -1)
    return sum, file

def get_result(future):
    sum, file = future.result()
    save_path=os.path.join(submission_prefix,'{}'.format(file))
    np.savez_compressed(save_path, sum.astype(np.uint8))

executor = ProcessPoolExecutor(40)
future_list = []    

for i, file in enumerate(files)  :
    future_list.append(executor.submit(thread_merge, file))

for future in as_completed(future_list):
    future.add_done_callback(get_result) 
