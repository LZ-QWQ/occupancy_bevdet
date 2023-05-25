import numpy as np
class_num = {"others": 0.0,
"barrier": 1015729320.0,
"bicycle": 144966782.0,
"bus": 7079552577.0,
"car": 55893596988.0,
"construction_vehicle": 7199251990.0,
"motorcycle": 391169712.0,
"pedestrian": 7914083709.0,
"traffic_cone": 987786008.0,
"trailer": 19096465347.0,
"truck": 57271521750.0,
"driveable_surface": 1285277354548.0,
"other_flat": 39834698904.0,
"sidewalk": 441800360118.0,
"terrain": 618642874066.0,
"manmade": 2219788115205.0,
"vegetation": 2585882065216.0,
"free": 249794623977140.0}

num_list = np.array(list(class_num.values()))
print(num_list)
print(num_list / num_list[-1])
