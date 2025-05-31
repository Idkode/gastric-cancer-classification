import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import csv

all_data = {}

for folder in os.listdir(base_path := '25954813/HMU-GC-HE-30K/all_image/'):
    images = os.listdir(folder_path := (base_path + folder + '/'))
    all_data[folder] = [os.path.join(folder_path, item) for item in images]


with open('data.csv', mode='r+') as f:
    writer = csv.writer(f, delimiter=',', lineterminator='\n')
    writer.writerow(['path', 'label'])
    for key in all_data.keys():
        for image in all_data[key]:
            writer.writerow([image, key])


data = pd.read_csv('data.csv', sep=',')

X_temp, X_test, y_temp, y_test = train_test_split(
                                            data['path'].values,
                                            data['label'].values,
                                            stratify=data['label'].values,
                                            random_state=29,
                                            test_size=0.2
                                            )

X_train, X_val, y_train, y_val = train_test_split(
                                            X_temp,
                                            y_temp,
                                            stratify=y_temp,
                                            random_state=11,
                                            test_size=0.1
                                            )


train = pd.DataFrame({'path': X_train, 'label': y_train})
val = pd.DataFrame({'path': X_val, 'label': y_val})
test = pd.DataFrame({'path': X_test, 'label': y_test})

train.to_csv('train.csv', index=False, lineterminator='\n')
val.to_csv('validation.csv', index=False, lineterminator='\n')
test.to_csv('test.csv', index=False, lineterminator='\n')





