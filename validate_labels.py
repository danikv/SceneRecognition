import os
import re
import json


def fix_labels(label_file):
    data = "".join(label_file)
    data = data.rstrip()
    print(data)
    regex = re.compile(r"\:.?\"")
    new_string = regex.sub(lambda m : '{0}'.format(m.group(0).replace(':', ',')), data)
    new_string = new_string.rstrip()
    print(new_string)
    return new_string


def fix_empty_labels(label_file):
    data = "".join(label_file)
    data = data.rstrip()
    print(data)
    json_object = {}
    json_object['labels'] = []
    if data == '["labels": ]':
        return json.dumps(json_object)
    return data

dataset_folder = 'D:\Dataset'
labels_folder = os.path.join(dataset_folder, 'labels')

for (dirpath, dirnames, filenames) in os.walk(labels_folder):
    for file in filenames:
        with open(os.path.join(labels_folder, file), 'r+') as f:
            fixed_label = fix_empty_labels(f.readlines())
            f.seek(0)
            f.write(fixed_label)
            f.truncate()


# for (dirpath, dirnames, filenames) in os.walk(labels_folder):
#     for file in filenames:
#         if not file.endswith('.txt'):
#             old_file_name = os.path.join(labels_folder, file)
#             os.rename(old_file_name, old_file_name + '.txt')