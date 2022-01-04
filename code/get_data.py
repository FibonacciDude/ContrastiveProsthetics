import numpy as np
from tqdm import trange

data = []

for i in trange(0, 368):
    dat = np.load('../data/logits_%d.npy'%i)
    data.append(data)

print(data.shape)
data = np.array(data)
print(data.shape)
labels = np.load('../data/logits_labels.npy')

print("Writing to excel")
writer = pd.ExcelWriter('../data/logits.xlsx', engine='xlsxwriter')
df = pd.DataFrame(data)
df.to_excel(writer, sheet_name='logits')

print("Writing labels")
df = pd.DataFrame(labels)
df.to_excel(writer, sheet_name='labels')

writer.save()

