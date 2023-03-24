import json

f = open("./hand_v2_no_bg/transforms.json", "r")

data = json.load(f)

#print(data)


dataA = data


train = []
val = []
test = []
for i in range(53):
    #print(i, data["frames"][i]['file_path'])
    if i!=0:
        train.append(data["frames"][i])
    if i==0:
        val.append(data["frames"][i])
        test.append(data["frames"][i])

#print(data["frames"][20]['file_path'])

dataA["frames"] = train
d = json.dumps(dataA)
with open("transforms_train.json", "w") as out:
    out.write(d)

dataA["frames"] = val
d = json.dumps(dataA)
with open("transforms_val.json", "w") as out:
    out.write(d)

dataA["frames"] = test
d = json.dumps(dataA)
with open("transforms_test.json", "w") as out:
    out.write(d)
