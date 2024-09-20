res = '=AVERAGE('

for i in range(80):
    res += f'E{19+i*5},'

print(res)