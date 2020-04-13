if __name__ == '__main__':
    f = 'human_batch_results.csv'
    import csv

    keys = ['all', 'coherent', 'gram', 'concise']
    sent = {'all': [], 'coherent': [], 'gram': [], 'concise': []}
    edu = {'all': [], 'coherent': [], 'gram': [], 'concise': []}
    ref = {'all': [], 'coherent': [], 'gram': [], 'concise': []}
    with open(f, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')

        for idx, row in enumerate(spamreader):
            if idx == 0:
                continue
            # print(row[-6:-2])
            # ['Input.l1', 'Input.l2', 'Input.l3', 'Answer.taskAnswers']
            l1, l2, l3 = row[-4], row[-3], row[-2]
            cat = [l1, l2, l3]
            ans = eval(row[-1])[0]
            # print(ans)
            # 123 all coherent gram coherent concise
            for i in range(1, 4):
                for k in keys:
                    n = "{}{}".format(i, k)
                    kk = cat[i - 1]
                    if kk == 'sent':
                        sent[k] = sent[k] + [ans[n]]
                    elif kk == 'edu':
                        edu[k] = edu[k] + [ans[n]]
                    else:
                        ref[k] = ref[k] + [ans[n]]
    from statistics import *

    draw, a, b, c = 0, 0, 0, 0
    lists = [sent['gram'], edu['gram'], ref['gram']]
    for x, y, z in zip(sent['gram'], edu['gram'], ref['gram']):
        max_val = max(x, y, z)
        min_val = min(x, y, z)
        if x == y and y == z:
            draw += 1
        elif x > y and x > z:
            a += 1
        elif y > x and y > z:
            b += 1
        elif z > x and z > y:
            c += 1
        elif min_val == x:
            b += 0.5
            c += 0.5
        elif min_val == y:
            a += 0.5
            c += 0.5
        elif min_val == z:
            a += 0.5
            b += 0.5
        else:
            raise NotImplementedError
    print(draw,a,b,c)
    for k, v in sent.items():
        print(k)
        print(sum(v) / len(v))
        print(stdev(v))
    for k, v in ref.items():
        print(k)
        print(sum(v) / len(v))
        print(stdev(v))
    for k, v in edu.items():
        print(k)
        print(sum(v) / len(v))
        print(stdev(v))
