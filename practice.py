'''
Input: 'abc/defgh$ij'
Output: 'jih/gfedc$ba'
'''

n_ = "abc/defgh$ij"
n = list(n_)

i = 0
j = len(n) - 1

while i < j:
    if not n[i].isalpha():
        i += 1
    elif not n[j].isalpha():
        j -= 1
    else:
        n[i], n[j] = n[j], n[i]
        i +=1
        j -= 1

print("Input string: ",n_)
print("Output string: ","".join(n))
