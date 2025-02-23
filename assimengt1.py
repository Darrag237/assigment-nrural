import random
e=2.71828182845904
in_1=.05
in_2=.10
bias_1=.5
bias_2=.7

weights = []

for _ in range(8):
    weights.append(random.uniform(-0.05, 0.5))
w1, w2, w3, w4, w5, w6, w7, w8 = weights

in_h1=(w1*in_1)+(w2*in_2)+bias_1
in_h2=(w3*in_1)+(w4*in_2)+bias_1

out_h1=((e**(in_h1))-(e**(-in_h1)))/((e**(in_h1))+(e**(-in_h1)))
out_h2=((e**(in_h2))-(e**(-in_h2)))/((e**(in_h2))+(e**(-in_h2)))


in_o1=(out_h1*w5)+(out_h2*w6)+bias_2
in_o2=(out_h1*w7)+(out_h2*w8)+bias_2

final_out_1=((e**(in_o1))-(e**(-in_o1)))/((e**(in_o1))+(e**(-in_o1)))
final_out_2=((e**(in_o2))-(e**(-in_o2)))/((e**(in_o2))+(e**(-in_o2)))


print("final_out_1"+str(final_out_1))
print("final_out_2"+str(final_out_2))