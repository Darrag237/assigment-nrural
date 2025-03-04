@ -0,0 +1,83 @@
# القيم الثابتة
e = 2.71828182845904
in_1 = 0.05
in_2 = 0.10
bias_1 = 0.35
bias_2 = 0.6
t_1 = 0.01
t_2 = 0.99
LR = 0.5

w1, w2, w3, w4 = 0.15, 0.2, 0.25, 0.3
w5, w6, w7, w8 = 0.4, 0.45, 0.5, 0.55

# Forward
in_h1 = (w1 * in_1) + (w2 * in_2) + bias_1
in_h2 = (w3 * in_1) + (w4 * in_2) + bias_1

out_h1 = ((e ** (in_h1)) - (e ** (-in_h1))) / ((e ** (in_h1)) + (e ** (-in_h1)))
out_h2 = ((e ** (in_h2)) - (e ** (-in_h2))) / ((e ** (in_h2)) + (e ** (-in_h2)))

in_o1 = (out_h1 * w5) + (out_h2 * w6) + bias_2
in_o2 = (out_h1 * w7) + (out_h2 * w8) + bias_2

out_1 = ((e ** (in_o1)) - (e ** (-in_o1))) / ((e ** (in_o1)) + (e ** (-in_o1)))
out_2 = ((e ** (in_o2)) - (e ** (-in_o2))) / ((e ** (in_o2)) + (e ** (-in_o2)))

error_o1 = 0.5 * ((t_1 - out_1) ** 2)
error_o2 = 0.5 * ((t_2 - out_2) ** 2)
error_total = error_o1 + error_o2

#  Backpropagation
E_total_w5 = (out_1 - t_1) * (out_1 * (1 - out_1)) * out_h1
E_total_w6 = (out_1 - t_1) * (out_1 * (1 - out_1)) * out_h2
E_total_w7 = (out_2 - t_2) * (out_2 * (1 - out_2)) * out_h1
E_total_w8 = (out_2 - t_2) * (out_2 * (1 - out_2)) * out_h2

n_w5 = w5 - LR * E_total_w5
n_w6 = w6 - LR * E_total_w6
n_w7 = w7 - LR * E_total_w7
n_w8 = w8 - LR * E_total_w8

E_total_out_h1 = (out_1 - t_1) * (out_1 * (1 - out_1)) * w5 + (out_2 - t_2) * (out_2 * (1 - out_2)) * w7
E_total_out_h2 = (out_1 - t_1) * (out_1 * (1 - out_1)) * w6 + (out_2 - t_2) * (out_2 * (1 - out_2)) * w8

E_total_w1 = E_total_out_h1 * (1 - out_h1**2) * in_1
E_total_w2 = E_total_out_h1 * (1 - out_h1**2) * in_2
E_total_w3 = E_total_out_h2 * (1 - out_h2**2) * in_1
E_total_w4 = E_total_out_h2 * (1 - out_h2**2) * in_2

n_w1 = w1 - LR * E_total_w1
n_w2 = w2 - LR * E_total_w2
n_w3 = w3 - LR * E_total_w3
n_w4 = w4 - LR * E_total_w4

print(f"in_h1: {in_h1}")
print(f"in_h2: {in_h2}")
print(f"out_h1: {out_h1}")
print(f"out_h2: {out_h2}")
print(f"in_o1: {in_o1}")
print(f"in_o2: {in_o2}")
print(f"out_1: {out_1}")
print(f"out_2: {out_2}")
print(f"error_o1: {error_o1}")
print(f"error_o2: {error_o2}")
print(f"error_total: {error_total}")
print(f"E_total_w5: {E_total_w5}")
print(f"E_total_w6: {E_total_w6}")
print(f"E_total_w7: {E_total_w7}")
print(f"E_total_w8: {E_total_w8}")
print(f"n_w5: {n_w5}")
print(f"n_w6: {n_w6}")
print(f"n_w7: {n_w7}")
print(f"n_w8: {n_w8}")
print(f"E_total_out_h1: {E_total_out_h1}")
print(f"E_total_out_h2: {E_total_out_h2}")
print(f"E_total_w1: {E_total_w1}")
print(f"E_total_w2: {E_total_w2}")
print(f"E_total_w3: {E_total_w3}")
print(f"E_total_w4: {E_total_w4}")
print(f"n_w1: {n_w1}")
print(f"n_w2: {n_w2}")
print(f"n_w3: {n_w3}")
print(f"n_w4: {n_w4}")
