import random

# القيم الثابتة
e = 2.71828182845904
in_1 = 0.05
in_2 = 0.10
bias_1 = 0.5
bias_2 = 0.7
t_1 = 0.01
t_2 = 0.99
LR = 0.5

weights = [random.uniform(-0.05, 0.5) for _ in range(8)]
w1, w2, w3, w4, w5, w6, w7, w8 = weights

error_threshold = 0.001

while True:
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

    if error_total < error_threshold:
        break

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

    w1, w2, w3, w4, w5, w6, w7, w8 = n_w1, n_w2, n_w3, n_w4, n_w5, n_w6, n_w7, n_w8

print(" Inputs")
print(f"in_1: {in_1}, in_2: {in_2}")
print(f"bias_1: {bias_1}, bias_2: {bias_2}")
print(f"t_1: {t_1}, t_2: {t_2}")
print(f"Learning Rate (LR): {LR}")

print(" Final Weights")
print(f"w1: {w1}, w2: {w2}, w3: {w3}, w4: {w4}")
print(f"w5: {w5}, w6: {w6}, w7: {w7}, w8: {w8}")

print("Forward Computed Values")
print(f"in_h1: {in_h1}, in_h2: {in_h2}")
print(f"out_h1: {out_h1}, out_h2: {out_h2}")
print(f"in_o1: {in_o1}, in_o2: {in_o2}")
print(f"out_1: {out_1}, out_2: {out_2}")

print("Errors")
print(f"error_o1: {error_o1}, error_o2: {error_o2}")
print(f"total_error: {error_total}")