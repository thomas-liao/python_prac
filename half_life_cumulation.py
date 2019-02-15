import matplotlib.pyplot as plt
import numpy as np

days = 10

half_life = 36 # 36 hours of half life
# N = No * (0.5) t / T
dose_each = 400 # 400 mg intake every day

days_span = [i for i in range(days)]

# assuming we are taking every day

cum_ = [0 for _ in range(days)]


cum_[0] = dose_each

for i in range(1, days):
    cum_[i] = cum_[i-1]*(0.5**(24/half_life)) + dose_each


plt.plot(days_span, cum_)
plt.show()

print(cum_)