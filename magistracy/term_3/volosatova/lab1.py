from math import pi as m_pi
from math import sin
from math import cos
from math import sqrt
from math import atan
import matplotlib.pyplot as plt
import numpy as np

def ak_f1(k):
  return (1. / (m_pi * k)) * sin((m_pi * k) / 2.) \
    + (2. / ((m_pi  * k) ** 2)) * (cos((m_pi * k) / 2.) - 1.)

def bk_f1(k):
  return (2. / ((m_pi  * k) ** 2)) * sin(m_pi * k / 2.) \
    - (1. / (m_pi * k)) * cos((m_pi * k) / 2.)

def ak_f2(k):
  return (-1. / (m_pi * k)) * sin((m_pi * k) / 2.) \
    - (2. / ((m_pi * k) ** 2)) * (((-1.) ** k) - cos((m_pi * k ) / 2.))

def bk_f2(k):
  return (1. / (m_pi * k)) * cos((m_pi * k) / 2.) \
    + (2. / ((m_pi * k) ** 2)) * sin((m_pi * k) / 2.)

def fourier_cos_sum(n, x, ak):
  s = 0.
  tx = 32.

  for k in range(1, n + 1):
    s += ak(k) * cos(2. * m_pi * k * x / tx)

  return s

def fourier_sin_sum(n, x, bk):
  s = 0.
  tx = 32.

  for k in range(1, n + 1):
    s += bk(k) * sin(2. * m_pi * k * x / tx)

  return s

def fourier_decomposition(n, x):
  a0_f1 = 0.25
  a0_f2 = 0.25
  f1 = (a0_f1 / 2) + fourier_cos_sum(n, x, ak_f1) + fourier_sin_sum(n, x, bk_f1)
  f2 = (a0_f2 / 2) + fourier_cos_sum(n, x, ak_f2) + fourier_sin_sum(n, x, bk_f2)
  return f1 + f2

def main():
  n_harmonics = 50
  x = np.linspace(0, 64, 256)
  y = []

  for p in x:
    y.append(fourier_decomposition(n_harmonics, p))

  plt.xlim(left=0)
  plt.xlabel('x')
  plt.ylabel('f(x)')
  plt.title('n = 50')
  plt.xticks(np.arange(0, 65, 4))
  plt.yticks(np.arange(0, 1.1, 0.2))
  plt.plot(x, y)
  plt.savefig('lab1_5.png')

  d_k = np.zeros(n_harmonics + 1)
  fi_k = np.zeros(n_harmonics + 1)
  d_k[0] = 0.25

  for k in range(1, n_harmonics + 1):
    a_k = ak_f1(k) + ak_f2(k)
    b_k = bk_f1(k) + bk_f2(k)

    if a_k == 0:
      continue

    d_k[k] = sqrt(a_k ** 2 + b_k ** 2)
    fi_k[k] = atan(b_k / a_k)

    plt.clf()
    plt.ylim(top = 0.5, bottom = 0)

    for k in range(len(d_k)):
      plt.axvline(x = k, ymin = 0, ymax = d_k[k] * 2)

    plt.xlabel('k')
    plt.ylabel('d(k)')
    plt.xticks(np.arange(0, n_harmonics + 1, 1))
    plt.plot(np.arange(0, n_harmonics + 1), d_k, marker='_', mew=2, linestyle='None')
    plt.savefig('lab1_7.png')


    plt.clf()
    plt.ylim(top = 2, bottom = 0)

    for k in range(len(fi_k)):
      plt.axvline(x = k, ymin = 0, ymax = fi_k[k] * 0.5)

    plt.xlabel('k')
    plt.ylabel('fi(k)')
    plt.xticks(np.arange(0, n_harmonics + 1, 1))
    plt.plot(np.arange(0, n_harmonics + 1), fi_k, marker='_', mew=2, linestyle='None')
    plt.savefig('lab1_8.png')

if __name__ == '__main__':
  main()
