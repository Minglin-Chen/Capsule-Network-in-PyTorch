import numpy as np 
import matplotlib.pyplot as plt 


if __name__=='__main__':

    x = np.linspace(0.0, 100.0, 100, endpoint=True)
    y = x / (1 + x)

    plt.plot(x, y)
    plt.title('squash function')
    plt.show()