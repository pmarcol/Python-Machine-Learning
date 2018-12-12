from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

def plot_regions(x, y, clsf, xl, yl, tit, leg_loc=2):
    plot_decision_regions(x, y, clf=clsf, legend=leg_loc)

    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(tit)
    plt.show()

iris_yl = 'petal length [cm]'
iris_xl = 'sepal length [cm]'