

from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
matplotlib.use('Agg')


class variables:
    counter = 0


def plotGraph(input):
    iris = datasets.load_iris()
    X1 = iris.data[:, [0, 1]]
    y1 = iris.target

    svm = SVC(C=0.5, kernel='linear')
    svm.fit(X1, y1)
    plt.clf()
    plt.figure(facecolor='#171717')

    plot_decision_regions(X1, y1, clf=svm, legend=2)

    xValue = 0
    yValue = 0

    print(input)
    print(X1.shape)
    print(y1.shape)

    if input == 0:
        xValue = 4.5
        yValue = 5
    elif input == 2:
        xValue = 5.5
        yValue = 2.25
    elif input == 3:
        xValue = 7.5
        yValue = 2.5

    plt.scatter(
        xValue, yValue, s=80, facecolors="red", zorder=10, edgecolor="k"
    )

    # plt.rcParams['axes.facecolor'] = 'black'

    image_file_name = 'static/assets/images/spectrograms' + \
        str(variables.counter)+'.jpg'
    plt.savefig(image_file_name)
