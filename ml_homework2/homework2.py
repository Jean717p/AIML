print(__doc__)
# pylint: disable=E1101
import operator
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn.preprocessing as prepro
from matplotlib import gridspec
from mlxtend.plotting import plot_decision_regions as plt_reg

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1,stratify=y)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.6, random_state=1, stratify= y_test)

C_vals = []
C_i = 0.001
i=0
for i in range(7):
    C_vals.append(C_i)
    C_i*=10
models = []
scores = np.empty((len(C_vals),3))
#fig = plt.figure(figsize=(10,8))
fig, axarr = plt.subplots(7,3,figsize=(10,8))
ax = axarr.flatten
for j,C in enumerate(C_vals):
    model_k = svm.SVC(kernel='linear',C=C)
    model_l = svm.LinearSVC(C=C)
    model_rbf = svm.SVC(kernel='rbf',C=C)
    model_k.fit(x_train,y_train)
    model_l.fit(x_train,y_train)
    model_rbf.fit(x_train,y_train)
    models.append(model_k)
    models.append(model_l)
    models.append(model_rbf)
    scores[j,0] = model_k.score(x_val,y_val)
    scores[j,1] = model_l.score(x_val,y_val)
    scores[j,2] = model_rbf.score(x_val,y_val)
    
    plt_reg(x_test,y_test,clf=model_k, legend=2, ax=ax)
    plt_reg(x_test,y_test,clf=model_l, legend=2, ax=ax)
    plt_reg(x_test,y_test,clf=model_rbf, legend=2, ax=ax)

plt.show()    
param_range = np.logspace(-3, 3, len(C_vals))
plt.ylim(0.0, 1.1)
lw = 2
bestC = np.empty((len(scores), 3))
for j in range(len(scores[0])):
        c, value = max(enumerate(scores[:,j]),key=operator.itemgetter(1))
        bestC[j,0] = C_vals[c]
        bestC[j,1] = value
        bestC[j,2] = c

plt.semilogx(param_range, scores[:,0], label="Linear Kernel - best score="+str(round(bestC[0,1],2))+" with C="+str(bestC[0,0]), lw=lw)
plt.semilogx(param_range, scores[:,1], label="Linear - best score:"+str(round(bestC[1,1],2))+" with C="+str(bestC[1,0]), lw=lw)
plt.semilogx(param_range, scores[:,2], label="RBF - best score:"+str(round(bestC[2,1],2))+" with C="+str(bestC[2,0]), lw=lw)
plt.xlabel('C')
plt.ylabel('Scores')
plt.legend(loc='best')
plt.show()

fig, sub = plt.subplot(2,4)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 20), 
        np.arange(y_min, y_max, 20))
Z=[]
for clf, title, ax in zip(models,titles,sub.flatten()):
        Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx,yy,Z,cmap=plt.cm.coolwarm, alpha = 0.8)
        ax.scatter(x_test[:, 0], x_test[:, 1],c=y_test, s=20, edgecolor='k')
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm,alpha=0.4)
plt.scatter(x_test[:, 0], x_test[:, 1], s=20, edgecolor='k')
plt.show()

g= 0.7 #gamma
""" linear_models = (svm.SVC(kernel='linear',C=C), 
            svm.LinearSVC(C=C)) """
# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')



def naiveBayesClassifier(input_matrix, classes, firstPC=0, lastPC=0):
        scaled = prepro.StandardScaler().fit_transform(input_matrix)
        if firstPC == 0 and lastPC == 0:
                x_train, x_test, y_train, y_test = train_test_split(scaled, classes, test_size=0.1)
        else:
                #Select only PCA in range
                pca_tot = PCA()
                pca_tot = pca_tot.fit(scaled)
                twoComponents = pca_tot.components_[firstPC:lastPC]
                pca_tot.components_ = twoComponents
                x = pca_tot.transform(scaled)   
                #Train and use the model
                x_train, x_test, y_train, y_test = train_test_split(x, classes, test_size=0.1)

        clf = GaussianNB()
        clf.fit(x_train,y_train)
        prediction = clf.predict(x_test)
        accuracy = accuracy_score(y_test,prediction)
        print(accuracy)

def plotClfBoundaries(input_matrix,classes): #to change with flag which pca to use
        cvec = []
        step = 20
        projected = PCA(2).fit_transform(input_matrix)
        #to project even with 3-4 and as asked from text
        x = projected[:,[0,1]]
        print(x.size)
        x_train, x_test, y_train, y_test = train_test_split(projected, classes, test_size=0.1)
        cvec = [label_color[labels[k]] for k in y_test]

        clf = GaussianNB()
        clf.fit(x_train,y_train)
        # Plotting decision regions
        x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
        y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                        np.arange(y_min, y_max, step))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(x_test[:, 0], x_test[:, 1], c=cvec, s=20, edgecolor='k')
        plt.show()
        plt.clf()

# Inizializzazioni
y = np.zeros((nSamples,), dtype=int)
x, y = importImages('all')
nImg = 10

#PCA transformations {1.2}

showVarCovederageRateoWithPCA(-1,x)
showColorClasses(x)
image_list = []
image_list.append(x[nImg,:])
image_list.append(showImageWithPca(60, x, nImg, False))
image_list.append(showImageWithPca(6, x, nImg, False))
image_list.append(showImageWithPca(2, x, nImg, False))
image_list.append(showImageWithPcaLast(6, x, nImg, False))

figure = plt.figure()
plt.subplots_adjust(hspace = 0.2, wspace = 0.4)

for i in range(len(image_list)):
        figure.add_subplot(2, 3, i+1)
        plt.imshow(np.reshape(image_list[i]/255.0,(227,227,3)))

plt.show()

#PCA last 6 computation
naiveBayesClassifier(x,y) #è possibile inserire le PC inizali e finali (es: (x,y,0,1))
plotClfBoundaries(x,y)
