#!./venv/bin/python3
import numpy as np
import scipy
import scipy.optimize
import json

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, StratifiedKFold


seed = 31

def get_hard_svm_lp(x, y):
    # var (w, b, w_abs)

    n, m = x.shape

    c = np.zeros((2*m + 1))
    c[m+1:] = 1
    a1 = np.zeros((n, 2*m + 1))
    
    a1[:, :m] = x
    a1[:, m] = 1

    a1 *= -y[:, None]
    b1 = -np.ones((n,))

    a2 = np.zeros((2*m, 2*m + 1))

    a2[np.arange(m), np.arange(m)] = 1
    a2[m + np.arange(m), np.arange(m)] = -1

    a2[np.arange(m), m + 1 + np.arange(m)] = -1
    a2[m + np.arange(m), m + 1 + np.arange(m)] = -1

    b2 = np.zeros((2*m,))

    a = a1
    b = b1

    a = np.concat((a1, a2), axis=0)
    b = np.concat((b1, b2), axis=0)

    bounds = [(None, None)]*(m + 1) + [(0, None)]*m

    return c, a, b, bounds

def get_soft_svm_lp(x, y, l_pos=10, l_neg=None, l_w=1):
    if l_neg is None:
        l_neg = l_pos

    n, m = x.shape

    c_h, a_h, b, bounds_h = get_hard_svm_lp(x, y)
    c_h = c_h*l_w

    c_s = np.zeros((n,))
    c_s[y == 1] = l_pos
    c_s[y == -1] = l_neg

    c = np.concat((c_h, c_s))

    a_s = np.concat((-np.identity(n),np.zeros((2*m, n))), axis=0)

    a = np.concat((a_h, a_s), axis=1)

    bounds = bounds_h + [(0, None)]*n

    return c, a, b, bounds

def get_min_component_svm_lp(x, y, l_pos=10, l_neg=None, l_w=1):
    if l_neg is None:
        l_neg = l_pos

    n, m = x.shape
    
    c, a, b, bounds = get_soft_svm_lp(x, y, l_pos, l_neg, l_w)
    
    c_mc = np.zeros((c.shape[0] + 1, ))
    c_mc[-1] = 1

    b_mc = np.concat((b, np.zeros((3,))))

    a_e1 = np.zeros((3, a.shape[1]))
    a_e1[0, m+1:2*m+1] = l_w
    a_e1[1, 2*m+1:][y == 1] = l_pos
    a_e1[2, 2*m+1:][y == -1] = l_neg

    a_mc = np.concat((a, a_e1), axis=0)
    
    a_e2 = np.zeros_like(b_mc)
    a_e2[-3:] = -1

    a_mc = np.concat((a_mc, a_e2.reshape(-1, 1)), axis=1)

    bounds_mc = bounds + [(0, None)]

    return c_mc, a_mc, b_mc, bounds_mc

    

class LinprogSVM:
    def __init__(self, l_pos=None, l_neg=None, l_w=1, use_min_com=False):
        self.l_pos = l_pos
        self.l_neg = l_neg
        self.l_w = l_w
        self.use_min_com = use_min_com

    def get_params(self, deep=True):
        return {"l_pos": self.l_pos, "l_neg": self.l_neg, "l_w": self.l_w, 
                "use_min_com": self.use_min_com}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
    
    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        
        self.n_pos_ = np.sum(y == 1)
        self.n_neg_ = np.sum(y == -1)

        if self.l_pos is None and self.l_neg is None:
            self.c_, self.a_, self.b_, self.bounds_= get_hard_svm_lp(X, y)
        elif self.use_min_com:
            self.c_, self.a_, self.b_, self.bounds_ = get_min_component_svm_lp(X, y, 
                l_pos=self.l_pos, l_neg=self.l_neg, l_w=self.l_w)
        else:
            self.c_, self.a_, self.b_, self.bounds_ = get_soft_svm_lp(X, y, 
                l_pos=self.l_pos, l_neg=self.l_neg, l_w=self.l_w)
        
        res = scipy.optimize.linprog(self.c_, self.a_, self.b_, bounds=self.bounds_)

        n, m = X.shape

        self.w_ = res.x[:m,]
        self.b_ = res.x[m]

        return self

    def predict(self, X):
        y_pred = np.sign(self.coef_(X))
        y_pred[np.logical_and(y_pred != -1, y_pred != 1)] = 1 if self.n_pos_ >= self.n_neg_ else -1
        
        return y_pred

    def coef_(self, X):
        return X @ self.w_ + self.b_

class Steuer_opt:
    def __init__(self, max_iter=4):
        self.max_iter = max_iter

    def fit(self, X, y):
        # (prec, v, it)
        k = 3
        self.best_fit = (0, None, None)

        v = np.zeros((2*k + 1, k))
        v[0:k, :] = np.identity(k)

        def calc_v_follow():
            v[k, :] = np.sum(v[0:k, :], axis=0)/k
            v[k+1:2*k+1, :] = np.sum(v[0:k+1, :], axis=0)[None, :]/k - v[0:k, :]/k

        calc_v_follow()

        def get_accu(l):
            estimator = LinprogSVM(l_pos=l[1], l_neg=l[2], l_w=l[0])
            cv_res = cross_validate(estimator, X, y, scoring='balanced_accuracy', n_jobs=-1, 
                                    cv=StratifiedKFold(5, random_state=seed, shuffle=True))

            return float(np.mean(cv_res['test_score']))


        for it in range(self.max_iter):
            accu = [get_accu(v[i, :]) for i in range(2*k+1)]
            best_idx = np.argmax(accu)
            if accu[best_idx] > self.best_fit[0]:
                self.best_fit = (accu[best_idx], v[best_idx, :], it)

            if best_idx <= k+1:
                p = v[best_idx, :]

            else:
                j = best_idx - (k + 2)
                p = np.sum(v[0:k, :][np.arange(k) != j, :], axis=0)/(k - 1)
            
            v[0:k, :] = (v[0:k, :] + p[None, :])/2
            calc_v_follow()
        

        l = self.best_fit[1]
        self.clf_ = LinprogSVM(l_pos=l[1], l_neg=l[2], l_w=l[0])
        self.clf_.fit(X, y)


    def predict(self, X):
        return self.clf_.predict(X)


def normalize_features(x):
    m = np.mean(x, axis=0)
    sd = np.std(x, axis=0)

    x_n = (x - m)/sd

    return x_n

def load_banknote(norm_feat=True):
    ds = np.genfromtxt('data/BankNote_Authentication.csv', delimiter=',', skip_header=1)

    x = ds[:,0:4]
    y = np.astype(ds[:,4], int)
    y[y == 0] = -1

    if norm_feat:
        x = normalize_features(x)
    
    return x, y

def load_breast_cancer(norm_feat=True):
    ds = np.genfromtxt('data/breast_cancer.csv', delimiter=',', skip_header=1, 
                       converters= { 1 : lambda s: 1.0 if s == 'M' else -1.0})

    x = ds[:,2:]
    y = ds[:,1].astype(int)

    if norm_feat:
        x = normalize_features(x)
    
    return x, y

def load_rice(norm_feat=True):
    ds = np.genfromtxt('data/rice.csv', delimiter=',',
                       converters= { 7 : lambda s: 1.0 if s == 'Cammeo' else -1.0})

    x = ds[:,0:7]
    y = np.astype(ds[:,7], int)

    if norm_feat:
        x = normalize_features(x)
    
    return x, y

def generate_data(m=20, n=1000, p_pos=0.5, n_pos=None, n_neg=None, diff_mult=1):
    if n_pos is None and n_neg is None:
        n_pos = int(np.round(n*p_pos))
        n_neg = n - n_pos

    y = np.ones((n_pos + n_neg,))
    y[n_pos:] = -1

    mean = np.zeros((m,))
    cov = np.identity(m)

    x = np.random.multivariate_normal(mean, cov, n)
    x_diff = np.random.multivariate_normal(mean, diff_mult*cov, 1)
    x[n_pos:, :] += x_diff
    
    return x, y

if __name__ == '__main__':

    # m = 2
    # n = 3
    # p_pos = 0.4
    # x, y = generate_data(m=m, n=n, p_pos=p_pos)
    # print(get_min_component_svm_lp(x, y))

    x, y = load_breast_cancer()
    # x, y = load_banknote()
    # x, y = load_rice()

    f = LinprogSVM(l_pos=10, use_min_com=True).fit(x, y)

    ds_name = zip(
        [load_banknote, load_breast_cancer, load_rice],
        ["banknote", "breast_cancer", "rice"])

    data = {}

    for ds, name in ds_name:
        
        x, y = ds()
        print("--", name, "--")
        x_trn, x_tst, y_trn, y_tst = train_test_split(x, y, test_size=0.1, shuffle=True, random_state=seed)
        
        pl = (0.2, 1, 5, 25, 125)
        param = { 'l_pos' : pl, 'l_neg' : pl}
        gs = GridSearchCV(LinprogSVM(), param, scoring='balanced_accuracy', n_jobs=-1,
                          cv=StratifiedKFold(5, random_state=seed, shuffle=True))
        gs.fit(x_trn, y_trn)

        y_pred = gs.predict(x_tst)

        accu = accuracy_score(y_tst, y_pred)
        cm = confusion_matrix(y_tst, y_pred)
        l_pos = gs.best_params_['l_pos']
        l_neg = gs.best_params_['l_pos']

        print("Grid search")
        print("accu:", accu)
        print("cm:")
        print(cm)
        print(f"l pos: {l_pos}, l neg: {l_neg}")

        data[name] = {}
        data[name]["gs"] = {}

        data[name]["gs"]["accu"] = float(accu)
        data[name]["gs"]["cm"] = cm.tolist()
        data[name]["gs"]["l_pos"] = float(l_pos)
        data[name]["gs"]["l_neg"] = float(l_neg)

        gs_mc = GridSearchCV(LinprogSVM(use_min_com=True), param, scoring='balanced_accuracy', n_jobs=-1,
                             cv=StratifiedKFold(5, random_state=seed, shuffle=True))
        
        gs_mc.fit(x_trn, y_trn)

        y_pred = gs_mc.predict(x_tst)

        accu = accuracy_score(y_tst, y_pred)
        cm = confusion_matrix(y_tst, y_pred)
        l_pos = gs_mc.best_params_['l_pos']
        l_neg = gs_mc.best_params_['l_pos']
        
        print("Grid search - max component")
        print("accu:", accu)
        print("cm:")
        print(cm)
        print(f"l pos: {l_pos}, l neg: {l_neg}")

        data[name]["mc"] = {}

        data[name]["mc"]["accu"] = float(accu)
        data[name]["mc"]["cm"] = cm.tolist()
        data[name]["mc"]["l_pos"] = float(l_pos)
        data[name]["mc"]["l_neg"] = float(l_neg)
    
        so = Steuer_opt()
        so.fit(x_trn, y_trn)

        y_pred = so.predict(x_tst)

        accu = accuracy_score(y_tst, y_pred)
        cm = confusion_matrix(y_tst, y_pred)
        best_fit_v = so.best_fit[1]
        l_pos = best_fit_v[1]/best_fit_v[0]
        l_neg = best_fit_v[2]/best_fit_v[0]
        
        print("Steuer")
        print("accu:", accu)
        print("cm:")
        print(cm)
        print(f"l pos: {l_pos}, l neg: {l_neg}")


        data[name]["steuer"] = {}

        data[name]["steuer"]["accu"] = float(accu)
        data[name]["steuer"]["cm"] = cm.tolist()
        data[name]["steuer"]["l_pos"] = float(l_pos)
        data[name]["steuer"]["l_neg"] = float(l_neg)
        data[name]["steuer"]["iter"] = so.best_fit[2]

    with open(f"results.json", "w") as f:
        json.dump(data, f, indent=2)