import numpy as np
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
                             
def get_delta_vec(vec_type, word, kv):    
    vec_lut = {
        "c1": word[0],
        "c2": word[1],
        "mu1": f"{word[0]}*",
        "mu2": f"*{word[1]}",
        "d1": f"d1({word[0]}|{word})",
        "d2": f"d2({word[1]}|{word})",
        "dw": f"dw({word})",
        "wv": word
    }    
    return kv.get_vector(vec_lut.get(vec_type), norm=False)

def feature_extraction(feat_type, entries, kv):
    Xvecs = []
    ylabels = []
    for entry_x in entries:
        token = entry_x["token_key"]
        try:
            if "+" not in feat_type:
                vec = get_delta_vec(feat_type, token, kv)
            else:
                feats = [x.strip() for x in feat_type.strip().split("+")]
                vec = np.concatenate([get_delta_vec(feats[0], token, kv), 
                                      get_delta_vec(feats[1], token, kv)], axis=0)
        except Exception as ex:                        
            continue
        Xvecs.append(vec)
        ylabels.append(entry_x["MorphoSyntax"])
    Xvecs = np.vstack(Xvecs)
    ylabels = np.array(ylabels)
    assert Xvecs.shape[0] == len(ylabels)
   
    return Xvecs, ylabels

def feature_extraction_with_key(feat_type, entries, kv):
    Xvecs = []
    ylabels = []
    key_list = []
    for entry_x in entries:
        token = entry_x["token_key"]
        try:
            if "+" not in feat_type:
                vec = get_delta_vec(feat_type, token, kv)
            else:
                feats = [x.strip() for x in feat_type.strip().split("+")]
                vec = np.concatenate([get_delta_vec(feats[0], token, kv), 
                                      get_delta_vec(feats[1], token, kv)], axis=0)
        except Exception as ex:                        
            continue
        Xvecs.append(vec)
        ylabels.append(entry_x["MorphoSyntax"])
        key_list.append(token)
    Xvecs = np.vstack(Xvecs)
    ylabels = np.array(ylabels)
    assert Xvecs.shape[0] == len(ylabels)
   
    return Xvecs, key_list, ylabels

def estimate_lda(feat_type, clf_entries, kv, kf_seed=12345):
    kf = KFold(n_splits=5, shuffle=True, random_state=kf_seed)       
    Xvecs, ylabels = feature_extraction(feat_type, clf_entries, kv)
        
    data_idxs = np.arange(ylabels.shape[0])
    metrics = []
    for fold_idx, (train_idxs, test_idxs) in enumerate(kf.split(data_idxs)):
        Xtrain = Xvecs[train_idxs]; Xtest = Xvecs[test_idxs]
        ytrain = ylabels[train_idxs]; ytest = ylabels[test_idxs] 

        lda = LinearDiscriminantAnalysis()
        lda.fit(Xtrain, ytrain)
        ypred_test = lda.predict(Xtest)
        ypred_train = lda.predict(Xtrain)

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(Xtrain, ytrain)
        ydummy = dummy.predict(Xtest)
        
        test_acc = accuracy_score(ytest, ypred_test)
        train_acc = accuracy_score(ytrain, ypred_train)
        dummy_acc = accuracy_score(ytest, ydummy)
        metrics.append(dict(     
            clf = "lda",
            feat_type = feat_type,
            fold_idx = fold_idx,
            n_train = len(train_idxs),
            n_test = len(test_idxs),
            train_acc = train_acc,
            test_acc = test_acc,
            dummy_acc = dummy_acc            
        ))
    return metrics

def estimate_tree(clf_entries, kf_seed=12345):
    kf = KFold(n_splits=5, shuffle=True, random_state=kf_seed)    
    X = np.array([list(x["token"]) for x in clf_entries])
    y = np.array([x["MorphoSyntax"] for x in clf_entries])
        
    data_idxs = np.arange(y.shape[0])
    metrics = []
    for fold_idx, (train_idxs, test_idxs) in enumerate(kf.split(data_idxs)):
        Xtrain = X[train_idxs]; Xtest = X[test_idxs]
        ytrain = y[train_idxs]; ytest = y[test_idxs] 
                
        enc = OneHotEncoder(handle_unknown='ignore')
        Xtrain = enc.fit_transform(Xtrain)
        tree = DecisionTreeClassifier(random_state=kf_seed)
        tree.fit(Xtrain, ytrain)
        ypred_test = tree.predict(enc.transform(Xtest))
        ypred_train = tree.predict(Xtrain)

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(Xtrain, ytrain)
        ydummy = dummy.predict(Xtest)
        
        test_acc = accuracy_score(ytest, ypred_test)
        train_acc = accuracy_score(ytrain, ypred_train)
        dummy_acc = accuracy_score(ytest, ydummy)
        metrics.append(dict( 
            clf = "tree",
            feat_type = "char",
            fold_idx = fold_idx,
            n_train = len(train_idxs),
            n_test = len(test_idxs),
            train_acc = train_acc,
            test_acc = test_acc,
            dummy_acc = dummy_acc            
        ))
    return metrics