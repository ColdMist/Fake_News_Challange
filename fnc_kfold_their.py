import sys
import numpy as np
import nltk
nltk.download('wordnet')
from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from sklearn import svm
from utils.system import parse_params, check_version
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    return X,y
def do_PCA(vectors, n_components = 2):
    pca = PCA(n_components = n_components)
    principalComponents = pca.fit_transform(vectors)

    return principalComponents

def ANN_classifier(X_train, y_train):
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    model = Sequential()
    model.add(Dense(8, input_dim=44, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train,dummy_y, epochs=10, batch_size=64)
    return model

if __name__ == "__main__":
    check_version()
    parse_params()

    #Load the training dataset and generate folds
    d = DataSet()
    #print(d)
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition")
    #print(X_competition)
    #print(X_competition.shape)
    #print(y_competition)
    #print(y_competition.shape)
    #exit()
    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")
    for fold in fold_stances:
        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))


    best_score = 0
    best_fold = None


    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))



        X_test = Xs[fold]
        y_test = ys[fold]
        #X_train = do_PCA(X_train, n_components=2)
        #X_test = do_PCA(X_test, n_components=2)
        clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
        #clf = svm.SVC(gamma='scale',verbose=True)
        #clf = RandomForestClassifier(n_estimators=200, max_depth=2, random_state = 14128)
        #clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (10, 5), random_state = 14128)
        #clf = ANN_classifier(X_train, y_train)
        #clf = svm.SVC(kernel='rbf',gamma='auto', verbose=True)
        #clf = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
        clf.fit(X_train, y_train)


        predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
        #predicted = [LABELS[int(a)] for a in clf.predict_classes(X_test)]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = clf



    #Run on Holdout set and report the final score on the holdout set
    #X_holdout = do_PCA(X_holdout, n_components=2)
    #predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    #predicted = [LABELS[int(a)] for a in best_fold.predict_classes(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout]

    print("Scores on the dev set")
    report_score(actual,predicted)
    print("")
    print("")

    #Run on competition dataset
    #X_competition = do_PCA(X_competition, n_components=2)
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
    #predicted = [LABELS[int(a)] for a in best_fold.predict_classes(X_competition)]
    actual = [LABELS[int(a)] for a in y_competition]

    print("Scores on the test set")
    report_score(actual,predicted)
