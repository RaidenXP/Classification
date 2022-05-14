import pandas as pd
import classification

#for this lab 
#true postive = edible mushrooms recognized as edible
#true negative = posionous mushrooms recongized as poisonous
#false positive = poisonous classified as edible
#false negaive = edible recognized as poisonous

def evaluate(prefix, y, predy):
    correct = 0
    for p,v in zip(predy, y):
        if p == v: correct += 1
    print("%sAccuracy: %.2f"%(prefix, correct*1.0/len(y)))
    
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for p,v in zip(predy, y):
        if p == 'e' and p == v:
            true_positive += 1
        if p == 'e' and p != v:
            false_positive += 1
        if p == 'p' and p != v:
            false_negative += 1
    
    print("%sPrecision: %.2f"%(prefix,
            true_positive/(true_positive + false_positive)))
    
    print("%sRecall: %.2f"%(prefix,
            true_positive/(true_positive + false_negative)))

def get_columns(rows, columns, single=False):
    if single:
        return [row[columns[0]] for row in rows]
    return [[row[c] for c in columns] for row in rows]

def analysis():
    training_df = pd.read_csv("training.csv")
    test_df = pd.read_csv("test.csv")

    training = []
    test = []

    for i,row in training_df.iterrows():
        training.append(row)

    for i,row in test_df.iterrows():
        test.append(row)

    columns = ["odor", "ring-type"]

    target = ["class"]

    classifier = classification.DecisionTree()
    tx = get_columns(training, columns)
    ty = get_columns(training, target, single=True)
    classifier.fit(tx, ty)
    predty = classifier.predict(tx)
    evaluate("Mushroom Training ", ty, predty)

    print("\n")

    tex = get_columns(test, columns)
    tey = get_columns(test, target, single=True)
    predtey = classifier.predict(tex)
    evaluate("Mushroom Testing ", tey, predtey)

    return classifier

def main():
    final_classifier = analysis()

    val_df = pd.read_csv("validation.csv")

    val = []

    #make sure to change this if you are changing it in the analysis function
    columns = ["odor", "ring-type"]

    target = ["class"]

    for i,row in val_df.iterrows():
        val.append(row)

    print("\n")

    #comment and uncomment this part so you can try out your columns
    vx = get_columns(val, columns)
    vy = get_columns(val, target, single=True)
    predvy = final_classifier.predict(vx)
    evaluate("Mushroom Validation ", vy, predvy)

if __name__ == "__main__":
    main()