import pandas as pd
import random

def main():
    df = pd.read_csv("mushrooms.csv")
    training = []
    remaining = []
    for i,row in df.iterrows():
        if random.random() < 0.70:
            training.append(row)
        else:
            remaining.append(row)

    validation = []
    test = []
    for row in remaining:
        if random.random() > 0.50:
            validation.append(row)
        else:
            test.append(row)

    
    training_df = pd.DataFrame(training)
    test_df = pd.DataFrame(test)
    validation_df = pd.DataFrame(validation)

    training_df.to_csv("training.csv", index=False)
    test_df.to_csv("test.csv", index=False)
    validation_df.to_csv("validation.csv", index=False)


if __name__ == "__main__":
    main()