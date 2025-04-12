import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from deap import base, creator, tools, algorithms
import numpy as np
import os

def DataExplore(df, name):
    print(f"\nEDA for {name}:")
    print(f"Dataset Shape: {df.shape}")
    print(f"Dataset Columns: {df.columns}")
    
    print("\nData Types and Non-Null Counts:")
    print(df.info())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nSummary Statistics:")
    print(df.describe())

    print("\nUnique Counts for Each Column:")
    unique_counts = df.nunique()
    print(unique_counts)

    print("\nFirst 5 rows (df.head()):")
    print(df.head())

    print("\nLast 5 rows (df.tail()):")
    print(df.tail())

    print(f"\nNumber of Duplicates: {df.duplicated().sum()}")
    print(f"Number of Nulls: {df.isnull().sum().sum()}")

    if 'OCCUPATION_TYPE' in df.columns:
        print("\nUnique values in 'OCCUPATION_TYPE':")
        print(df['OCCUPATION_TYPE'].value_counts(dropna=False))
    
    return df
df1 = pd.read_csv('application_record.csv')
df1 = DataExplore(df1, 'application_record')

df2 = pd.read_csv('credit_record.csv')
df2 = DataExplore(df2, 'credit_record')

if os.path.exists("mergedDataset.csv"):
    print("Merged dataset found. Loading it...")
    final_df = pd.read_csv("mergedDataset.csv")
else:
    print("Merging datasets...")
    final_df = pd.merge(df1, df2, on="ID", how="inner")
    final_df.to_csv("mergedDataset.csv", index=False)

labelEncoder = LabelEncoder()
dataToEncode = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE"
]

for label in dataToEncode:
    if label in final_df.columns:
        final_df[label + "_Encoded"] = labelEncoder.fit_transform(final_df[label])
    else:
        print(f"Column '{label}' not found in the dataset.")

final_df.drop(columns=[col for col in dataToEncode if col in final_df.columns], inplace=True)

if 'STATUS' in final_df.columns:
    column_to_move = 'STATUS'
    last_column = final_df.pop(column_to_move)
    final_df[column_to_move] = last_column

status_mapping = {'C': 0, 'X': 0, '0': 1, '1': 1, '2': 1, '3': 1, '4': 1, '5': 1}
final_df["STATUS"] = final_df["STATUS"].map(status_mapping)

print(f"Initial number of duplicates: {final_df.duplicated().sum()}")
print(f"Initial number of missing values: {final_df.isnull().sum().sum()}")

final_df.drop_duplicates(inplace=True)

final_df.fillna(-1, inplace=True)

print(f"After cleaning, number of duplicates: {final_df.duplicated().sum()}")
print(f"After cleaning, number of missing values: {final_df.isnull().sum().sum()}")


print("******************************************************************")
print("FINAL DF IS NAN",final_df["STATUS"].isna().sum())
print("FINAL DF IS NULL",final_df["STATUS"].isnull().sum())
print("******************************************************************")


for column in final_df.select_dtypes(include=['int64', 'float64']).columns:
    if final_df[column].min() < 0:
        print(f"Found negative values in column '{column}'. Replacing with median.")
        median_value = final_df[column][final_df[column] >= 0].median()
        final_df[column] = final_df[column].apply(lambda x: median_value if x < 0 else x)

X = final_df.drop(columns=["STATUS"])
y = final_df["STATUS"]
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train_full, X_test, y_train_full, y_test = train_test_split(X_resampled, y_resampled, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1765, random_state=42)

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", np.random.randint, 0, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(X_train.columns))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    selected_features = [feature for feature, select in zip(X_train.columns, individual) if select == 1]
    if len(selected_features) == 0:
        return 0,
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train[selected_features], y_train)
    y_val_pred = clf.predict(X_val[selected_features])
    return accuracy_score(y_val, y_val_pred),

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=20)
ngen, cxpb, mutpb = 10, 0.7, 0.3
result = algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=True)

best_individual = tools.selBest(population, k=1)[0]
selected_features = [feature for feature, select in zip(X_train.columns, best_individual) if select == 1]
print("\nSelected Features:", selected_features)

param_dist_dt = {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]}
param_dist_knn = {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]}
param_dist_mlp = {"hidden_layer_sizes": [(50,), (100,), (50, 50)], "activation": ["relu", "tanh"], "solver": ["adam", "sgd"], "learning_rate": ["constant", "adaptive"]}

models = {
    "DecisionTree": (DecisionTreeClassifier(random_state=42), param_dist_dt),
    "KNN": (KNeighborsClassifier(), param_dist_knn),
    "MLP": (MLPClassifier(random_state=42, max_iter=300), param_dist_mlp)
}

for model_name, (model, param_dist) in models.items():
    print(f"\nTuning hyperparameters for {model_name}...")

    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=3, random_state=42, scoring="accuracy", n_jobs=-1)
    random_search.fit(X_train[selected_features], y_train)
    
    print(f"{model_name} Random Search Best Parameters: {random_search.best_params_}")
    print(f"{model_name} Random Search Best Score: {random_search.best_score_:.2f}")

    best_model = random_search.best_estimator_
    y_test_pred = best_model.predict(X_test[selected_features])
    print(f"\n{model_name} Test Accuracy with Best Model: {accuracy_score(y_test, y_test_pred):.2f}")
    print(classification_report(y_test, y_test_pred))
