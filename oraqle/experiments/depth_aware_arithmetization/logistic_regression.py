import numpy as np
from gurobipy import Model, GRB, quicksum
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Load and prepare dataset
    data = load_breast_cancer()
    X, y = data.data, data.target  # type: ignore
    y = 2 * y - 1  # Convert labels to {-1, 1}
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)

    # Consider the range to be [-100, 100]
    bound = 1.0001
    p = 786433
    half = 100

    def fixed_prec(x: float) -> int:
        print(x)
        assert -bound <= x <= bound
        return round(x / bound * half)

    X_encoded = np.vectorize(fixed_prec)(X)
    print(X_encoded)


    # Reduce dataset for demo
    X, X_test, y, y_test = train_test_split(X_encoded, y, train_size=100, random_state=0)
    n_samples, n_features = X.shape

    # Gurobi model
    model = Model("IntegerLogisticRegression")
    weight_bounds = (-100, 100)

    # Add integer weights and continuous intercept
    w = [model.addVar(vtype=GRB.INTEGER, lb=weight_bounds[0], ub=weight_bounds[1], name=f"w_{j}") for j in range(n_features)]
    b = model.addVar(vtype=GRB.INTEGER, name="b")
    model.update()

    # Build exp(-y_i * (w·x + b)) loss
    loss_terms = []
    for i in range(n_samples):
        # Create variables
        yz_var = model.addVar(lb=-GRB.INFINITY, name=f"yz_{i}")
        exp_var = model.addVar(lb=0.0, name=f"exp_{i}")

        # yz = -y_i * (w·x + b)
        model.addConstr(yz_var == -y[i] * (quicksum(w[j] * X[i, j] for j in range(n_features)) + b),
                        name=f"yz_constraint_{i}")

        # z = exp(yz)
        model.addGenConstrExp(yz_var, exp_var, name=f"exp_constraint_{i}")
        loss_terms.append(exp_var)

    # Objective: minimize total exponential loss
    model.setObjective(quicksum(loss_terms), GRB.MINIMIZE)

    model.setParam('OutputFlag', 1)
    model.optimize()
    assert model.status == GRB.OPTIMAL

    # Compute the logits (z = w·x + b) for the test set
    weights = np.array([int(v.X) for v in w])
    intercept = b.X
    print(weights)
    print(intercept)
    print(X_test)
    logits = np.dot(X_test, weights) + intercept

    # Find the highest and lowest logits
    max_logit = np.max(logits)
    min_logit = np.min(logits)

    print(f"Highest logit: {max_logit:.4f}")
    print(f"Lowest logit: {min_logit:.4f}")


    correct = 0
    wrong = 0
    for xx, yy in zip(X_test, y_test):
        prediction = ((np.dot(xx, weights) + intercept) % p) < (p // 2)
        prediction = prediction * 2 - 1
        if yy == prediction:
            correct += 1
        else:
            wrong += 1
    print(correct, wrong)


    # Generate the circuit
    TODO
