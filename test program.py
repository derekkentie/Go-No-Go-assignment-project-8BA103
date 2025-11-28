from main import train_model, predict

# Example training
params = train_model(X_train, y_train)

# Example prediction
y_pred = predict(X_test, params)