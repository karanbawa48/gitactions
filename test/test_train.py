import joblib
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train import train_and_save_model

def test_train_and_save_model():
    # Train the model and save it
    train_and_save_model()
    
    # Check if the model file is created
    assert os.path.exists('model.joblib')
    
    # Load the model and check if it is a RandomForestClassifier
    model = joblib.load('model.joblib')
    from sklearn.ensemble import RandomForestClassifier
    assert isinstance(model, RandomForestClassifier)
