from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class BaseModel(ABC):
    @abstractmethod
    def train(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def evaluate(self, X, y):
        pass
    
    @abstractmethod
    def save(self, path):
        pass
    
    @abstractmethod
    def load(self, path):
        pass

class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        
    def train(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)
        return {'accuracy': acc, 'report': report}
    
    def save(self, path):
        joblib.dump(self.model, path)
        
    def load(self, path):
        if os.path.exists(path):
            self.model = joblib.load(path)
        else:
            raise FileNotFoundError(f"Model file not found: {path}")

class TransformerModel(BaseModel):
    """
    Transformer 模型占位符
    待大样本情况下启用
    """
    def __init__(self):
        pass
        
    def train(self, X, y):
        print("Transformer training not implemented yet.")
        
    def predict(self, X):
        return None
        
    def evaluate(self, X, y):
        return {}
        
    def save(self, path):
        pass
        
    def load(self, path):
        pass
