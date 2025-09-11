from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

#pre-load once at startup
spam_vectorizer = joblib.load('spam-vectorizer.joblib')
spam_classifier = joblib.load('spam-classifier.joblib')

#region models
class Message(BaseModel):
    text: str
#endregion

#region api
@app.post("/is_spam")
def is_spam(msg: Message) -> dict:
    """Returns if a message is spam or not, with probability."""
    msg_vector = spam_vectorizer.transform([msg.text])
    probability = spam_classifier.predict_proba(msg_vector)[0][1]
    return {"text": msg.text, "is_spam": bool(probability > 0.5), "probability": probability}
#endregion