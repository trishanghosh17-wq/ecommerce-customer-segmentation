import joblib,numpy as np,pandas as pd
from pathlib import Path
BASE=Path(__file__).resolve().parent
scaler=joblib.load(BASE/"models/rfm_scaler.joblib"); kmeans=joblib.load(BASE/"models/kmeans_model.joblib"); clf=joblib.load(BASE/"models/customer_segment_classifier.joblib"); meta=joblib.load(BASE/"models/metadata.joblib")
def predict_segment(recency,frequency,monetary):
 X=pd.DataFrame([[recency,frequency,monetary]],columns=meta["features"]); xl=np.log1p(X); xs=scaler.transform(xl); c=int(kmeans.predict(xs)[0]); p=int(clf.predict(xl)[0]); return {"cluster":c,"segment":meta["cluster_names"][p],"classifier_prediction":p}
if __name__=="__main__": print(predict_segment(30,5,2000))
