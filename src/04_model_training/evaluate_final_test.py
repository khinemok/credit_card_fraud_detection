"""Phase 4.5: evaluate ONE frozen candidate on the untouched test split.
Run from project root: python src/04_model_training/evaluate_final_test.py
Edit phase_4_5_frozen_config.json first. Never tune using test results.
"""
from pathlib import Path
import json, hashlib, platform, sys
from datetime import datetime, timezone
import joblib, numpy as np, pandas as pd, sklearn
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'phase_4_5_frozen_config.json'; OUT=ROOT/'outputs/phase_4_5'; OUT.mkdir(parents=True,exist_ok=True)
MODEL_PATHS={
 ('Logistic Regression','Baseline'):ROOT/'artifacts/models/logistic_regression_baseline.joblib',
 ('Decision Tree','Baseline'):ROOT/'artifacts/models/decision_tree_baseline.joblib',
 ('Random Forest','Baseline'):ROOT/'artifacts/models/random_forest_baseline.joblib',
 ('Logistic Regression','Class-weighted'):ROOT/'artifacts/class_weighted/logistic_regression_class_weighted.joblib',
 ('Decision Tree','Class-weighted'):ROOT/'artifacts/class_weighted/decision_tree_class_weighted.joblib',
 ('Random Forest','Class-weighted'):ROOT/'artifacts/class_weighted/random_forest_class_weighted.joblib',
 ('Logistic Regression','SMOTE'):ROOT/'artifacts/smote/logistic_regression_smote.joblib',
 ('Decision Tree','SMOTE'):ROOT/'artifacts/smote/decision_tree_smote.joblib',
 ('Random Forest','SMOTE'):ROOT/'artifacts/smote/random_forest_smote.joblib'}
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
def read(path):
 if not path.exists(): raise FileNotFoundError(f'Missing required file: {path}')
 return pd.read_csv(path)
def main():
 if not CONFIG.exists(): raise FileNotFoundError(f'Create config at {CONFIG} before running.')
 cfg=json.loads(CONFIG.read_text()); model_name=cfg.get('model'); strategy=cfg.get('strategy'); threshold=cfg.get('threshold'); rationale=str(cfg.get('selection_rationale','')).strip()
 if (model_name,strategy) not in MODEL_PATHS: raise ValueError('Invalid model/strategy in config. See guide for valid values.')
 if not isinstance(threshold,(int,float)) or not 0<float(threshold)<1: raise ValueError('Threshold must be between 0 and 1.')
 if not rationale or 'REPLACE' in rationale.upper(): raise ValueError('Replace the selection_rationale placeholder with the validation-based rationale.')
 data=ROOT/'outputs/preprocessing'; yfile=data/'unscaled/test_y.csv'; y=read(yfile).squeeze('columns').astype(int)
 # Match each training experiment's preprocessing: baseline models and SMOTE models were trained on scaled X;
 # class-weighted LR was scaled, class-weighted trees were unscaled.
 scaled=(strategy in ('Baseline','SMOTE')) or model_name=='Logistic Regression'
 xfile=(data/'logistic_regression_scaled/test_X_scaled.csv') if scaled else (data/'unscaled/test_X.csv')
 X=read(xfile)
 if len(X)!=len(y): raise ValueError(f'Test X/y row mismatch: {len(X)} vs {len(y)}')
 if not np.isfinite(X.to_numpy(dtype=float)).all(): raise ValueError('Non-finite test predictors found; stop and investigate.')
 if set(y.unique())-set([0,1]): raise ValueError('Test target must be binary 0/1.')
 mpath=MODEL_PATHS[(model_name,strategy)]
 if not mpath.exists(): raise FileNotFoundError(f'Selected model artifact missing: {mpath}')
 model=joblib.load(mpath); cols=getattr(model,'feature_names_in_',None)
 if cols is not None and list(cols)!=list(X.columns): raise ValueError('Test feature names/order differ from model. Stop and reconcile preprocessing.')
 prob=model.predict_proba(X)[:,1]; pred=(prob>=float(threshold)).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 result={'phase':'4.5_final_test','model':model_name,'strategy':strategy,'threshold':float(threshold),'selection_rationale':rationale,'n_test':int(len(y)),'fraud_count':int(y.sum()),'legitimate_count':int((y==0).sum()),'fraud_prevalence':float(y.mean()),'average_precision':float(average_precision_score(y,prob)),'roc_auc':float(roc_auc_score(y,prob)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'predicted_fraud_count':int(pred.sum()),'model_artifact':str(mpath.relative_to(ROOT)),'model_sha256':sha(mpath),'test_predictor_file':str(xfile.relative_to(ROOT)),'test_predictor_sha256':sha(xfile),'test_label_file':str(yfile.relative_to(ROOT)),'test_label_sha256':sha(yfile),'test_set_used_for_training_or_tuning':False,'timestamp_utc':datetime.now(timezone.utc).isoformat(),'python_version':sys.version,'platform':platform.platform(),'scikit_learn_version':sklearn.__version__,'note':'One frozen candidate evaluated. Do not tune/select using these test results.'}
 pd.DataFrame([result]).to_csv(OUT/'final_test_metrics.csv',index=False)
 pd.DataFrame({'y_true':y.to_numpy(),'fraud_probability':prob,'predicted_class':pred,'threshold':float(threshold)}).to_csv(OUT/'final_test_predictions.csv',index=False)
 pd.DataFrame([[tn,fp],[fn,tp]],index=['Actual legitimate (0)','Actual fraud (1)'],columns=['Predicted legitimate (0)','Predicted fraud (1)']).to_csv(OUT/'final_test_confusion_matrix.csv')
 (OUT/'final_test_metadata.json').write_text(json.dumps(result,indent=2))
 (OUT/'final_test_classification_report.txt').write_text(classification_report(y,pred,labels=[0,1],target_names=['Legitimate','Fraud'],zero_division=0))
 print('\nPHASE 4.5 COMPLETE — FINAL TEST EVALUATION')
 print(f"Candidate: {model_name} | {strategy} | threshold={threshold}")
 for k in ['n_test','fraud_count','average_precision','roc_auc','precision','recall','f1','tn','fp','fn','tp']: print(f'{k}: {result[k]}')
 print(f'Outputs: {OUT}\nDo not tune or select another model using test results.')
if __name__=='__main__': main()
