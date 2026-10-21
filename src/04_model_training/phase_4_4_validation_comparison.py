"""Phase 4.4: validation-only comparison and threshold selection. Run from project root."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs'/'phase_4_4'
FILES={
('Logistic Regression','Baseline'):ROOT/'outputs/evaluation/baseline_predictions/logistic_regression_validation_predictions.csv',
('Decision Tree','Baseline'):ROOT/'outputs/evaluation/baseline_predictions/decision_tree_validation_predictions.csv',
('Random Forest','Baseline'):ROOT/'outputs/evaluation/baseline_predictions/random_forest_validation_predictions.csv',
('Logistic Regression','Class-weighted'):ROOT/'outputs/class_weighted/validation_predictions/logistic_regression_validation_predictions.csv',
('Decision Tree','Class-weighted'):ROOT/'outputs/class_weighted/validation_predictions/decision_tree_validation_predictions.csv',
('Random Forest','Class-weighted'):ROOT/'outputs/class_weighted/validation_predictions/random_forest_validation_predictions.csv',
('Logistic Regression','SMOTE'):ROOT/'outputs/smote/validation_predictions/logistic_regression_validation_predictions.csv',
('Decision Tree','SMOTE'):ROOT/'outputs/smote/validation_predictions/decision_tree_validation_predictions.csv',
('Random Forest','SMOTE'):ROOT/'outputs/smote/validation_predictions/random_forest_validation_predictions.csv'}

def load(path):
    if not path.exists(): raise FileNotFoundError(f'Missing {path}. Check earlier Phase 4 output paths; do not use test predictions.')
    d=pd.read_csv(path)
    if 'y_true' not in d: raise ValueError(f'{path}: expected y_true; found {list(d.columns)}')
    pc=next((c for c in ['fraud_probability','y_probability','y_prob','probability'] if c in d),None)
    if pc is None: raise ValueError(f'{path}: no recognized probability column: {list(d.columns)}')
    y=d.y_true.astype(int).to_numpy(); p=pd.to_numeric(d[pc],errors='raise').to_numpy(float)
    if len(y)==0 or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError(f'Invalid labels/probabilities in {path}')
    return y,p

def at_threshold(y,p,t):
    pred=(p>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'threshold':float(t),'precision':precision_score(y,pred,zero_division=0),'recall':recall_score(y,pred,zero_division=0),'f1':f1_score(y,pred,zero_division=0),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'flagged_count':int(pred.sum())}

def threshold_options(y,p):
    pr,rc,ts=precision_recall_curve(y,p); pr,rc=pr[:-1],rc[:-1]
    f1=np.divide(2*pr*rc,pr+rc,out=np.zeros_like(pr),where=(pr+rc)>0)
    ix=int(np.argmax(f1)); rows=[('Maximum validation F1',None,ix)]
    for target in (.90,.95):
        eligible=np.flatnonzero(pr>=target)
        ix2=sorted(eligible,key=lambda i:(rc[i],f1[i],ts[i]),reverse=True)[0] if len(eligible) else None
        rows.append((f'Maximum recall with precision >= {target:.0%}',target,ix2))
    out=[]
    for rule,target,i in rows:
        if i is None: out.append({'selection_rule':rule,'precision_target':target,'threshold':np.nan,'validation_precision':np.nan,'validation_recall':np.nan,'validation_f1':np.nan,'rule_met':False})
        else: out.append({'selection_rule':rule,'precision_target':target,'threshold':float(ts[i]),'validation_precision':float(pr[i]),'validation_recall':float(rc[i]),'validation_f1':float(f1[i]),'rule_met':True})
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'figures').mkdir(exist_ok=True)
    summary=[]; thresholds=[]; loaded={}
    for (model,strategy),path in FILES.items():
        y,p=load(path); loaded[(model,strategy)]=(y,p)
        summary.append({'model':model,'strategy':strategy,'n_validation':len(y),'fraud_count':int(y.sum()),'fraud_prevalence':float(y.mean()),'average_precision':average_precision_score(y,p),'roc_auc':roc_auc_score(y,p),**at_threshold(y,p,.5),'prediction_file':str(path.relative_to(ROOT))})
        for opt in threshold_options(y,p):
            row={'model':model,'strategy':strategy,**opt}
            if np.isfinite(opt['threshold']): row.update({f'threshold_{k}':v for k,v in at_threshold(y,p,opt['threshold']).items()})
            thresholds.append(row)
    s=pd.DataFrame(summary); t=pd.DataFrame(thresholds)
    s.to_csv(OUT/'validation_comparison_threshold_0_5.csv',index=False); t.to_csv(OUT/'validation_threshold_selection.csv',index=False)
    pivot=s.pivot(index='model',columns='strategy',values='average_precision').reindex(['Logistic Regression','Decision Tree','Random Forest'])
    ax=pivot.plot(kind='bar',figsize=(10,6)); ax.set_title('Validation Average Precision by Model and Strategy'); ax.set_ylabel('Average Precision'); ax.set_xlabel(''); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(OUT/'figures'/'average_precision_comparison.png',dpi=200); plt.close()
    curve_rows=[]
    for model in ['Logistic Regression','Decision Tree','Random Forest']:
        plt.figure(figsize=(8,6))
        for strategy in ['Baseline','Class-weighted','SMOTE']:
            y,p=loaded[(model,strategy)]; pr,rc,ts=precision_recall_curve(y,p); ap=average_precision_score(y,p); plt.plot(rc,pr,label=f'{strategy} (AP={ap:.3f})')
            for j in range(len(pr)): curve_rows.append({'model':model,'strategy':strategy,'recall':rc[j],'precision':pr[j],'threshold':ts[j] if j<len(ts) else np.nan})
        prevalence=float(loaded[(model,'Baseline')][0].mean()); plt.axhline(prevalence,linestyle='--',linewidth=1,label=f'Prevalence ({prevalence:.3%})'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'Precision–Recall Curves: {model}'); plt.grid(True,alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'figures'/(model.lower().replace(' ','_')+'_precision_recall.png'),dpi=200); plt.close()
    pd.DataFrame(curve_rows).to_csv(OUT/'precision_recall_curve_points.csv',index=False)
    metadata={'phase':'4.4','test_set_loaded':False,'primary_metric':'Average Precision','threshold_0_5_used_for_initial_comparison':True,'threshold_selection':'Validation-only maximum F1 and maximum recall subject to precision >= 0.90 / 0.95','note':'AP is not necessarily identical to trapezoidal PR-AUC. Threshold results are validation estimates, not final test results. Review the operational objective before freezing a candidate.'}
    (OUT/'phase_4_4_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('\nPHASE 4.4 COMPLETE — VALIDATION ONLY')
    print(s[['model','strategy','average_precision','roc_auc','precision','recall','f1','fp','fn']].to_string(index=False))
    print('\nOutputs:',OUT,'\nTest set was not loaded.')
if __name__=='__main__': main()
