from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

DATA_FILE = Path(r'D:\RECLAIM\data\generated\reclaim_causal_experiment_v3.csv')
OUTPUT_DIR = Path(r'D:\RECLAIM\data\generated')
N_RUNS = 20
OUTER_TEST_SIZE = 0.30
INNER_VALID_SIZE = 0.25
RF_ESTIMATORS = 300
RF_MIN_LEAF = 20
THRESHOLDS = np.array([-200,-150,-100,-75,-50,-25,0,25,50,75,100,150,200], dtype=float)
SEEDS = range(1000,1020)


def header(s):
    print('\n' + '='*70); print(s); print('='*70)

def money(x): return f'₹{x:,.2f}'

def first(df, names):
    return next((x for x in names if x in df.columns), None)


def schema(df):
    return {
        'treatment': first(df,['treatment','treatment_assignment_before_outcome']),
        'recovery': first(df,['recovered','recovery','recovery_flag']),
        'effect': first(df,['treatment_effect','incremental_recovery_probability']),
        'observed_net': first(df,['net_recovered_value']),
        'amount': first(df,['amount','amount_prediction']),
    }


def validate(df,s):
    header('VALIDATING V3 EXPERIMENT')
    missing=[k for k in ['treatment','recovery','effect','observed_net'] if s[k] is None]
    if missing: raise ValueError(f'Missing semantic columns: {missing}\nAvailable: {list(df.columns)}')
    t=pd.to_numeric(df[s['treatment']],errors='coerce')
    y=pd.to_numeric(df[s['recovery']],errors='coerce')
    e=pd.to_numeric(df[s['effect']],errors='coerce')
    n=pd.to_numeric(df[s['observed_net']],errors='coerce')
    if t.isna().any() or not set(t.astype(int).unique()).issubset({0,1}): raise ValueError('Treatment must be 0/1')
    if y.isna().any(): raise ValueError('Recovery contains invalid values')
    if e.isna().any(): raise ValueError('Treatment effect contains invalid values')
    if n.isna().any(): raise ValueError('Observed economic column contains invalid values')
    print('[PASS] Required columns present')
    print('[PASS] Treatment assignment valid')
    print('[PASS] Recovery outcome valid')
    print('[PASS] Treatment-effect values valid')
    print('[PASS] Economic values valid')


def features(df,s):
    excluded={
        s['treatment'],s['recovery'],s['effect'],s['observed_net'],
        'control','treatment_group','experiment_version','random_seed',
        'experiment_transaction_id','experiment_source_transaction_id',
        'base_probability','actual_recovery_probability','observed_recovery_probability',
        'expected_control_recovery','expected_treatment_recovery','expected_incremental_recovery',
        'control_recovery_probability','treatment_recovery_probability','incremental_recovery_probability',
        'expected_incremental_net_value','recovered_amount','recovery_status','recovery_flag',
        'expected_recovery','expected_net_recovery','expected_recovery_prediction',
        'expected_net_recovery_prediction','action_recovery_probability','decision_reason',
        'recommended_action','recommended_policy','policy_threshold','priority','opportunity_rank',
        'intervention_cost','expected_intervention_cost','customer_friction_cost','expected_friction_cost',
        'net_recovered_value','treatment_assignment_before_outcome','transaction_id','customer_id','timestamp'
    }
    cols=[c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    if not cols: raise ValueError('No numeric modeling features found')
    print('\n'+'='*70); print('LEAKAGE PROTECTION'); print('='*70)
    print('[PASS] Outcome/treatment/counterfactual columns excluded')
    print('Modeling features:',len(cols))
    return cols


def economic_layer(df,s):
    '''Create a transparent V9-style economic stress layer from the V3 truth.
    Treatment effect is used ONLY for counterfactual evaluation, never features.'''
    obs=pd.to_numeric(df[s['observed_net']],errors='coerce').to_numpy(float)
    t=pd.to_numeric(df[s['treatment']],errors='coerce').to_numpy(int)
    e=pd.to_numeric(df[s['effect']],errors='coerce').to_numpy(float)
    if s['amount']:
        a=pd.to_numeric(df[s['amount']],errors='coerce').fillna(0).to_numpy(float)
    else: a=np.full(len(df),np.nanmedian(np.abs(obs)))
    scale=max(float(np.nanmedian(np.abs(a))),1.0)
    af=np.clip(a/scale,.25,3.0)
    ce=e-e.mean()
    inc=850*ce + 240*np.sign(ce)*np.sqrt(np.abs(ce)+1e-12)*af - (35+.025*np.clip(a,0,None))
    control=np.where(t==0,obs,obs-inc)
    treatment=control+inc
    lo,hi=np.percentile(np.r_[control,treatment],[0.5,99.5])
    control=np.clip(control,lo,hi); treatment=np.clip(treatment,lo,hi); inc=treatment-control
    return control,treatment,inc


def fit_uplift(train,pred,cols,s,seed):
    X=train[cols]; P=pred[cols]
    y=pd.to_numeric(train[s['recovery']],errors='coerce').astype(int).to_numpy()
    t=pd.to_numeric(train[s['treatment']],errors='coerce').astype(int).to_numpy()
    mt=RandomForestClassifier(n_estimators=RF_ESTIMATORS,min_samples_leaf=RF_MIN_LEAF,max_features='sqrt',random_state=seed,n_jobs=-1,class_weight='balanced_subsample')
    mc=RandomForestClassifier(n_estimators=RF_ESTIMATORS,min_samples_leaf=RF_MIN_LEAF,max_features='sqrt',random_state=seed+1,n_jobs=-1,class_weight='balanced_subsample')
    mt.fit(X.loc[t==1],y[t==1]); mc.fit(X.loc[t==0],y[t==0])
    pt=mt.predict_proba(P)[:,1]; pc=mc.predict_proba(P)[:,1]
    return pt,pc,pt-pc


def qini(y,t,u):
    order=np.argsort(-u); y=np.asarray(y,float)[order]; t=np.asarray(t,int)[order]
    tm=(t==1); cm=(t==0)
    nt=np.cumsum(tm); nc=np.cumsum(cm)
    yt=np.cumsum(y*tm); yc=np.cumsum(y*cm)
    ctrl=np.divide(yc,nc,out=np.zeros_like(yc),where=nc>0)
    inc=yt-nt*ctrl
    return float(np.trapezoid(inc,x=np.arange(1,len(y)+1)))


def policy(control,treatment,u,thr):
    p=u>thr; v=np.where(p,treatment,control)
    return float(v.mean()),float(p.mean()),p


def choose_threshold(u,control,treatment):
    best=None
    for th in THRESHOLDS:
        v,r,_=policy(control,treatment,u,th)
        # Economic value first; when essentially tied, prefer less intervention.
        score=(v,-r)
        if best is None or score>best[0]: best=(score,float(th),v,r)
    return best[1],best[2],best[3]


def ci(x):
    x=np.asarray(x,float); m=x.mean(); se=x.std(ddof=1)/np.sqrt(len(x))
    return float(m),float(m-1.96*se),float(m+1.96*se)


def run(df,s,cols,control,treatment,seed):
    idx=np.arange(len(df))
    tr,te=train_test_split(idx,test_size=OUTER_TEST_SIZE,random_state=seed,stratify=df[s['treatment']])
    outer=df.iloc[tr].copy(); test=df.iloc[te].copy()
    # Threshold is selected on inner validation only.
    itr,iv=train_test_split(np.arange(len(outer)),test_size=INNER_VALID_SIZE,random_state=seed+5000,stratify=outer[s['treatment']])
    inner=outer.iloc[itr].copy(); valid=outer.iloc[iv].copy()
    valid_global=tr[iv]
    _,_,uv=fit_uplift(inner,valid,cols,s,seed)
    th,_,_=choose_threshold(uv,control[valid_global],treatment[valid_global])
    _,_,ut=fit_uplift(outer,test,cols,s,seed+20000)
    value,rate,pol=policy(control[te],treatment[te],ut,th)
    retry=float(treatment[te].mean()); no=float(control[te].mean())
    inc=treatment[te]-control[te]; oracle_pol=inc>0; oracle_value=float(np.where(oracle_pol,treatment[te],control[te]).mean())
    y=pd.to_numeric(test[s['recovery']],errors='coerce').astype(int).to_numpy(); tt=pd.to_numeric(test[s['treatment']],errors='coerce').astype(int).to_numpy()
    q=qini(y,tt,ut)
    adv=value-retry; oadv=oracle_value-retry
    return dict(seed=seed,training_rows=len(outer),test_rows=len(test),mean_probability_uplift=float(ut.mean()),selected_threshold=th,intervention_rate=rate,targeting_value=value,retry_all_value=retry,no_intervention_value=no,learned_advantage_vs_retry_all=adv,targeting_advantage_vs_no_intervention=value-no,oracle_value=oracle_value,oracle_intervention_rate=float(oracle_pol.mean()),oracle_advantage_vs_retry_all=oadv,qini=q,qini_per_row=q/len(test),oracle_opportunity_capture=adv/oadv if oadv>0 else np.nan,beats_retry_all=adv>0,beats_no_intervention=value>no)


def main():
    header('RECLAIM CAUSAL UPLIFT OPTIMIZER V10')
    print('CROSS-FITTED-STYLE UPLIFT + QINI + ECONOMIC POLICY STRESS TEST')
    print('VALIDATION-ONLY THRESHOLD SELECTION + UNTOUCHED OUTER TEST')
    print('File:',DATA_FILE)
    if not DATA_FILE.exists(): raise FileNotFoundError(DATA_FILE)
    df=pd.read_csv(DATA_FILE); print('[PASS] Rows loaded:',len(df)); print('[PASS] Columns loaded:',len(df.columns))
    s=schema(df); validate(df,s); cols=features(df,s)
    header('EXPERIMENT BASELINE')
    tm=df[s['treatment']].astype(int)==1; cm=~tm
    rt=df.loc[tm,s['recovery']].astype(float).mean(); rc=df.loc[cm,s['recovery']].astype(float).mean()
    print(f'Treatment recovery rate: {rt:.2%}'); print(f'Control recovery rate:   {rc:.2%}'); print(f'Observed recovery lift:  {rt-rc:+.2%}')
    e=df[s['effect']].astype(float)
    header('TREATMENT EFFECT HETEROGENEITY')
    print(f'Mean effect: {e.mean():.2%}'); print(f'Std effect: {e.std():.2%}'); print(f'Minimum effect: {e.min():.2%}'); print(f'Maximum effect: {e.max():.2%}')
    print('Positive effects:',int((e>0).sum())); print('Negative effects:',int((e<0).sum())); print('Zero effects:',int((e==0).sum()))
    if not (e>0).any() or not (e<0).any(): raise ValueError('V10 requires positive and negative treatment effects')
    header('CREATING V10 ECONOMIC POLICY LAYER')
    control,treatment,inc=economic_layer(df,s)
    print('Positive incremental-value rows:',int((inc>0).sum())); print('Negative incremental-value rows:',int((inc<0).sum())); print('Mean incremental value:',money(inc.mean())); print('Std incremental value:',money(inc.std()))
    header('RUNNING V10 REPEATED HELD-OUT VALIDATION')
    print('Validation runs:',N_RUNS); print('Outer test size:',f'{OUTER_TEST_SIZE:.0%}'); print('Inner validation size:',f'{INNER_VALID_SIZE:.0%}')
    rows=[]
    for i,seed in enumerate(SEEDS,1):
        print('\n'+'-'*70); print(f'[{i:02d}/{N_RUNS}] SEED {seed}')
        r=run(df,s,cols,control,treatment,seed); rows.append(r)
        print(f"Training rows: {r['training_rows']} | Test rows: {r['test_rows']}")
        print(f"Mean probability uplift: {r['mean_probability_uplift']:.2%}")
        print(f"Selected threshold: {money(r['selected_threshold'])}")
        print(f"Intervention rate: {r['intervention_rate']:.2%}")
        print(f"QINI: {r['qini']:,.2f}")
        print('Targeting value/transaction:',money(r['targeting_value']))
        print('RETRY_ALL value/transaction:',money(r['retry_all_value']))
        print('Learned advantage vs RETRY_ALL:',money(r['learned_advantage_vs_retry_all']))
        print('Oracle advantage vs RETRY_ALL:',money(r['oracle_advantage_vs_retry_all']))
        print('Oracle opportunity captured:',f"{r['oracle_opportunity_capture']:.1%}" if np.isfinite(r['oracle_opportunity_capture']) else 'N/A')
    val=pd.DataFrame(rows)
    qm,ql,qh=ci(val.qini); am,al,ah=ci(val.learned_advantage_vs_retry_all); om,ol,oh=ci(val.oracle_advantage_vs_retry_all)
    header('V10 STABILITY SUMMARY')
    print('Validation runs:',len(val)); print(f'Mean QINI: {qm:,.2f}'); print(f'QINI 95% CI: [{ql:,.2f}, {qh:,.2f}]'); print('Positive-QINI rate:',f'{(val.qini>0).mean():.2%}')
    print(f'Mean probability uplift: {val.mean_probability_uplift.mean():.2%}'); print(f'Mean intervention rate: {val.intervention_rate.mean():.2%}')
    header('TARGETING VS NO_INTERVENTION'); print('Mean advantage:',money(val.targeting_advantage_vs_no_intervention.mean())); print('Runs beating NO_INTERVENTION:',f'{val.beats_no_intervention.mean():.2%}')
    header('TARGETING VS RETRY_ALL'); print('Mean learned advantage:',money(am)); print(f'95% CI: [{money(al)}, {money(ah)}]'); print('Runs beating RETRY_ALL:',f'{val.beats_retry_all.mean():.2%}')
    header('ORACLE BENCHMARK'); print('Mean oracle advantage:',money(om)); print('Oracle positive-rate:',f'{(val.oracle_advantage_vs_retry_all>0).mean():.2%}'); print('Mean oracle intervention rate:',f'{val.oracle_intervention_rate.mean():.2%}'); print('Mean oracle opportunity captured:',f'{val.oracle_opportunity_capture.mean():.2%}')
    header('POLICY STABILITY'); print(val.selected_threshold.value_counts().sort_index().to_string())
    header('FINAL V10 CAUSAL POLICY VERDICT')
    print(f'Mean QINI: {qm:,.2f}'); print('Positive QINI rate:',f'{(val.qini>0).mean():.2%}'); print('Mean learned advantage:',money(am)); print(f'Learned 95% CI: [{money(al)}, {money(ah)}]'); print('Mean oracle advantage:',money(om)); print('Mean oracle opportunity captured:',f'{val.oracle_opportunity_capture.mean():.2%}')
    if qm>0 and al>0 and val.beats_retry_all.mean()>=.80: verdict='V10_TARGETING_SUPPORTED'
    elif al>0 and val.beats_retry_all.mean()>=.80: verdict='V10_ECONOMIC_TARGETING_SUPPORTED_BUT_QINI_WEAK'
    else: verdict='V10_INCONCLUSIVE'
    print('\nFINAL VERDICT:',verdict)
    # Full-data scoring for deployment-style output; not used for validation metrics.
    header('GENERATING FINAL FULL-DATA POLICY SCORE')
    pt,pc,u=fit_uplift(df,df,cols,s,123456); th=float(val.selected_threshold.median()); fp=u>th; fv=np.where(fp,treatment,control).mean(); fr=treatment.mean(); print('Median selected threshold:',money(th)); print('Full-data intervention rate:',f'{fp.mean():.2%}'); print('Full-data targeting value:',money(fv)); print('Full-data RETRY_ALL value:',money(fr)); print('Full-data advantage:',money(fv-fr))
    pred=pd.DataFrame(index=df.index)
    for c in ['transaction_id','customer_id']:
        if c in df.columns: pred[c]=df[c]
    pred['predicted_treatment_probability']=pt; pred['predicted_control_probability']=pc; pred['predicted_probability_uplift']=u; pred['selected_policy']=np.where(fp,'TARGET','RETRY_ALL'); pred['economic_threshold']=th
    pred['oracle_control_value']=control; pred['oracle_treatment_value']=treatment; pred['oracle_incremental_value']=inc
    pol=pd.DataFrame({'policy':['NO_INTERVENTION','LEARNED_TARGETING','RETRY_ALL','ORACLE'],'mean_value':[val.no_intervention_value.mean(),val.targeting_value.mean(),val.retry_all_value.mean(),val.oracle_value.mean()],'advantage_vs_retry_all':[val.no_intervention_value.mean()-val.retry_all_value.mean(),am,0,om],'intervention_rate':[0,val.intervention_rate.mean(),1,val.oracle_intervention_rate.mean()]})
    summary=pd.DataFrame([{'version':'V10','validation_runs':N_RUNS,'mean_qini':qm,'qini_ci_lower':ql,'qini_ci_upper':qh,'positive_qini_rate':(val.qini>0).mean(),'mean_probability_uplift':val.mean_probability_uplift.mean(),'mean_intervention_rate':val.intervention_rate.mean(),'mean_targeting_value':val.targeting_value.mean(),'mean_retry_all_value':val.retry_all_value.mean(),'mean_learned_advantage':am,'learned_ci_lower':al,'learned_ci_upper':ah,'runs_beating_retry_all':val.beats_retry_all.mean(),'mean_oracle_advantage':om,'oracle_ci_lower':ol,'oracle_ci_upper':oh,'mean_oracle_opportunity_capture':val.oracle_opportunity_capture.mean(),'median_threshold':th,'final_verdict':verdict}])
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    paths={'Validation':OUTPUT_DIR/'causal_uplift_v10_validation_results.csv','Predictions':OUTPUT_DIR/'causal_uplift_v10_predictions.csv','Policies':OUTPUT_DIR/'causal_uplift_v10_policy_comparison.csv','Summary':OUTPUT_DIR/'causal_uplift_v10_summary.csv'}
    val.to_csv(paths['Validation'],index=False); pred.to_csv(paths['Predictions'],index=False); pol.to_csv(paths['Policies'],index=False); summary.to_csv(paths['Summary'],index=False)
    header('V10 COMPLETE'); print(f'Mean QINI: {qm:,.2f}'); print('Positive QINI rate:',f'{(val.qini>0).mean():.2%}'); print('Mean learned advantage:',money(am)); print(f'Learned 95% CI: [{money(al)}, {money(ah)}]'); print('Mean oracle advantage:',money(om)); print('FINAL VERDICT:',verdict)
    for k,p in paths.items(): print(f'[PASS] {k}: {p}')

if __name__=='__main__': main()