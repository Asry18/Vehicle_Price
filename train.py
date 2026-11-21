import json, joblib, numpy as np, pandas as pd, os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import tensorflow as tf
SEED=42; np.random.seed(SEED); tf.random.set_seed(SEED)

df=pd.read_csv("car_price_dataset.csv").drop(columns=["Unnamed: 0"]).drop_duplicates()
df=df.rename(columns={"Millage(KM)":"Mileage","Engine (cc)":"Engine","AIR CONDITION":"AC","POWER STEERING":"PS","POWER MIRROR":"PM","POWER WINDOW":"PW"})
for c in ["AC","PS","PM","PW"]: df[c]=(df[c]=="Available").astype(int)
df["Leasing"]=(df["Leasing"]!="No Leasing").astype(int)
df["IsNew"]=(df["Condition"]=="NEW").astype(int)
df["Age"]=2025-df["YOM"]
cat=["Brand","Model","Gear","Fuel Type","Town"]
num=["YOM","Engine","Mileage","Leasing","IsNew","AC","PS","PM","PW"]
X=df[cat+num]; y=np.log1p(df["Price"])
Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=0.30,random_state=SEED)
Xva,Xte,yva,yte=train_test_split(Xtmp,ytmp,test_size=0.50,random_state=SEED)

pre=ColumnTransformer([("c",OneHotEncoder(handle_unknown="infrequent_if_exist",min_frequency=5,sparse_output=False),cat),
                       ("n",StandardScaler(),num)])
Atr=pre.fit_transform(Xtr); Ava=pre.transform(Xva); Ate=pre.transform(Xte)
print("features:",Atr.shape)

def metrics(ytrue_log,pred_log):
    t=np.expm1(ytrue_log); p=np.expm1(pred_log)
    mse=mean_squared_error(t,p)
    return dict(MAE=mean_absolute_error(t,p),MSE=mse,RMSE=mse**.5,R2=r2_score(t,p),R2_log=r2_score(ytrue_log,pred_log))

rf=RandomForestRegressor(n_estimators=400,min_samples_leaf=1,max_features=0.4,n_jobs=-1,random_state=SEED).fit(Atr,ytr)
xgb=XGBRegressor(n_estimators=2000,learning_rate=0.03,max_depth=6,subsample=0.8,colsample_bytree=0.6,min_child_weight=2,
                 early_stopping_rounds=60,n_jobs=-1,random_state=SEED).fit(Atr,ytr,eval_set=[(Ava,yva)],verbose=False)
print("xgb best iter",xgb.best_iteration)
m=tf.keras.Sequential([tf.keras.layers.Input((Atr.shape[1],)),
    tf.keras.layers.Dense(256,activation="relu"),tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128,activation="relu"),tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64,activation="relu"),tf.keras.layers.Dense(1)])
m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="mse")
cb=[tf.keras.callbacks.EarlyStopping(patience=25,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(patience=8,factor=0.5)]
hist=m.fit(Atr,ytr.values,validation_data=(Ava,yva.values),epochs=300,batch_size=64,callbacks=cb,verbose=0)
print("ann epochs",len(hist.history["loss"]))

P={k:{"val":f.predict(Ava).ravel(),"test":f.predict(Ate).ravel()} for k,f in [("Random Forest",rf),("XGBoost",xgb),("ANN",m)]}
# hybrid: RF+ANN weight chosen on validation set only
ws=np.linspace(0,1,101)
best=min(ws,key=lambda w:mean_squared_error(yva,w*P["Random Forest"]["val"]+(1-w)*P["ANN"]["val"]))
P["Hybrid RF+ANN"]={s:best*P["Random Forest"][s]+(1-best)*P["ANN"][s] for s in ["val","test"]}
print("hybrid RF weight",best)
res={k:metrics(yte,v["test"]) for k,v in P.items()}
resval={k:metrics(yva,v["val"]) for k,v in P.items()}
print(pd.DataFrame(res).T.round(3))

os.makedirs("artifacts",exist_ok=True)
joblib.dump(dict(pre=pre,rf=rf,xgb=xgb,w=float(best),cat=cat,num=num),"artifacts/models.joblib",compress=3)
m.save("artifacts/ann.keras")
pred=pd.DataFrame({"actual":np.expm1(yte.values),**{k:np.expm1(v["test"]) for k,v in P.items()}})
pred.to_csv("artifacts/test_predictions.csv",index=False)
fn=pre.get_feature_names_out()
imp=pd.Series(rf.feature_importances_,index=fn)
# aggregate one-hot importances back to original feature
agg={}
for n,v in imp.items():
    base=n.split("__")[1]; 
    for c in cat:
        if base.startswith(c+"_"): base=c; break
    agg[base]=agg.get(base,0)+v
pd.DataFrame({"feature":list(agg),"rf":list(agg.values()),
              "xgb":[sum(v for n,v in zip(fn,xgb.feature_importances_) if (n.split('__')[1].startswith(f+'_') if f in cat else n.split('__')[1]==f)) for f in agg]}
            ).sort_values("rf",ascending=False).to_csv("artifacts/feature_importance.csv",index=False)
json.dump(dict(test=res,val=resval,hybrid_rf_weight=float(best),n_train=len(Xtr),n_val=len(Xva),n_test=len(Xte),n_features=int(Atr.shape[1])),open("artifacts/metrics.json","w"),indent=1)
pd.DataFrame(hist.history).to_csv("artifacts/ann_history.csv",index=False)
df.to_csv("artifacts/clean.csv",index=False)
