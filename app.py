import json, os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import joblib, numpy as np, pandas as pd, plotly.express as px, streamlit as st
st.set_page_config(page_title="SL Used Vehicle Price – Ensemble Dashboard",layout="wide")
UNIT="LKR 100,000 (lakhs)"

@st.cache_data
def load():
    return (pd.read_csv("artifacts/clean.csv"),pd.read_csv("artifacts/test_predictions.csv"),
            pd.read_csv("artifacts/feature_importance.csv"),pd.read_csv("artifacts/ann_history.csv"),
            json.load(open("artifacts/metrics.json")))
@st.cache_resource
def models():
    import tensorflow as tf
    return joblib.load("artifacts/models.joblib"), tf.keras.models.load_model("artifacts/ann.keras")
df,pred,imp,hist,M=load()
page=st.sidebar.radio("Section",["Dataset overview","Exploratory analysis","Model comparison","Actual vs predicted","Feature importance","Predict a price"])
st.sidebar.caption(f"Prices in {UNIT}. Split: {M['n_train']} train / {M['n_val']} val / {M['n_test']} test; {M['n_features']} encoded features.")
st.title("Sri Lankan Used Vehicle Price – Ensemble Learning Dashboard")

if page=="Dataset overview":
    c=st.columns(4); c[0].metric("Listings",f"{len(df):,}"); c[1].metric("Brands",df.Brand.nunique()); c[2].metric("Models",df.Model.nunique()); c[3].metric("Median price",f"{df.Price.median():.1f}")
    st.dataframe(df.head(200),width="stretch")
    st.subheader("Summary statistics"); st.dataframe(df[["YOM","Engine","Mileage","Price"]].describe().T)
    st.warning("Mileage is almost perfectly (inversely) correlated with year of manufacture in this dataset (r ≈ -1.0, only ~60 distinct values), so it behaves as a derived field rather than an independent odometer reading.")
elif page=="Exploratory analysis":
    log=st.checkbox("Log scale for price",True)
    st.plotly_chart(px.histogram(df,x="Price",nbins=80,log_x=log,title="Price distribution"),width="stretch")
    a,b=st.columns(2)
    a.plotly_chart(px.box(df,x="Fuel Type",y="Price",log_y=log,title="Price by fuel type"),width="stretch")
    b.plotly_chart(px.box(df,x="Gear",y="Price",log_y=log,title="Price by transmission"),width="stretch")
    st.plotly_chart(px.scatter(df.sample(min(3000,len(df)),random_state=1),x="YOM",y="Price",color="Fuel Type",log_y=log,opacity=.5,title="Price vs year of manufacture"),width="stretch")
    top=df.groupby("Brand").Price.agg(["median","count"]).query("count>=30").sort_values("median",ascending=False).head(20).reset_index()
    st.plotly_chart(px.bar(top,x="Brand",y="median",title="Median price, top 20 brands (≥30 listings)"),width="stretch")
    num=df[["YOM","Engine","Mileage","Price","AC","PS","PM","PW"]].corr()
    st.plotly_chart(px.imshow(num,text_auto=".2f",color_continuous_scale="RdBu_r",zmin=-1,zmax=1,title="Correlation matrix"),width="stretch")
elif page=="Model comparison":
    t=pd.DataFrame(M["test"]).T; v=pd.DataFrame(M["val"]).T
    st.subheader("Test-set metrics (original price units)"); st.dataframe(t.style.format("{:.3f}").highlight_min(subset=["MAE","MSE","RMSE"],color="#cfe8cf").highlight_max(subset=["R2","R2_log"],color="#cfe8cf"),width="stretch")
    st.plotly_chart(px.bar(t.reset_index().melt("index",["MAE","RMSE"]),x="index",y="value",color="variable",barmode="group",labels={"index":"Model","value":"Error"}),width="stretch")
    st.subheader("Hybrid RF + ANN"); w=M["hybrid_rf_weight"]
    st.write(f"Weighted average: **{w:.2f} × Random Forest + {1-w:.2f} × ANN**, weight tuned on the validation set only.")
    base=t.loc[["Random Forest","ANN"]]; h=t.loc["Hybrid RF+ANN"]
    st.info("Hybrid vs its components on test: "+"; ".join(f"{m} MAE {base.loc[m,'MAE']:.2f}" for m in base.index)+f"; hybrid MAE {h['MAE']:.2f}, R² {h['R2']:.3f}. See table for whether hybridisation helped.")
    with st.expander("Validation-set metrics"): st.dataframe(v.style.format("{:.3f}"))
    with st.expander("ANN training curve"): st.line_chart(hist)
elif page=="Actual vs predicted":
    m=st.selectbox("Model",[c for c in pred.columns if c!="actual"],index=3)
    lim=st.checkbox("Zoom to prices under 150",True)
    d=pred[pred.actual<150] if lim else pred
    f=px.scatter(d,x="actual",y=m,opacity=.5,labels={"actual":"Actual","y":"Predicted"}); mx=float(d[["actual",m]].max().max())
    f.add_shape(type="line",x0=0,y0=0,x1=mx,y1=mx,line=dict(dash="dash")); st.plotly_chart(f,width="stretch")
    st.plotly_chart(px.histogram(d.assign(error=d[m]-d.actual),x="error",nbins=80,title="Residuals (predicted − actual)"),width="stretch")
elif page=="Feature importance":
    n=st.slider("Top features",5,len(imp),min(12,len(imp)))
    for col,name in [("rf","Random Forest"),("xgb","XGBoost")]:
        d=imp.sort_values(col,ascending=False).head(n)
        st.plotly_chart(px.bar(d,x=col,y="feature",orientation="h",title=name).update_yaxes(autorange="reversed"),width="stretch")
    st.caption("One-hot columns are summed back to their original feature (e.g. all Model_* columns → Model).")
else:
    (b,ann)=models()
    c1,c2,c3=st.columns(3)
    brand=c1.selectbox("Brand",sorted(df.Brand.unique()),index=sorted(df.Brand.unique()).index("TOYOTA"))
    mods=sorted(df[df.Brand==brand].Model.unique()); model=c2.selectbox("Model",mods)
    sub=df[(df.Brand==brand)&(df.Model==model)]
    town=c3.selectbox("Town",sorted(df.Town.unique()),index=sorted(df.Town.unique()).index("Colombo"))
    d1,d2,d3=st.columns(3)
    yom=d1.slider("Year of manufacture",1980,2024,int(sub.YOM.median()))
    eng=d2.number_input("Engine (cc)",500,6000,int(sub.Engine.median()),step=50)
    # mileage is a near-deterministic function of YOM in this data → estimate it, allow override
    fit=np.polyfit(df.YOM,df.Mileage,1); mil=d3.number_input("Mileage (km)",0,900000,int(max(0,np.polyval(fit,yom))),step=1000)
    e1,e2,e3=st.columns(3)
    gear=e1.selectbox("Transmission",sorted(df.Gear.unique())); fuel=e2.selectbox("Fuel type",sorted(df["Fuel Type"].unique())); 
    new=e3.checkbox("Brand new"); lease=e3.checkbox("Ongoing lease")
    f=st.columns(4); ac=f[0].checkbox("Air conditioning",True); ps=f[1].checkbox("Power steering",True); pm=f[2].checkbox("Power mirrors",True); pw=f[3].checkbox("Power windows",True)
    row=pd.DataFrame([dict(Brand=brand,Model=model,Gear=gear,**{"Fuel Type":fuel},Town=town,YOM=yom,Engine=eng,Mileage=mil,Leasing=int(lease),IsNew=int(new),AC=int(ac),PS=int(ps),PM=int(pm),PW=int(pw))])[b["cat"]+b["num"]]
    A=b["pre"].transform(row)
    p={"Random Forest":b["rf"].predict(A)[0],"XGBoost":b["xgb"].predict(A)[0],"ANN":float(ann.predict(A,verbose=0)[0,0])}
    p["Hybrid RF+ANN"]=b["w"]*p["Random Forest"]+(1-b["w"])*p["ANN"]
    p={k:float(np.expm1(v)) for k,v in p.items()}
    cols=st.columns(4)
    for c,(k,v) in zip(cols,p.items()): c.metric(k,f"{v:,.1f}",help=UNIT)
    st.caption(f"Predicted price in {UNIT}, i.e. 100 = LKR 10 million. {len(sub)} similar {brand} {model} listings in the data (median {sub.Price.median():.1f}).")
