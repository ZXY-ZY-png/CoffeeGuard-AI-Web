import csv
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent
DISEASE_MODEL_PATH = BASE_DIR / "coffee_disease_model.pth"
SEVERITY_MODEL_PATH = BASE_DIR / "coffee_severity_model.pth"
HISTORY_FILE = BASE_DIR / "recognition_history.csv"

CLASSES = ["miner", "rust", "phoma", "cercospora"]
CLASS_NAMES = {"miner":"咖啡潜叶虫","rust":"咖啡叶锈病","phoma":"Phoma 病","cercospora":"Cercospora 病"}
SEVERITY_NAMES = {0:"0级",1:"1级",2:"2级",3:"3级",4:"4级"}
HEADERS = ["时间","图片","识别结果","AI置信度(%)","严重程度","严重程度置信度(%)","综合风险等级"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

st.set_page_config(page_title="CoffeeGuard AI", page_icon="☕", layout="wide")

@st.cache_resource
def load_models():
    if not DISEASE_MODEL_PATH.exists() or not SEVERITY_MODEL_PATH.exists():
        raise FileNotFoundError("模型文件尚未上传到 GitHub 仓库根目录")
    d = models.resnet18(weights=None); d.fc = nn.Linear(d.fc.in_features,4)
    s = models.resnet18(weights=None); s.fc = nn.Linear(s.fc.in_features,5)
    ds = torch.load(DISEASE_MODEL_PATH,map_location=DEVICE)
    ss = torch.load(SEVERITY_MODEL_PATH,map_location=DEVICE)
    if isinstance(ds,dict) and "model_state_dict" in ds: ds=ds["model_state_dict"]
    if isinstance(ss,dict) and "model_state_dict" in ss: ss=ss["model_state_dict"]
    d.load_state_dict(ds); s.load_state_dict(ss)
    d.to(DEVICE).eval(); s.to(DEVICE).eval()
    return d,s

def risk_level(n): return {4:"高风险",3:"较高风险",2:"中等风险",1:"较低风险",0:"低风险"}[n]

def predict(img):
    d,s=load_models(); x=transform(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): dp=torch.softmax(d(x),1)[0]; sp=torch.softmax(s(x),1)[0]
    dc,di=torch.max(dp,0); sc,si=torch.max(sp,0); key=CLASSES[di.item()]; sev=si.item()
    return {"disease":CLASS_NAMES[key],"confidence":dc.item()*100,"severity":SEVERITY_NAMES[sev],"severity_conf":sc.item()*100,"risk":risk_level(sev),"probs":{CLASS_NAMES[c]:dp[i].item()*100 for i,c in enumerate(CLASSES)}}

def save_history(name,r):
    exists=HISTORY_FILE.exists()
    with HISTORY_FILE.open("a",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f)
        if not exists: w.writerow(HEADERS)
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),name,r["disease"],f'{r["confidence"]:.2f}',r["severity"],f'{r["severity_conf"]:.2f}',r["risk"]])

def history():
    if not HISTORY_FILE.exists(): return pd.DataFrame(columns=HEADERS)
    try: return pd.read_csv(HISTORY_FILE,encoding="utf-8-sig")
    except Exception: return pd.DataFrame(columns=HEADERS)

st.title("☕ CoffeeGuard AI")
st.caption("咖啡园病虫害智能巡检与安全 IPM 决策平台 · Web v1.0")
with st.sidebar:
    page=st.radio("功能导航",["单图智能诊断","快速批量巡检","数据驾驶舱","识别历史"])
    st.divider(); st.caption(f"推理设备：{DEVICE}")
    st.warning("AI置信度 ≠ 经独立真实标签测试集验证的 Accuracy。")

if page=="单图智能诊断":
    st.subheader("单图智能诊断")
    f=st.file_uploader("上传咖啡叶图片",type=["jpg","jpeg","png","bmp","webp"])
    if f:
        img=Image.open(f).convert("RGB"); a,b=st.columns([1,1])
        with a: st.image(img,caption=f.name,use_container_width=True)
        with b:
            try:
                with st.spinner("AI 正在识别..."): r=predict(img)
                save_history(f.name,r)
                st.success(f'识别结果：{r["disease"]}')
                c1,c2=st.columns(2); c1.metric("AI置信度",f'{r["confidence"]:.2f}%'); c2.metric("严重程度",r["severity"])
                c3,c4=st.columns(2); c3.metric("综合风险",r["risk"]); c4.metric("严重度AI置信度",f'{r["severity_conf"]:.2f}%')
                st.bar_chart(pd.Series(r["probs"]))
            except Exception as e: st.error(str(e))
        st.info("本系统为安全优先的IPM决策辅助工具。AI图片结果不能替代现场诊断，也不能单独决定是否施药。")
        st.markdown("**安全 IPM 原则：** 优先现场复核、田间卫生、通风透光、树势管理和持续监测。涉及药剂干预时，应以当地合法登记、最新产品标签、当地法规及专业植保人员意见为准；系统不自动给出农药剂量、浓度、混配比例或采收安全间隔。")

elif page=="快速批量巡检":
    st.subheader("⚡ 快速批量巡检")
    fs=st.file_uploader("一次选择多张咖啡叶图片",type=["jpg","jpeg","png","bmp","webp"],accept_multiple_files=True)
    if fs and st.button("开始批量识别",type="primary"):
        rows=[]; bar=st.progress(0)
        for i,f in enumerate(fs,1):
            try:
                r=predict(Image.open(f).convert("RGB")); save_history(f.name,r)
                rows.append({"图片":f.name,"识别结果":r["disease"],"AI置信度(%)":round(r["confidence"],2),"严重程度":r["severity"],"综合风险等级":r["risk"]})
            except Exception as e: rows.append({"图片":f.name,"识别结果":"识别失败","AI置信度(%)":"","严重程度":"","综合风险等级":str(e)})
            bar.progress(i/len(fs))
        df=pd.DataFrame(rows); st.dataframe(df,use_container_width=True,hide_index=True)
        st.download_button("下载本次巡检 CSV",df.to_csv(index=False,encoding="utf-8-sig").encode("utf-8-sig"),"CoffeeGuard_batch_results.csv","text/csv")

elif page=="数据驾驶舱":
    st.subheader("📊 数据驾驶舱"); df=history(); valid=df[df["识别结果"].isin(CLASS_NAMES.values())].copy() if not df.empty else df
    total=len(valid); high=int(valid["综合风险等级"].isin(["高风险","较高风险"]).sum()) if total else 0; conf=pd.to_numeric(valid["AI置信度(%)"],errors="coerce").mean() if total else 0
    c1,c2,c3=st.columns(3); c1.metric("累计有效识别",total); c2.metric("高 / 较高风险",high); c3.metric("平均 AI 置信度",f"{conf:.1f}%" if total else "0.0%")
    if total:
        l,r=st.columns(2)
        with l: st.write("**病害类别分布**"); st.bar_chart(valid["识别结果"].value_counts())
        with r: st.write("**综合风险态势**"); st.bar_chart(valid["综合风险等级"].value_counts())
    else: st.info("暂无识别历史。")
    st.caption("这里展示的是AI识别历史与AI置信度，不代表独立真实标签测试集验证得到的模型Accuracy。")

else:
    st.subheader("🕘 识别历史"); df=history()
    if df.empty: st.info("暂无识别历史。")
    else:
        st.dataframe(df.iloc[::-1],use_container_width=True,hide_index=True)
        st.download_button("下载完整历史 CSV",df.to_csv(index=False,encoding="utf-8-sig").encode("utf-8-sig"),"recognition_history.csv","text/csv")

st.divider(); st.caption("CoffeeGuard AI Web v1.0 · 安全优先的咖啡病虫害 IPM 决策辅助系统")
