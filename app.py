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

transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
st.set_page_config(page_title="CoffeeGuard AI", page_icon="☕", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1450px}.cg-hero{padding:24px 28px;border-radius:18px;background:linear-gradient(120deg,#123c2e,#21634b);color:white;margin-bottom:20px}.cg-hero h1{margin:0;font-size:38px}.cg-hero p{margin:8px 0 0;color:#dcebe5}.cg-card{border:1px solid #e6ebe8;border-radius:14px;padding:16px;background:white}.stMetric{border:1px solid #e4ebe7;border-radius:14px;padding:14px;background:#fbfdfc}div[data-testid="stSidebar"]{background:#f4f8f6}.safe{padding:14px 16px;border-radius:12px;background:#f5f8e8;border-left:5px solid #7a8f36}
</style>""",unsafe_allow_html=True)

@st.cache_resource
def load_models():
    if not DISEASE_MODEL_PATH.exists() or not SEVERITY_MODEL_PATH.exists(): raise FileNotFoundError("模型文件未找到，请检查 Git LFS 模型文件。")
    d=models.resnet18(weights=None); d.fc=nn.Linear(d.fc.in_features,4)
    s=models.resnet18(weights=None); s.fc=nn.Linear(s.fc.in_features,5)
    ds=torch.load(DISEASE_MODEL_PATH,map_location=DEVICE); ss=torch.load(SEVERITY_MODEL_PATH,map_location=DEVICE)
    if isinstance(ds,dict) and "model_state_dict" in ds: ds=ds["model_state_dict"]
    if isinstance(ss,dict) and "model_state_dict" in ss: ss=ss["model_state_dict"]
    d.load_state_dict(ds); s.load_state_dict(ss); d.to(DEVICE).eval(); s.to(DEVICE).eval(); return d,s

def risk_level(n): return {4:"高风险",3:"较高风险",2:"中等风险",1:"较低风险",0:"低风险"}[n]
def predict(img):
    d,s=load_models(); x=transform(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): dp=torch.softmax(d(x),1)[0]; sp=torch.softmax(s(x),1)[0]
    dc,di=torch.max(dp,0); sc,si=torch.max(sp,0); key=CLASSES[di.item()]; sev=si.item()
    return {"key":key,"disease":CLASS_NAMES[key],"confidence":dc.item()*100,"severity":SEVERITY_NAMES[sev],"severity_conf":sc.item()*100,"risk":risk_level(sev),"probs":{CLASS_NAMES[c]:dp[i].item()*100 for i,c in enumerate(CLASSES)}}
def save_history(name,r):
    exists=HISTORY_FILE.exists()
    with HISTORY_FILE.open("a",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f)
        if not exists:w.writerow(HEADERS)
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),name,r["disease"],f'{r["confidence"]:.2f}',r["severity"],f'{r["severity_conf"]:.2f}',r["risk"]])
def history():
    if not HISTORY_FILE.exists(): return pd.DataFrame(columns=HEADERS)
    try:return pd.read_csv(HISTORY_FILE,encoding="utf-8-sig")
    except:return pd.DataFrame(columns=HEADERS)
def ipm_plan(key,risk):
    plans={
      "miner":["重点复核叶片潜道、虫体或虫粪等特征，扩大抽样范围确认发生程度。","及时清理严重受害叶片和园内病残组织，改善通风透光并持续观察新叶。","优先保护和利用自然天敌，避免不必要的广谱干预。"],
      "rust":["现场复核叶片背面是否存在典型锈色孢子堆，并检查相邻植株。","加强园内通风、合理遮阴和树势管理，及时处理明显病叶。","结合当地咖啡叶锈病监测与专业植保建议持续巡查。"],
      "phoma":["复核病斑边缘、扩展状态及相邻叶片症状，排除机械损伤等混淆因素。","清理病残组织，降低叶面长期潮湿，改善通风和园区卫生。","记录病斑变化，在扩展明显时提高人工调查优先级。"],
      "cercospora":["复核斑点形态、中心与边缘颜色，并检查同株及周边叶片。","改善营养、通风和水分管理，减少植株胁迫并清理严重病叶。","持续记录发生范围和严重度变化，必要时请专业人员复核。"]}
    a=plans[key]
    return a+[f"当前系统风险分级为“{risk}”，该分级用于安排现场复核优先级，不等同于专家确诊。","如考虑药剂干预，必须先完成现场核实，并遵循当地合法登记、最新产品标签、当地法规、采收安全间隔及专业植保人员意见；本系统不自动提供药剂名称、剂量、浓度或混配比例。"]

st.markdown('<div class="cg-hero"><h1>☕ CoffeeGuard AI</h1><p>咖啡园病虫害智能巡检与安全 IPM 决策平台 · Web v2.0</p></div>',unsafe_allow_html=True)
with st.sidebar:
    st.markdown("### CoffeeGuard AI")
    page=st.radio("功能导航",["🏠 平台总览","🍃 单图智能诊断","⚡ 快速批量巡检","📊 数据驾驶舱","🕘 识别历史","🛡️ 安全 IPM 中心"])
    st.divider(); st.caption(f"AI推理设备：{DEVICE}"); st.caption("病害模型：已接入"); st.caption("严重度模型：已接入")
    st.warning("AI置信度 ≠ 经独立真实标签测试集验证的 Accuracy。")

if page=="🏠 平台总览":
    df=history(); valid=df[df["识别结果"].isin(CLASS_NAMES.values())] if not df.empty else df
    total=len(valid); high=int(valid["综合风险等级"].isin(["高风险","较高风险"]).sum()) if total else 0
    conf=pd.to_numeric(valid["AI置信度(%)"],errors="coerce").mean() if total else 0
    st.subheader("智能巡检平台总览")
    a,b,c,d=st.columns(4); a.metric("累计有效识别",total); b.metric("高 / 较高风险",high); c.metric("平均AI置信度",f"{conf:.1f}%" if total else "0.0%"); d.metric("在线AI模型","2 个")
    st.markdown("### 核心能力")
    c1,c2,c3=st.columns(3)
    c1.info("**双模型智能诊断**\n\n病虫害类别识别 + 0–4级严重程度评估")
    c2.info("**快速批量巡检**\n\n多图片批量推理、风险汇总、CSV结果导出")
    c3.info("**安全 IPM 决策辅助**\n\n现场复核、田间管理、生物防治与安全干预原则")
    st.markdown("### 系统闭环")
    st.success("咖啡叶图像 → 病虫害AI识别 → 严重程度AI评估 → 综合风险分级 → 安全IPM建议 → 历史与驾驶舱")
    st.caption("Web v2.0 已实现公网浏览器访问。正式任务中心、持久化断点续检和任务专属PDF属于后续云端增强模块。")

elif page=="🍃 单图智能诊断":
    st.subheader("🍃 单图智能诊断")
    f=st.file_uploader("上传咖啡叶图片",type=["jpg","jpeg","png","bmp","webp"],key="single")
    if f:
        img=Image.open(f).convert("RGB")
        try:
            with st.spinner("双模型正在进行云端CPU推理..."): r=predict(img)
            sig=f"{f.name}|{getattr(f,'size',0)}"
            if st.session_state.get("last_saved")!=sig: save_history(f.name,r); st.session_state.last_saved=sig
            left,right=st.columns([1.05,1])
            with left: st.image(img,caption=f.name,use_container_width=True)
            with right:
                st.success(f'识别结果：{r["disease"]}')
                a,b=st.columns(2); a.metric("AI置信度",f'{r["confidence"]:.2f}%'); b.metric("严重程度",r["severity"])
                c,d=st.columns(2); c.metric("综合风险",r["risk"]); d.metric("严重度AI置信度",f'{r["severity_conf"]:.2f}%')
                st.write("**病害类别概率分布**"); st.bar_chart(pd.Series(r["probs"]))
            st.markdown("### 🛡️ 本次安全 IPM 决策建议")
            for i,x in enumerate(ipm_plan(r["key"],r["risk"]),1): st.write(f"**{i}.** {x}")
        except Exception as e: st.error(f"模型推理失败：{e}")

elif page=="⚡ 快速批量巡检":
    st.subheader("⚡ 快速批量巡检")
    st.caption("快速模式：无需创建正式任务，适合临时检查、模型测试和批量识别。")
    fs=st.file_uploader("选择多张咖啡叶图片",type=["jpg","jpeg","png","bmp","webp"],accept_multiple_files=True,key="batch")
    if fs and st.button("开始批量识别",type="primary"):
        rows=[]; bar=st.progress(0); status=st.empty()
        for i,f in enumerate(fs,1):
            status.write(f"正在识别 {i}/{len(fs)}：{f.name}")
            try:
                r=predict(Image.open(f).convert("RGB")); save_history(f.name,r)
                rows.append({"图片":f.name,"识别结果":r["disease"],"AI置信度(%)":round(r["confidence"],2),"严重程度":r["severity"],"严重度AI置信度(%)":round(r["severity_conf"],2),"综合风险":r["risk"]})
            except Exception as e: rows.append({"图片":f.name,"识别结果":"失败","错误":str(e)})
            bar.progress(i/len(fs))
        status.success(f"批量巡检完成：{len(rows)} 张")
        df=pd.DataFrame(rows); st.dataframe(df,use_container_width=True,hide_index=True)
        if "综合风险" in df:
            a,b,c=st.columns(3); a.metric("本次图片",len(df)); b.metric("高/较高风险",int(df["综合风险"].isin(["高风险","较高风险"]).sum())); c.metric("识别成功",int((df["识别结果"]!="失败").sum()))
        st.download_button("⬇️ 下载本次巡检 CSV",df.to_csv(index=False).encode("utf-8-sig"),"CoffeeGuard_batch_results.csv","text/csv")

elif page=="📊 数据驾驶舱":
    st.subheader("📊 CoffeeGuard AI · 数据驾驶舱"); df=history(); valid=df[df["识别结果"].isin(CLASS_NAMES.values())].copy() if not df.empty else df
    total=len(valid); high=int(valid["综合风险等级"].isin(["高风险","较高风险"]).sum()) if total else 0; conf=pd.to_numeric(valid["AI置信度(%)"],errors="coerce").mean() if total else 0
    a,b,c,d=st.columns(4); a.metric("累计有效识别",total); b.metric("高 / 较高风险",high); c.metric("平均 AI 置信度",f"{conf:.1f}%" if total else "0.0%"); d.metric("病害类别","4 类")
    if total:
        l,r=st.columns(2)
        with l: st.write("**病害类别分布**"); st.bar_chart(valid["识别结果"].value_counts())
        with r: st.write("**综合风险态势**"); st.bar_chart(valid["综合风险等级"].value_counts())
        st.write("**严重程度分布**"); st.bar_chart(valid["严重程度"].value_counts().sort_index())
    else: st.info("暂无识别历史。")
    st.caption("驾驶舱展示的是AI识别历史与AI置信度，不代表独立真实标签测试集验证得到的模型Accuracy。")

elif page=="🕘 识别历史":
    st.subheader("🕘 识别历史"); df=history()
    if df.empty: st.info("暂无识别历史。")
    else:
        st.dataframe(df.iloc[::-1],use_container_width=True,hide_index=True)
        st.download_button("⬇️ 下载完整历史 CSV",df.to_csv(index=False).encode("utf-8-sig"),"recognition_history.csv","text/csv")
        if st.button("🧹 清空当前云端历史"):
            pd.DataFrame(columns=HEADERS).to_csv(HISTORY_FILE,index=False,encoding="utf-8-sig"); st.success("当前云端实例历史已清空。"); st.rerun()

else:
    st.subheader("🛡️ 安全 IPM 决策中心")
    st.markdown("""<div class="safe"><b>系统原则</b><br>AI负责辅助识别、严重度评估和风险分级，不自动替代现场诊断，也不自动“开农药”。优先采用现场复核、田间卫生、通风透光、树势管理、生物防治与持续监测。</div>""",unsafe_allow_html=True)
    st.markdown("### 四类病虫害决策知识")
    for k in CLASSES:
        with st.expander(CLASS_NAMES[k]):
            for i,x in enumerate(ipm_plan(k,"由本次识别结果确定"),1): st.write(f"{i}. {x}")
    st.warning("药剂干预必须基于现场核实，并遵循当地合法登记、产品最新标签、当地法规、采收安全间隔和专业植保建议。本平台不自动提供药剂名称、剂量、浓度、混配比例或施用次数。")

st.divider(); st.caption("CoffeeGuard AI Web v2.0 · 安全优先的咖啡病虫害 IPM 决策辅助平台 · AI置信度不等同于真实Accuracy")