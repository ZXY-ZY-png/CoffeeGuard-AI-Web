import csv
from datetime import datetime
from io import BytesIO
from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

BASE_DIR=Path(__file__).resolve().parent
DISEASE_MODEL_PATH=BASE_DIR/"coffee_disease_model.pth"; SEVERITY_MODEL_PATH=BASE_DIR/"coffee_severity_model.pth"; HISTORY_FILE=BASE_DIR/"recognition_history.csv"
CLASSES=["miner","rust","phoma","cercospora"]; CLASS_NAMES={"miner":"咖啡潜叶虫","rust":"咖啡叶锈病","phoma":"Phoma 病","cercospora":"Cercospora 病"}; SEVERITY_NAMES={0:"0级",1:"1级",2:"2级",3:"3级",4:"4级"}
HEADERS=["时间","图片","识别结果","AI置信度(%)","严重程度","严重程度置信度(%)","综合风险等级"]; DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
st.set_page_config(page_title="CoffeeGuard AI",page_icon="☕",layout="wide",initial_sidebar_state="expanded")
st.markdown("""<style>.block-container{padding-top:2rem;max-width:1450px}.cg-hero{padding:24px 28px;border-radius:18px;background:linear-gradient(120deg,#123c2e,#21634b);color:white;margin-bottom:20px}.cg-hero h1{margin:0;font-size:38px}.cg-hero p{margin:8px 0 0;color:#dcebe5}.stMetric{border:1px solid #e4ebe7;border-radius:14px;padding:14px;background:#fbfdfc}div[data-testid="stSidebar"]{background:#f4f8f6}.safe{padding:14px 16px;border-radius:12px;background:#f5f8e8;border-left:5px solid #7a8f36}</style>""",unsafe_allow_html=True)

@st.cache_resource
def load_models():
    if not DISEASE_MODEL_PATH.exists() or not SEVERITY_MODEL_PATH.exists(): raise FileNotFoundError("模型文件未找到，请检查 Git LFS 模型文件。")
    d=models.resnet18(weights=None); d.fc=nn.Linear(d.fc.in_features,4); s=models.resnet18(weights=None); s.fc=nn.Linear(s.fc.in_features,5)
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
    plans={"miner":["重点复核叶片潜道、虫体或虫粪等特征，扩大抽样范围确认发生程度。","及时清理严重受害叶片和园内病残组织，改善通风透光并持续观察新叶。","优先保护和利用自然天敌，避免不必要的广谱干预。"],"rust":["现场复核叶片背面是否存在典型锈色孢子堆，并检查相邻植株。","加强园内通风、合理遮阴和树势管理，及时处理明显病叶。","结合当地咖啡叶锈病监测与专业植保建议持续巡查。"],"phoma":["复核病斑边缘、扩展状态及相邻叶片症状，排除机械损伤等混淆因素。","清理病残组织，降低叶面长期潮湿，改善通风和园区卫生。","记录病斑变化，在扩展明显时提高人工调查优先级。"],"cercospora":["复核斑点形态、中心与边缘颜色，并检查同株及周边叶片。","改善营养、通风和水分管理，减少植株胁迫并清理严重病叶。","持续记录发生范围和严重度变化，必要时请专业人员复核。"]}
    return plans[key]+[f"当前系统风险分级为“{risk}”，该分级用于安排现场复核优先级，不等同于专家确诊。","如考虑药剂干预，必须先完成现场核实，并遵循当地合法登记、最新产品标签、当地法规、采收安全间隔及专业植保人员意见；本系统不自动提供药剂名称、剂量、浓度或混配比例。"]
def init_tasks():
    if "formal_tasks" not in st.session_state: st.session_state.formal_tasks=[]
def task_summary(t):
    rows=t.get("rows",[]); ok=[x for x in rows if x.get("识别结果")!="失败"]; high=sum(x.get("综合风险") in ["高风险","较高风险"] for x in ok); major=pd.Series([x.get("识别结果") for x in ok]).value_counts().index[0] if ok else "—"; return len(ok),high,major
def build_task_pdf(t):
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light")); buf=BytesIO(); doc=SimpleDocTemplate(buf,pagesize=A4,rightMargin=16*mm,leftMargin=16*mm,topMargin=16*mm,bottomMargin=16*mm); styles=getSampleStyleSheet(); title=ParagraphStyle("ct",parent=styles["Title"],fontName="STSong-Light",fontSize=18,leading=24); body=ParagraphStyle("cb",parent=styles["BodyText"],fontName="STSong-Light",fontSize=9,leading=14)
    ok,high,major=task_summary(t); story=[Paragraph("CoffeeGuard AI 正式巡检任务报告",title),Spacer(1,8),Paragraph("本报告为AI辅助巡检结果，不等同于专家现场确诊或独立真实标签测试集验证结果。",body),Spacer(1,10)]
    meta=[["任务名称",t["name"]],["创建时间",t["created"]],["任务状态",t.get("status","待巡检")],["总图片",str(t.get("total",0))],["成功识别",str(ok)],["高/较高风险",str(high)],["主要病害",major]]; tb=Table(meta,colWidths=[38*mm,125*mm]); tb.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),"STSong-Light"),("FONTSIZE",(0,0),(-1,-1),9),("GRID",(0,0),(-1,-1),0.4,colors.grey),("BACKGROUND",(0,0),(0,-1),colors.HexColor("#EAF3EE"))])); story += [tb,Spacer(1,12),Paragraph("巡检结果明细",body)]
    data=[["图片","识别结果","置信度","严重度","风险"]]+[[str(r.get("图片",""))[:26],r.get("识别结果",""),str(r.get("AI置信度(%)","")),r.get("严重程度",""),r.get("综合风险","")] for r in t.get("rows",[])[:120]]; rt=Table(data,colWidths=[58*mm,35*mm,25*mm,22*mm,28*mm],repeatRows=1); rt.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),"STSong-Light"),("FONTSIZE",(0,0),(-1,-1),7),("GRID",(0,0),(-1,-1),0.25,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#DDEDE5"))])); story += [Spacer(1,6),rt,Spacer(1,12),Paragraph("安全原则：严重度和风险等级用于辅助安排现场复核优先级。任何药剂干预均应在现场核实后，遵循当地合法登记、产品最新标签、当地法规、采收安全间隔及专业植保建议。",body)]; doc.build(story); return buf.getvalue()

init_tasks(); st.markdown('<div class="cg-hero"><h1>☕ CoffeeGuard AI</h1><p>咖啡园病虫害智能巡检与安全 IPM 决策平台 · Web v2.1</p></div>',unsafe_allow_html=True)
with st.sidebar:
    st.markdown("### CoffeeGuard AI"); page=st.radio("功能导航",["🏠 平台总览","🍃 单图智能诊断","⚡ 快速批量巡检","🗂️ 正式巡检任务","📊 数据驾驶舱","🕘 识别历史","🛡️ 安全 IPM 中心"]); st.divider(); st.caption(f"AI推理设备：{DEVICE}"); st.caption("病害模型：已接入"); st.caption("严重度模型：已接入"); st.warning("AI置信度 ≠ 经独立真实标签测试集验证的 Accuracy。")

if page=="🏠 平台总览":
    df=history(); valid=df[df["识别结果"].isin(CLASS_NAMES.values())] if not df.empty else df; total=len(valid); high=int(valid["综合风险等级"].isin(["高风险","较高风险"]).sum()) if total else 0; conf=pd.to_numeric(valid["AI置信度(%)"],errors="coerce").mean() if total else 0
    st.subheader("智能巡检平台总览"); a,b,c,d=st.columns(4); a.metric("累计有效识别",total); b.metric("高 / 较高风险",high); c.metric("平均AI置信度",f"{conf:.1f}%" if total else "0.0%"); d.metric("正式任务",len(st.session_state.formal_tasks)); st.markdown("### 核心能力"); c1,c2,c3,c4=st.columns(4); c1.info("**双模型智能诊断**\n\n病虫害类别 + 0–4级严重度"); c2.info("**快速批量巡检**\n\n临时多图识别与CSV导出"); c3.info("**正式巡检任务**\n\n任务管理、统计与专属PDF"); c4.info("**安全 IPM 辅助**\n\n复核、管理、监测与安全原则"); st.markdown("### 系统闭环"); st.success("创建任务 → 上传咖啡叶图像 → 双模型AI识别 → 严重程度评估 → 风险分级 → IPM建议 → 任务报告"); st.caption("Web v2.1 正式任务数据当前保存在本次浏览器会话中；云端重启后不保证长期持久保存。后续版本可接入数据库。")
elif page=="🍃 单图智能诊断":
    st.subheader("🍃 单图智能诊断"); f=st.file_uploader("上传咖啡叶图片",type=["jpg","jpeg","png","bmp","webp"],key="single")
    if f:
        img=Image.open(f).convert("RGB")
        try:
            with st.spinner("双模型正在进行云端CPU推理..."): r=predict(img)
            sig=f"{f.name}|{getattr(f,'size',0)}"
            if st.session_state.get("last_saved")!=sig: save_history(f.name,r); st.session_state.last_saved=sig
            left,right=st.columns([1.05,1]); left.image(img,caption=f.name,use_container_width=True)
            with right:
                st.success(f'识别结果：{r["disease"]}'); a,b=st.columns(2); a.metric("AI置信度",f'{r["confidence"]:.2f}%'); b.metric("严重程度",r["severity"]); c,d=st.columns(2); c.metric("综合风险",r["risk"]); d.metric("严重度AI置信度",f'{r["severity_conf"]:.2f}%'); st.write("**病害类别概率分布**"); st.bar_chart(pd.Series(r["probs"]))
            st.markdown("### 🛡️ 本次安全 IPM 决策建议"); [st.write(f"**{i}.** {x}") for i,x in enumerate(ipm_plan(r["key"],r["risk"]),1)]
        except Exception as e: st.error(f"模型推理失败：{e}")
elif page=="⚡ 快速批量巡检":
    st.subheader("⚡ 快速批量巡检"); st.caption("快速模式：无需创建正式任务，适合临时检查、模型测试和批量识别。"); fs=st.file_uploader("选择多张咖啡叶图片",type=["jpg","jpeg","png","bmp","webp"],accept_multiple_files=True,key="batch")
    if fs and st.button("开始批量识别",type="primary"):
        rows=[]; bar=st.progress(0); status=st.empty()
        for i,f in enumerate(fs,1):
            status.write(f"正在识别 {i}/{len(fs)}：{f.name}")
            try:r=predict(Image.open(f).convert("RGB")); save_history(f.name,r); rows.append({"图片":f.name,"识别结果":r["disease"],"AI置信度(%)":round(r["confidence"],2),"严重程度":r["severity"],"严重度AI置信度(%)":round(r["severity_conf"],2),"综合风险":r["risk"]})
            except Exception as e:rows.append({"图片":f.name,"识别结果":"失败","错误":str(e)})
            bar.progress(i/len(fs))
        status.success(f"批量巡检完成：{len(rows)} 张"); rdf=pd.DataFrame(rows); st.dataframe(rdf,use_container_width=True,hide_index=True); st.download_button("⬇️ 下载本次巡检 CSV",rdf.to_csv(index=False).encode("utf-8-sig"),"CoffeeGuard_batch_results.csv","text/csv")
elif page=="🗂️ 正式巡检任务":
    st.subheader("🗂️ 正式巡检任务中心"); st.caption("正式任务模式：创建任务 → 批量AI巡检 → 风险统计 → 查看结果 → 下载任务专属PDF。")
    with st.expander("➕ 创建新任务",expanded=not st.session_state.formal_tasks):
        name=st.text_input("任务名称",value=f"咖啡园巡检_{datetime.now().strftime('%Y%m%d_%H%M')}"); note=st.text_input("任务备注（可选）",placeholder="例如：A区雨后巡检")
        if st.button("创建正式巡检任务",type="primary"): st.session_state.formal_tasks.append({"id":datetime.now().strftime("%Y%m%d%H%M%S%f"),"name":name.strip() or "未命名巡检任务","created":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"note":note,"status":"待巡检","total":0,"rows":[]}); st.rerun()
    if not st.session_state.formal_tasks: st.info("暂无正式巡检任务，请先创建任务。")
    else:
        labels=[f'{t["name"]} · {t.get("status","待巡检")} · {t["created"]}' for t in st.session_state.formal_tasks]; idx=st.selectbox("选择任务",range(len(labels)),format_func=lambda i:labels[i]); t=st.session_state.formal_tasks[idx]; ok,high,major=task_summary(t); a,b,c,d=st.columns(4); a.metric("任务图片",t.get("total",0)); b.metric("成功识别",ok); c.metric("高/较高风险",high); d.metric("主要病害",major); st.write(f'**任务状态：** {t.get("status","待巡检")}　　**创建时间：** {t["created"]}')
        fs=st.file_uploader("为当前正式任务选择咖啡叶图片",type=["jpg","jpeg","png","bmp","webp"],accept_multiple_files=True,key=f'taskfiles_{t["id"]}')
        if fs and st.button("▶ 开始正式任务巡检",type="primary"):
            t["total"]=len(fs); t["rows"]=[]; t["status"]="巡检中"; bar=st.progress(0); status=st.empty()
            for i,f in enumerate(fs,1):
                status.write(f"正式任务识别 {i}/{len(fs)}：{f.name}")
                try:r=predict(Image.open(f).convert("RGB")); save_history(f.name,r); t["rows"].append({"图片":f.name,"识别结果":r["disease"],"AI置信度(%)":round(r["confidence"],2),"严重程度":r["severity"],"严重度AI置信度(%)":round(r["severity_conf"],2),"综合风险":r["risk"]})
                except Exception as e:t["rows"].append({"图片":f.name,"识别结果":"失败","AI置信度(%)":"","严重程度":"","严重度AI置信度(%)":"","综合风险":"","错误":str(e)})
                bar.progress(i/len(fs))
            t["status"]="已完成"; status.success("正式巡检任务已完成。"); st.rerun()
        if t.get("rows"):
            rdf=pd.DataFrame(t["rows"]); st.markdown("### 任务结果"); st.dataframe(rdf,use_container_width=True,hide_index=True); x,y=st.columns(2); x.download_button("⬇️ 下载任务 CSV",rdf.to_csv(index=False).encode("utf-8-sig"),f'{t["name"]}_巡检结果.csv',"text/csv"); y.download_button("📄 下载任务专属 PDF",build_task_pdf(t),f'{t["name"]}_巡检报告.pdf',"application/pdf")
        if st.button("🗑 删除当前任务"): st.session_state.formal_tasks.pop(idx); st.rerun()
elif page=="📊 数据驾驶舱":
    st.subheader("📊 CoffeeGuard AI · 数据驾驶舱"); df=history(); valid=df[df["识别结果"].isin(CLASS_NAMES.values())].copy() if not df.empty else df; total=len(valid); high=int(valid["综合风险等级"].isin(["高风险","较高风险"]).sum()) if total else 0; conf=pd.to_numeric(valid["AI置信度(%)"],errors="coerce").mean() if total else 0; a,b,c,d=st.columns(4); a.metric("累计有效识别",total); b.metric("高 / 较高风险",high); c.metric("平均 AI 置信度",f"{conf:.1f}%" if total else "0.0%"); d.metric("正式巡检任务",len(st.session_state.formal_tasks))
    if total:
        l,r=st.columns(2)
        with l: st.write("**病害类别分布**"); st.bar_chart(valid["识别结果"].value_counts())
        with r: st.write("**综合风险态势**"); st.bar_chart(valid["综合风险等级"].value_counts())
        st.write("**严重程度分布**"); st.bar_chart(valid["严重程度"].value_counts().sort_index())
    if st.session_state.formal_tasks:
        td=[]
        for t in st.session_state.formal_tasks:
            ok,h,m=task_summary(t); td.append({"任务名称":t["name"],"状态":t.get("status"),"总图片":t.get("total",0),"已识别":ok,"高/较高风险":h,"主要病害":m})
        st.markdown("### 正式任务进度"); st.dataframe(pd.DataFrame(td),use_container_width=True,hide_index=True)
    st.caption("驾驶舱展示的是AI识别历史与AI置信度，不代表独立真实标签测试集验证得到的模型Accuracy。")
elif page=="🕘 识别历史":
    st.subheader("🕘 识别历史"); df=history()
    if df.empty: st.info("暂无识别历史。")
    else:
        st.dataframe(df.iloc[::-1],use_container_width=True,hide_index=True); st.download_button("⬇️ 下载完整历史 CSV",df.to_csv(index=False).encode("utf-8-sig"),"recognition_history.csv","text/csv")
        if st.button("🧹 清空当前云端历史"): pd.DataFrame(columns=HEADERS).to_csv(HISTORY_FILE,index=False,encoding="utf-8-sig"); st.rerun()
else:
    st.subheader("🛡️ 安全 IPM 决策中心"); st.markdown("""<div class="safe"><b>系统原则</b><br>AI负责辅助识别、严重度评估和风险分级，不自动替代现场诊断，也不自动“开农药”。优先采用现场复核、田间卫生、通风透光、树势管理、生物防治与持续监测。</div>""",unsafe_allow_html=True); st.markdown("### 四类病虫害决策知识")
    for k in CLASSES:
        with st.expander(CLASS_NAMES[k]):
            for i,x in enumerate(ipm_plan(k,"由本次识别结果确定"),1): st.write(f"{i}. {x}")
    st.warning("药剂干预必须基于现场核实，并遵循当地合法登记、产品最新标签、当地法规、采收安全间隔和专业植保建议。本平台不自动提供药剂名称、剂量、浓度、混配比例或施用次数。")
st.divider(); st.caption("CoffeeGuard AI Web v2.1 · 正式巡检任务中心 · 安全优先的咖啡病虫害 IPM 决策辅助平台 · AI置信度不等同于真实Accuracy")