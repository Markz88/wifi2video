from imports.data_manager import load_data
from nets.gan import GAN
import yaml
import os
import shutil
import json
import streamlit as st

# Silhouette synthesis function
def synthesize_silh(opt):
    test = load_data(opt, usecase)  # Load test set signals
    model = GAN(opt, usecase, test_set=test)
    ssim_value, time_value = model.test()
    return round(ssim_value, 2), round(time_value, 2)

# Skeleton synthesis function
def synthesize_skel(opt):
    test = load_data(opt, usecase)  # Load test set signals
    model = GAN(opt, usecase, test_set=test)
    ssim_value, time_value = model.test()
    return round(ssim_value, 2), round(time_value, 2)

# Read configuration files
with open("config_silh.yaml", 'r') as stream:
    opt_silh = yaml.safe_load(stream)

with open("config_skel.yaml", 'r') as stream:
    opt_skel = yaml.safe_load(stream)

# Streamlit Page Configuration
st.set_page_config(page_title="Wi-Fi Video Synthesis", page_icon="✅", layout="wide")

# Initialize session state for metrics
if "silhouette_metrics" not in st.session_state:
    st.session_state["silhouette_metrics"] = None
if "skeleton_metrics" not in st.session_state:
    st.session_state["skeleton_metrics"] = None

size = 300 # image size

# Initialize metrics dictionary
# Load metrics from log if it exists, otherwise initialize
if os.path.exists('log.json'):
    with open('log.json', 'r') as log:
        metrics = json.load(log)

    st.session_state["silhouette_metrics"] = {
        "usecase": 1,
        "SSIM": str(metrics['1'][opt_silh['experiment']][0]),
        "Time": str(metrics['1'][opt_silh['experiment']][1]),
    }
    
    st.session_state["skeleton_metrics"] = {
        "usecase": 1,
        "SSIM": str(metrics['1'][opt_skel['experiment']][0]),
        "Time": str(metrics['1'][opt_skel['experiment']][1]),
    }
else:
    metrics = {
        '1':{
            opt_silh['experiment']:[0.0,0.0],
            opt_skel['experiment']:[0.0,0.0]
        },
        '2':{
            opt_silh['experiment']:[0.0,0.0],
            opt_skel['experiment']:[0.0,0.0]
        },
        '3':{
            opt_silh['experiment']:[0.0,0.0],
            opt_skel['experiment']:[0.0,0.0]
        }
    }
    
    with open('log.json','w') as log:
            json.dump(metrics,log)


    st.session_state["silhouette_metrics"] = {
        "usecase": 1,
        "SSIM": str(metrics['1'][opt_silh['experiment']][0]),
        "Time": str(metrics['1'][opt_silh['experiment']][1]),
    }
    
    st.session_state["skeleton_metrics"] = {
        "usecase": 1,
        "SSIM": str(metrics['1'][opt_skel['experiment']][0]),
        "Time": str(metrics['1'][opt_skel['experiment']][1]),
    }

# Layout
col1, col2, col3, _, col4 = st.columns([0.21, 0.21, 0.21, 0.035, 0.15])

# Control Panel (Buttons and Use Case Selection)
with col4:
    st.header("Control Panel")
    usecase = st.selectbox('Use Case #', ['1', '2', '3'])
    SynthSil = st.button("Synthesize Silhouette")
    SynthSkel = st.button("Synthesize Skeleton")
    ResetMetrics = st.button("Initialize WebApp")

    if ResetMetrics:
        #st.session_state["silhouette_metrics"] = None
        #st.session_state["skeleton_metrics"] = None
        metrics = {
            '1':{
                opt_silh['experiment']:[0.0,0.0],
                opt_skel['experiment']:[0.0,0.0]
            },
            '2':{
                opt_silh['experiment']:[0.0,0.0],
                opt_skel['experiment']:[0.0,0.0]
            },
            '3':{
                opt_silh['experiment']:[0.0,0.0],
                opt_skel['experiment']:[0.0,0.0]
            }
        }
        

        with open('log.json','w') as log:
            json.dump(metrics,log)

        # Remove output directories
        if os.path.exists(os.path.join(opt_silh['output_dir'])):
            shutil.rmtree(os.path.join(opt_silh['output_dir']))

    if SynthSil:
        ssim_value, time_value = synthesize_silh(opt_silh)
        st.session_state["silhouette_metrics"] = {
            "usecase": usecase,
            "SSIM": ssim_value,
            "Time": time_value,
        }
        # Save metrics
        metrics[usecase][opt_silh['experiment']] = [round(float(ssim_value),2), round(float(time_value),2)]
        with open('log.json','w') as log:
            json.dump(metrics,log)

    if SynthSkel:
        ssim_value, time_value = synthesize_skel(opt_skel)
        st.session_state["skeleton_metrics"] = {
            "usecase": usecase,
            "SSIM": ssim_value,
            "Time": time_value,
        }
        # Save metrics
        metrics[usecase][opt_skel['experiment']] = [round(float(ssim_value), 2), round(float(time_value), 2)]
        with open('log.json','w') as log:
            json.dump(metrics,log)

# Display Metrics for Both Silhouette and Skeleton
with col4:
    upper_right_panel = st.container(border=True)
    lower_right_panel = st.container(border=True)

with upper_right_panel:
    if st.session_state["silhouette_metrics"] and st.session_state["silhouette_metrics"]["usecase"] == usecase:
        st.markdown("**Silhouette Metrics** " + ":green-badge[:material/check: Success]")
        st.text("SSIM: " + str(st.session_state["silhouette_metrics"]["SSIM"]))
        st.markdown("Execution Time (s): " + str(st.session_state["silhouette_metrics"]["Time"]))
    elif st.session_state["silhouette_metrics"] and (os.path.exists('log.json') or st.session_state["silhouette_metrics"]["usecase"] != usecase):
            if metrics[usecase][opt_silh['experiment']][0] == 0.0:
                st.markdown("**Silhouette Metrics** " + ":orange-badge[⚠️ Needs run]")
                st.markdown("SSIM: " + str(metrics[usecase][opt_silh['experiment']][0]))
                st.markdown("Execution Time (s): " + str(metrics[usecase][opt_silh['experiment']][1]))
            else:
                st.markdown("**Silhouette Metrics** " + ":green-badge[:material/check: Success]")
                st.markdown("SSIM: " + str(metrics[usecase][opt_silh['experiment']][0]))
                st.markdown("Execution Time (s): " + str(metrics[usecase][opt_silh['experiment']][1]))
        

with lower_right_panel:
    if st.session_state["skeleton_metrics"] and st.session_state["skeleton_metrics"]["usecase"] == usecase:
        st.markdown("**Skeleton Metrics** " + ":green-badge[:material/check: Success]")
        st.markdown("SSIM: " + str(st.session_state["skeleton_metrics"]["SSIM"]))
        st.markdown("Execution Time (s): " + str(st.session_state["skeleton_metrics"]["Time"]))
    elif st.session_state["skeleton_metrics"] and (os.path.exists('log.json') or st.session_state["skeleton_metrics"]["usecase"] != usecase):
            if metrics[usecase][opt_skel['experiment']][0] == 0.0:
                st.markdown("**Skeleton Metrics** " + ":orange-badge[⚠️ Needs run]")
                st.markdown("SSIM: " + str(metrics[usecase][opt_skel['experiment']][0]))
                st.markdown("Execution Time (s): " + str(metrics[usecase][opt_skel['experiment']][1]))
            else:
                st.markdown("**Skeleton Metrics** " + ":green-badge[:material/check: Success]")
                st.markdown("SSIM: " + str(metrics[usecase][opt_skel['experiment']][0]))
                st.markdown("Execution Time (s): " + str(metrics[usecase][opt_skel['experiment']][1]))
        

# Second column content
with col1:
    st.header("Input Video")

    # Display rgb GIF according to usecase
    col1.image(os.path.join(opt_silh['dataroot'],opt_silh['dataset_name'],usecase,usecase+'.gif'), width=size)
    st.header("Input Wi-Fi Signal")
    # Display signal surface according to usecase
    col1.image(os.path.join(opt_silh['dataroot'],opt_silh['dataset_name'],usecase,usecase+'.png'), width=size)


with col2:
    # Display silhouette GT and synthesis
    if os.path.exists(os.path.join(opt_silh['output_dir'],opt_silh['experiment'],'test','demo',usecase,opt_silh['experiment'] + '_gt.gif')):
        st.header("Silhouette GT")
        col2.image(os.path.join(opt_silh['output_dir'],opt_silh['experiment'],'test','demo/',usecase,opt_silh['experiment'] + '_gt.gif'),width=size)
        st.header("Silhouette Synthesis")
        col2.image(os.path.join(opt_silh['output_dir'],opt_silh['experiment'],'test','demo/',usecase,opt_silh['experiment'] + '_pred.gif'),width=size)

with col3:
    # Display skeleton GT and synthesis
    if os.path.exists(os.path.join(opt_skel['output_dir'],opt_skel['experiment'],'test','demo',usecase,opt_skel['experiment'] + '_gt.gif')):
        st.header("Skeleton GT")
        col3.image(os.path.join(opt_skel['output_dir'],opt_skel['experiment'],'test','demo',usecase,opt_skel['experiment'] + '_gt.gif'),width=size)
        st.header("Skeleton Synthesis")
        col3.image(os.path.join(opt_skel['output_dir'],opt_skel['experiment'],'test','demo',usecase,opt_skel['experiment'] + '_pred.gif'),width=size)

st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("Danilo Avola, Marco Cascio, Luigi Cinque, Alessio Fagioli, and Gian Luca Foresti. “Human silhouette and skeleton video synthesis through Wi‑Fi signals”.  \n In: International Journal of Neural Systems vol. 32 n. 05 (2022), p. 1-20, DOI: 10.1142/S0129065722500150.")
