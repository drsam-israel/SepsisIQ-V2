"""SepsisIQ V2: evidence-only portfolio presentation; no clinical inference."""
import base64
from pathlib import Path
import pandas as pd
import altair as alt
import streamlit as st
import sys
import hashlib

st.set_page_config(page_title="SepsisIQ | Clinical Intelligence", page_icon="❤️", layout="wide", initial_sidebar_state="expanded")
BASE = Path(__file__).resolve().parent
header_b64 = base64.b64encode((BASE / 'assets' / 'approved_header.png').read_bytes()).decode()

st.markdown('''<style>
#MainMenu, footer {visibility:hidden} header[data-testid="stHeader"] {background:transparent;height:2rem}
.block-container {max-width:1650px;padding-top:1.1rem;padding-bottom:2rem;padding-left:1.6rem;padding-right:1.6rem}
[data-testid="stAppViewContainer"]{background:#f2f7fd}
section[data-testid="stSidebar"]{background:linear-gradient(160deg,#04172b,#082c50 65%,#04172b)}
section[data-testid="stSidebar"] *{color:#f4f8ff!important}
section[data-testid="stSidebar"] .stRadio label{padding:9px 7px;border-radius:8px}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){background:#096de5!important}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:#e9f3ff}
h1,h2,h3{color:#082b61!important;letter-spacing:-.02em} h2{font-size:1.43rem!important;margin-top:.35rem} h3{font-size:1.05rem!important}
[data-testid="stMetric"]{background:#fff;border:1px solid #d8e5f5;border-radius:12px;padding:14px 16px;min-height:110px;box-shadow:0 2px 5px #18345a09}
[data-testid="stMetricLabel"]{font-size:.86rem!important;color:#365577!important}
[data-testid="stMetricValue"]{font-size:1.8rem!important;font-weight:800;color:#083a87!important}
.stAlert{border-radius:10px}
div[data-testid="stVerticalBlockBorderWrapper"]{border-color:#d9e6f6!important;border-radius:14px!important;background:white}
[data-testid="stTabs"] button{font-weight:650}
.sepsisiq-banner{border-radius:12px;overflow:hidden;box-shadow:0 4px 16px #061d3b22;margin-bottom:16px}
.sepsisiq-banner img{display:block;width:100%;height:auto}
.page-title{font-size:1.42rem;font-weight:850;color:#082b61;margin:8px 0 1px}
.page-subtitle{font-size:.91rem;color:#315783;margin-bottom:16px}
.footer-wrap{margin-top:26px;padding:15px 19px;border:1px solid #d5e3f3;border-radius:11px;background:#e8f2fc}
.footer-line{display:flex;justify-content:space-between;gap:10px;font-weight:700;color:#082b61}
.footer-sub{font-size:.8rem;color:#54718f;margin-top:5px}
.status-no{color:#bd2339;font-weight:850}
.section-note{font-size:.78rem;color:#627992}
@media(max-width:850px){.footer-line{flex-direction:column}.block-container{padding-left:.7rem;padding-right:.7rem}}
</style>''',unsafe_allow_html=True)

NAV = ['01  Executive Command Center','02  Patient Risk Intelligence','03  Model Performance & Clinical Evidence','04  Clinical Safety & AI Governance','05  Clinical Workflow & Implementation Readiness']
with st.sidebar:
    st.markdown('<div style="font-size:1.65rem;font-weight:800;margin:40px 0 0">❤️ Sepsis<span style="color:#1898ff">IQ</span></div>',unsafe_allow_html=True)
    st.caption('Clinical Intelligence & ICU Safety Platform')
    st.divider()
    page=st.radio('Navigation',NAV,label_visibility='collapsed')
    st.divider()
    st.markdown('<div style="color:#c0d6f2;font-size:.88rem">Evidence-driven AI.<br>Safer patients. Stronger systems.</div>',unsafe_allow_html=True)
    st.markdown('<div style="background:#153f55;border-radius:9px;padding:12px;margin-top:20px;font-size:.84rem">PORTFOLIO DEMONSTRATION<br><b>Not authorized for clinical use</b></div>',unsafe_allow_html=True)

st.markdown(f'<div class="sepsisiq-banner"><img src="data:image/png;base64,{header_b64}" alt="Approved SepsisIQ header with ICU clinical imagery"></div>',unsafe_allow_html=True)

def title(num,name,sub):
    st.markdown(f'<div class="page-title">{num} — {name}</div><div class="page-subtitle">{sub}</div>',unsafe_allow_html=True)

def metrics(items,cols=3):
    for start in range(0,len(items),cols):
        row=st.columns(cols)
        for c,item in zip(row,items[start:start+cols]):
            with c: st.metric(item[1],item[0],help=item[2] if len(item)>2 else None)


def chart(data, x, y, title_, fmt=None):
    df = pd.DataFrame(data)

    st.markdown('**' + title_ + '**')

    # Preserve chronological order for ICU observation history
    if 'History' in df.columns:
        history_order = [
            '0 h',
            '1–5 h',
            '6–11 h',
            '12–23 h',
            '24+ h'
        ]
        df['History'] = pd.Categorical(
            df['History'],
            categories=history_order,
            ordered=True
        )
        df = df.sort_values('History')

    c = alt.Chart(df).mark_bar(
        cornerRadiusEnd=4,
        color='#1777dc'
    ).encode(
        x=alt.X(
            x,
            title=None,
            sort=history_order if 'History' in df.columns else None,
            axis=alt.Axis(labelAngle=0)
        ),
        y=alt.Y(y, title=None),
        tooltip=list(df.columns)
    ).properties(height=215)

    st.altair_chart(c, width='stretch')


def footer():
    st.markdown('''<div class="footer-wrap"><div class="footer-line"><span>SepsisIQ | Clinical Intelligence & ICU Safety Platform</span><span>Dr Samuel Israel</span></div><div class="footer-sub">Professional Portfolio Demonstration · <span class="status-no">Not authorized for clinical use</span></div></div>''',unsafe_allow_html=True)

if page==NAV[0]:
    title('01','Executive Command Center','Verified analytical cohort and retrospective held-out Test alert burden')
    with st.container(border=True):
        st.markdown('### AI-supported ICU sepsis risk intelligence')
        st.write('A clinical AI product demonstration built from frozen retrospective model evidence. No live patient monitoring, clinical alerts or treatment recommendations.')
        st.caption('Evidence status: internal retrospective validation · Operational status: demonstration only')
    metrics([('40,336','Analytical patients'),('1,527,226','Analytical patient-hours'),('0.192%','Positive target patient-hour prevalence'),('8.31','Alerts per 100 patient-hours — Test'),('69.00%','Patients with ≥1 alert — Test'),('NOT AUTHORIZED','Live clinical deployment')])
    c1,c2=st.columns([1.2,1])
    with c1:
        with st.container(border=True):
            st.markdown('### Clinical evidence snapshot')
            metrics([('0.8345','ROC-AUC — Test'),('48.41%','Sensitivity — Test'),('1.12%','PPV — Test')])
            st.caption('At frozen OP07 threshold 0.004 (0.4%). Low PPV signals substantial potential review burden.')
    with c2:
        with st.container(border=True):
            st.markdown('### Deployment decision')
            st.error('NO-GO — Live clinical deployment')
            st.write('Prospective silent validation is conditional on institutional approval and completion of prerequisites.')
            st.caption('16 documented live deployment blockers · 8 defined safety stops')
    st.info('Population denominators differ: analytical cohort metrics are not held-out Test metrics. Positive target rows are patient-hours, not positive patients.')

elif page==NAV[1]:
    title('02','Patient Risk Intelligence','Governed Train-derived demonstration trajectories — frozen B03 inference only')
    st.warning('PORTFOLIO DEMONSTRATION ONLY — NOT AUTHORIZED FOR CLINICAL USE. No live patient input, diagnosis, treatment advice, or held-out Test rescoring.')
    metrics([('B03','Frozen model identifier'),('230','Model predictors'),('0.004','Operating threshold (0.4%)')])

    # The original Sepsis_AI project must be available locally. No model or data
    # artifacts are bundled with this standalone dashboard.
    project_root = Path(__file__).resolve().parent
    demo_dir = project_root / 'streamlit_app' / 'demo_data'
    registry = project_root / 'artifacts' / 'phase03_features' / 'approved_feature_registry.csv'
    manifest_path = demo_dir / 'demo_case_manifest.csv'
    cases = {
        'Below OP07 — Train-derived demonstration': demo_dir / 'demo-low_trajectory.csv',
        'At/above OP07 — Train-derived demonstration': demo_dir / 'demo-review_trajectory.csv',
    }
    required = [registry, manifest_path, *cases.values(),
                project_root / 'streamlit_app' / 'services' / 'prediction_pipeline.py',
                project_root / 'streamlit_app' / 'services' / 'feature_engineering.py',
                project_root / 'streamlit_app' / 'services' / 'frozen_inference.py',
                project_root / 'artifacts' / 'phase04_modeling' / 'B03_frozen_fitted_estimator.pkl']
    missing = [str(f) for f in required if not f.is_file()]
    if missing:
        st.error('Frozen demonstration integration unavailable. No prediction was performed.')
        st.caption('Missing local files (check the original Sepsis_AI project folder):')
        st.code('\n'.join(missing))
    else:
        st.markdown('### Approved demonstration case')
        selected_case = st.selectbox('Select a Train-derived, non-clinical trajectory', list(cases))
        st.caption('Only the two fixed demonstration files can be selected. No uploads or arbitrary patient inputs are accepted.')
        if st.button('Run frozen demonstration', type='primary'):
            try:
                # Import only on explicit user action; frozen_inference verifies
                # the estimator SHA-256 before loading the pickle.
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                from streamlit_app.services.prediction_pipeline import run_longitudinal_prediction

                chosen = cases[selected_case]
                trajectory = pd.read_csv(chosen)
                manifest = pd.read_csv(manifest_path)
                if 'SourcePartition' not in manifest.columns:
                    raise ValueError('Demo manifest has no SourcePartition column; provenance cannot be verified.')
                if manifest.empty or not manifest['SourcePartition'].astype(str).str.strip().str.lower().eq('train').all():
                    raise ValueError('Demo manifest is not exclusively Train-derived.')
                if 'TrajectorySHA256' not in manifest.columns:
                    raise ValueError('Demo manifest has no TrajectorySHA256 column.')
                observed_hash = hashlib.sha256(chosen.read_bytes()).hexdigest()
                if observed_hash not in manifest['TrajectorySHA256'].astype(str).str.strip().tolist():
                    raise ValueError('Demonstration trajectory checksum is not registered in the manifest.')
                if 'PatientID' not in trajectory.columns or 'ICULOS' not in trajectory.columns:
                    raise ValueError('Trajectory is missing PatientID or ICULOS.')
                result = run_longitudinal_prediction(trajectory, registry)
                if result.get('predictor_count') != 230 or abs(result.get('operating_threshold', -1) - 0.004) > 1e-12:
                    raise RuntimeError('Frozen inference contract mismatch.')
                if result.get('deployment_authorization') != 'NOT AUTHORIZED':
                    raise RuntimeError('Unexpected deployment authorization status.')
                st.success('Frozen demonstration completed; source is Train-derived. Not a clinical prediction service.')
                a,b,c = st.columns(3)
                a.metric('Calibrated model score', f"{result['calibrated_probability']:.6f}")
                b.metric('Frozen OP07 threshold', '0.004')
                c.metric('Threshold flag', result['threshold_status'].replace('_',' '))
                st.write('**Demonstration observation hour:**', result['prediction_hour'])
                st.write('**Observed trajectory rows:**', result['observed_history_hours'])
                st.info('A threshold flag is not a sepsis diagnosis, clinical alert, treatment recommendation, or evidence of deployment readiness.')
            except Exception as exc:
                st.error('Frozen demonstration blocked. No result is presented.')
                st.code(f'{type(exc).__name__}: {exc}')
                st.caption('Do not modify the frozen estimator, calibration, threshold or closed held-out Test set to resolve an integration error.')
    with st.expander('Model integration safeguards'):
        st.write('Frozen B03 estimator; exactly 230 approved predictors; frozen logistic calibration; OP07 threshold 0.004; Train-derived demonstration files only; SHA-256 checks; no model fitting, recalibration, threshold changes or held-out Test access.')
        st.caption('The original project services perform feature engineering and model inference. This dashboard does not replicate their logic.')

elif page==NAV[2]:
    title('03','Model Performance & Clinical Evidence','Saved aggregate results from frozen, closed held-out Test evaluation')
    metrics([('0.834478','ROC-AUC'),('0.012340','PR-AUC'),('0.001914','Brier score'),('48.41%','Sensitivity'),('91.77%','Specificity'),('1.12%','Positive predictive value')])
    tabs=st.tabs(['Test evidence & 95% CIs','Set A vs Set B','Temporal safety','Global explainability'])
    with tabs[0]:
        st.dataframe(pd.DataFrame([
            ('ROC-AUC','0.834478','0.817657–0.851561'),('PR-AUC','0.012340','0.010485–0.016619'),('Brier score','0.001914','0.001741–0.002084'),('Sensitivity','48.41%','43.84–52.81%'),('Specificity','91.77%','90.94–92.61%'),('PPV','1.12%','0.99–1.27%'),('Alerts / 100 patient-hours','8.31','7.46–9.14'),('Patient alert exposure','69.00%','67.94–70.22%')
        ],columns=['Metric','Held-out Test','95% patient-cluster bootstrap CI']),hide_index=True,use_container_width=True)
        st.caption('1,000 patient-clustered bootstrap replicates; rounded display. Exact ROC/PR curves are omitted because saved coordinates are not integrated.')
    with tabs[1]:
        st.dataframe(pd.DataFrame([
            ('Patients','3,051','3,000'),('Patient-hours','116,561','111,891'),('ROC-AUC','0.812119','0.853363'),('PR-AUC','0.012361','0.013706'),('Sensitivity','46.47%','51.46%'),('PPV','1.16%','1.08%'),('Alerts / 100 patient-hours','9.27','7.30')
        ],columns=['Measure','Set A','Set B']),hide_index=True,use_container_width=True)
        chart([{'Set':'A','Sensitivity (%)':46.4684},{'Set':'B','Sensitivity (%)':51.4620}],'Set:N','Sensitivity (%):Q','Sensitivity by held-out Test cohort')
    with tabs[2]:
        chart([{'History':'0 h','Sensitivity (%)':96.3636},{'History':'1–5 h','Sensitivity (%)':12.1951},{'History':'6–11 h','Sensitivity (%)':10.4167},{'History':'12–23 h','Sensitivity (%)':11.6667},{'History':'24+ h','Sensitivity (%)':60.5932}],'History:N','Sensitivity (%):Q','Sensitivity by ICU observation history')
        st.warning('At 0 hours of observation history, the false-positive rate is 74.19%. Sensitivity is markedly unstable across history strata.')
    with tabs[3]:
        chart([{'Family':'Physiology','Share (%)':52.77},{'Family':'Care process','Share (%)':39.14},{'Family':'Context / history','Share (%)':8.09}],'Family:N','Share (%):Q','Global feature-family permutation importance')
        st.info('Global, non-causal permutation importance. These values are not individual-patient attributions or causal effects.')

elif page==NAV[3]:
    title('04','Clinical Safety & AI Governance','Documented governance framework — not implemented hospital controls')
    metrics([('15','Documented controls'),('8','Defined safety stops'),('12','Residual limitations'),('21','KPI/KRI definitions'),('19','Readiness domains'),('16','Live deployment blockers')])
    tabs=st.tabs(['System inventory','Governance status','Readiness decision'])
    with tabs[0]:
        st.dataframe(pd.DataFrame([('System','SepsisIQ'),('AI system ID','ANIHG-AI-002'),('Frozen model','B03 — Histogram-Based Gradient Boosting'),('Model predictors','230'),('Operating point','OP07 — 0.004 (0.4%)'),('Intended concept','ICU sepsis risk prediction for clinician review'),('Current product','Non-production portfolio demonstration')],columns=['Field','Recorded description']),hide_index=True,use_container_width=True)
    with tabs[1]:
        st.dataframe(pd.DataFrame([('AI system inventory','Documented'),('Model evidence','Frozen retrospective internal evaluation'),('Risk and safety controls','Defined / documented'),('Monitoring framework','Specified, not operationally validated'),('Safety stops','8 defined'),('Live clinical controls','Not established')],columns=['Governance domain','Status']),hide_index=True,use_container_width=True)
    with tabs[2]:
        st.error('NO-GO for live clinical deployment — not authorized')
        st.write('Controlled prospective silent validation remains conditional on institutional approval, local clinical ownership and prerequisite completion.')
        st.caption('The 19 readiness domains include 18 safety-critical domains. Documented controls do not equal implemented hospital controls.')

else:
    title('05','Clinical Workflow & Implementation Readiness','Proposed human-review workflow grounded in retrospective alert burden')
    st.markdown('### Proposed workflow — not an operational ICU alerting service')
    steps=st.columns(4)
    for c,(icon_,name,detail) in zip(steps,[('🔎','1. Risk assessment','Governed demonstration only'),('🔔','2. Review flag','At frozen threshold 0.004'),('👩‍⚕️','3. Clinician review','Assess clinical context'),('📝','4. Documentation','Record review and monitoring needs')]):
        with c:
            with st.container(border=True):
                st.markdown(f'<div style="font-size:1.5rem;text-align:center">{icon_}</div><div style="font-weight:750;text-align:center;color:#07377a">{name}</div><div style="font-size:.78rem;text-align:center;color:#56708a">{detail}</div>',unsafe_allow_html=True)
    st.markdown('### Observed retrospective alert burden — held-out Test')
    metrics([('8.31','Alerts per 100 patient-hours'),('69.00%','Patients with ≥1 alert'),('8.23%','False-positive rate (negative patient-hours)')])
    with st.container(border=True):
        st.markdown('### Outstanding prerequisites')
        for s in ['Institutional approval and locally defined clinical ownership','Independent validation in the intended care setting','Clinician workflow design and alert-fatigue assessment','Safety, data-quality and performance monitoring implementation','Formal governance authorization before any live clinical use']:
            st.markdown('• '+s)
    st.warning('No mortality reduction, shorter length of stay, financial savings or return on investment has been demonstrated. Proposed workflow is not implemented.')

footer()
