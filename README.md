# Time-Series Work and Conference

# - <a href = "#Conferences">Jump to Conferences page</a>
# Conf Submission Appropriate Time (Recent 7 Years)

> IJCAI : 1.14~2.15

> ICML  : 1.23~2.24

> KDD   : 2.3~2.17

> CIKM  : 5.15~5.26

> NIPS  : 5.18~6.5

> ICDM  : 6.5~6.17

> WSDM  : 7.17~8.16

> AAAI  : 9.5~9.15

> ICLR  : 9.25~10.27

> WWW   : 10.14~11.5

# Deep Learning Models for Time Series Task

- <a href = "#Multivariable-Time-Series-Forecasting">Multivariable Time Series Forecasting</a>
- <a href = "#Multivariable-Probabilistic-Time-Series-Forecasting">Multivariable Probabilistic Time Series Forecasting</a>
- <a href = "#Time-Series-Imputation">Time Series Imputation</a>
- <a href = "#Time-Series-Anomaly-Detection">Time Series Anomaly Detection</a>
- <a href = "#Demand-Prediction">Demand Prediction</a>
- <a href = "#Travel-Time-Estimation">Travel Time Estimation</a>
- <a href = "#Traffic-Location-Prediction">Traffic Location Prediction</a>
- <a href = "#Traffic-Accident-Prediction">Traffic Accident Prediction</a>
- <a href = "#Other-Forecasting">Other Forecasting</a>




# [Multivariable Time Series Forecasting](#content)
|  Task  |    Data |   Model  | Paper   |    Code    |   Publication    |
| :-: | :-: | :-: | :-: | - | - |
| Paper Nums:? | <img width=150/> | <img width=220/>  |   |   |   <img width=300/> | 
| Traffic Speed | NAVER-Seoul <br> METR-LA |         PM-MemNet         | [Learning to Remember Patterns: Pattern Matching Memory Networks for Traffic Forecasting](https://openreview.net/forum?id=wwDg3bbYBIq) | [Pytorch](https://github.com/HyunWookL/PM-MemNet) | ICLR 2022  <br>None But Top 
| Multivariable | PeMSD3 <br> PeMSD4 <br> PeMSD8 <br> COVID-19,etc |         TAMP-S2GCNets         | [TAMP-S2GCNets: Coupling Time-Aware Multipersistence Knowledge Representation with Spatio-Supra Graph Convolutional Networks for Time-Series Forecasting](https://openreview.net/forum?id=wv6g8fWLX2q) | [Pytorch](https://www.dropbox.com/sh/n0ajd5l0tdeyb80/AABGn-ejfV1YtRwjf_L0AOsNa?dl=0) | ICLR 2022 <br> None But Top 
| Multivariable | ETT <br> Electricity <br> Weather |         CoST         | [CoST: Contrastive Learning of Disentangled Seasonal-Trend Representations for Time Series Forecasting](https://openreview.net/forum?id=PilZY3omXV2) | [Pytorch](https://github.com/salesforce/CoST) | ICLR 2022 <br> None But Top 
| Multivariable | Electricity <br> traffic <br> M4 <br> CASIO <br> NP |         DEPTS         | [DEPTS: Deep Expansion Learning for Periodic Time Series Forecasting](https://openreview.net/forum?id=AJAR-JgNw__) | [Pytorch](https://github.com/weifantt/DEPTS) | ICLR 2022 <br> None But Top 
| Multivariable | ETT <br> Electricity <br> Wind <br> App Flow |         Pyraformer     | [Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting](https://openreview.net/forum?id=0EXmFzUn5I) | [Pytorch](https://github.com/alipay/Pyraformer) | ICLR 2022 <br> None But Top 
| Multivariable | ETT <br> ECL  <br> M4 <br> Air Quality <br> Nasdaq |         RevIN     | [Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift](https://openreview.net/forum?id=cGDAkQo1C0p) | [Pytorch](https://github.com/ts-kim/RevIN) | ICLR 2022 <br> None But Top 
| Multivariable | Temperature <br> Cloud cover  <br> Humidity <br> Wind |         CLCRN     | [Conditional Local Convolution for Spatio-temporal Meteorological Forecasting](https://aaai-2022.virtualchair.net/poster_aaai1716) | [Pytorch](https://github.com/BIRD-TAO/CLCRN) | AAAI 2022<br>A
| Traffic Flow | PeMSD3 <br> PeMSD4 <br> PeMSD7 <br> PeMSD8 <br> PeMSD7(M) <br> PeMSD7(L) |         STG-NCDE     | [Graph Neural Controlled Differential Equations for Traffic Forecasting](https://aaai-2022.virtualchair.net/poster_aaai1716) | [Pytorch](https://github.com/jeongwhanchoi/STG-NCDE) | AAAI 2022<br>A
| Multivariable | GT-221 <br> WRS-393 <br> ZGC-564 |         STDEN     | [STDEN: Towards Physics-guided Neural Networks for Traffic Flow Prediction](https://aaai-2022.virtualchair.net/poster_aaai211) | [Pytorch](https://github.com/Echo-Ji/STDEN)   | AAAI 2022<br>A
| Multivariable | Electricity <br> traffic <br> PeMSD7(M) <br> METR-LA  |         CATN     | [CATN: Cross Attentive Tree-Aware Network for Multivariate Time Series Forecasting](https://aaai-2022.virtualchair.net/poster_aaai7403) | None | AAAI 2022<br>A
| Multivariable | ETT <br> Electricity  |         TS2Vec     | [TS2Vec: Towards Universal Representation of Time Series](https://aaai-2022.virtualchair.net/poster_aaai8809) | [Pytorch](https://github.com/yuezhihan/ts2vec) | AAAI 2022<br>A
| Traffic Flow | TaxiBJ <br> BikeNYC |         ST-GSP     | [ST-GSP: Spatial-Temporal Global Semantic Representation Learning for Urban Flow Prediction](https://dl.acm.org/doi/abs/10.1145/3488560.3498444) | [Pytorch](https://github.com/k51/STGSP) | WSDM 2022 <br> B
| Traffic Speed | METR-LA <br> PeMS-Bay <br> Simulated |         STNN     | [Space Meets Time: Local Spacetime Neural Network For Traffic Flow Forecasting](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9679008) | [Pytorch](https://github.com/songyangco/STNN) | ICDM 2021<br>  B
| Traffic Speed | DiDiChengdu <br> DiDiXiAn  |         T-wave     | [Trajectory WaveNet: A Trajectory-Based Model for Traffic Forecasting](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9679147) | [Pytorch](https://github.com/songyangco/STNN) | ICDM 2021<br>  B
| Multivariable | Sanyo <br> Hanergy <br> Solar <br> Electricity  <br> Exchange  |         SSDNet     | [SSDNet: State Space Decomposition Neural Network for Time Series Forecasting](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9679135) | [Pytorch](https://github.com/YangLIN1997/SSDNet-ICDM2021) | ICDM 2021 <br> B
| Traffic Volumn | Hangzhou City <br> Jinan City |         CTVI     | [Temporal Multi-view Graph Convolutional Networks for Citywide Traffic Volume Inference](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9679045) | [Pytorch](https://github.com/dsj96/CTVI-master) | ICDM 2021 <br>  B
| Traffic Volumn | Uber Movements <br>  Grab-Posisi |         TEST-GCN     | [TEST-GCN: Topologically Enhanced Spatial-Temporal Graph Convolutional Networks for Traffic Forecasting](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9679077) | None | ICDM 2021<br> B





# [Multivariable Probabilistic Time Series Forecasting](#content)
|  Task  |    Data |   Model  | Paper   |    Code    |   Publication    |
| :-: | :-: | :-: | :-: | - | - |
| Paper Nums:9 | <img width=150/> | <img width=220/>  |   |   |   <img width=300/> | 
| probability & <br> Point & <br> Others |  UCI <br> VOC <br> electricity  |         AQF        | [Autoregressive Quantile Flows for Predictive Uncertainty Estimation](https://openreview.net/forum?id=z1-I6rOKv1S) | None | ICLR 2022 /<br> None But Top 
| probability  |     |         EMF        | [Embedded-model flows: Combining the inductive biases of model-free deep learning and explicit probabilistic modeling](https://openreview.net/forum?id=9pEJSVfDbba) | [Pytorch](https://github.com/gisilvs/EmbeddedModelFlows) | ICLR 2022 <br> None But Top 
| probability  | Bike Sharing <br> UCI <br> NYU Depth v2  |         NatPN        | [Natural Posterior Network: Deep Bayesian Predictive Uncertainty for Exponential Family Distributions](https://www.in.tum.de/daml/natpn/) | [Pytorch](https://github.com/borchero/natural-posterior-network) | ICLR 2022 <br> None But Top 
| probability & Point | Sichuan <br> Panama |         PrEF        | [PrEF: Probabilistic Electricity Forecasting via Copula-Augmented State Space Model](https://aaai-2022.virtualchair.net/poster_aisi7128) | None | AAAI 2022<br>A
| probability | Exchange <br> Solar <br> Electricity <br> Traffic <br>  Taxi  <br>   Wiki  |        TimeGrad      | [Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting](http://proceedings.mlr.press/v139/rasul21a.html) |  [Pytorch](https://github.com/zalandoresearch/pytorch-ts) | ICML 2021<br>A
| probability & Point | PeMSD3 <br> PeMSD4 <br> PeMSD7 <br> PeMSD8 <br>  Electricity  <br>   Traffic , etc |         AGCGRU        | [RNN with Particle Flow for Probabilistic Spatio-temporal Forecasting](https://proceedings.mlr.press/v139/pal21b.html) |  [TF](https://github.com/networkslab/rnn_flow) | ICML 2021<br>A
| probability | Tourism <br> Labour <br> Traffic <br> Wiki <br>  Electricity  <br>   Traffic , etc |         Hier-E2E        | [End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series](http://proceedings.mlr.press/v139/rangapuram21a.html) |  [MXNet](https://github.com/rshyamsundar/gluonts-hierarchical-ICML-2021) | ICML 2021<br>A
| probability | Exchange <br> Solar <br> Electricity <br> Traffic <br>  Taxi  <br>   Wiki  |        TimeGrad      | [Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting](http://proceedings.mlr.press/v139/rasul21a.html) |  [Pytorch](https://github.com/zalandoresearch/pytorch-ts) | ICML 2021<br>A
| probability | Sine <br> MNIST <br> Billiards <br> S&P <br>  Stock   |        Whittle      | [Whittle Networks: A Deep Likelihood Model for Time Series](http://proceedings.mlr.press/v139/yu21c.html) | [TF](https://github.com/ml-research/WhittleNetworks) | ICML 2021<br>A
| probability & Point | Energy <br> Wine <br> Power <br> MSD, etc |         PGBM        | [Probabilistic Gradient Boosting Machines for Large-Scale Probabilistic Regression](https://dl.acm.org/doi/10.1145/3447548.3467278) |  [Pytorch](https://github.com/elephaint/pgbm) | KDD 2021<br>A
| probability  | More |         More        | [https://github.com/zzw-zwzhang/Awesome-of-Time-Series-Prediction](https://github.com/zzw-zwzhang/Awesome-of-Time-Series-Prediction) |  More |  A
 
 
 



# [Time Series Imputation](#content)
|  Task  |    Data |   Model  | Paper   |    Code    |   Publication    |
| :-: | :-: | :-: | :-: | - | - |
| Paper Nums: 12  | <img width=150/> | <img width=220/>  |   |   |   <img width=300/> | 
| Imputation |  Air Quality <br> METR-LA <br> PeMS-Bay <br> CER-E  |         GRIN        | [Filling the G_ap_s-Multivariate Time Series Imputation by Graph Neural Networks](https://openreview.net/forum?id=kOu3-S3wJ7) |  [Pytorch](https://github.com/Graph-Machine-Learning-Group/grin) | ICLR 2022 <br> None But Top 
| Imputation |  PhysioNet <br> MIMIC-III <br> Climate  |         HeTVAE        | [Heteroscedastic Temporal Variational Autoencoder For Irregularly Sampled Time Series](https://openreview.net/forum?id=Az7opqbQE-3) |  [Pytorch](https://github.com/reml-lab/hetvae) | ICLR 2022 <br> None But Top 
| Imputation |  MIMIC-III <br> OPHTHALMIC <br> MNIST Physionet <br> |         GIL        | [Gradient Importance Learning for Incomplete Observations](https://openreview.net/forum?id=fXHl76nO2AZ) |  [TF](https://github.com/gaoqitong/gradient-importance-learning) | ICLR 2022 <br> None But Top 
| Imputation | Chlorine level <br> SML2010 <br> Air Quality |         D-NLMC        | [Dynamic Nonlinear Matrix Completion for Time-Varying Data Imputation](https://aaai-2022.virtualchair.net/poster_aaai12088) | [Matlab](https://github.com/jicongfan) <br> Author <br> Github | AAAI 2022<br>A
| Imputation | COMPAS <br> Adult <br> HSLS |         ME        | [Online Missing Value Imputation and Change Point Detection with the Gaussian Copula](https://aaai-2022.virtualchair.net/poster_aaai6237) | [gcimpute](https://github.com/yuxuanzhao2295/Online-Missing-Value-Imputation-and-Change-Point-Detection-with-the-Gaussian-Copula) | AAAI 2022<br>A
| Imputation |  |         Fair MIP Forest        | [Fairness without Imputation: A Decision Tree Approach for Fair Prediction with Missing Values](https://aaai-2022.virtualchair.net/poster_aaai6921) | None | AAAI 2022<br>A
| Imputation | Physionet <br> MIMIC-III <br> Human Activity  |         mTAND        | [Multi-Time Attention Networks for Irregularly Sampled Time Series](https://openreview.net/forum?id=4c0J6lwQ4_) | [Pytorch](https://github.com/reml-lab/mTAN)  | ICLR 2021 <br> None But Top 
| Imputation | METR-LA <br> NREL <br> USHCN <br> SeData |         IGNNK        | [Inductive Graph Neural Networks for Spatiotemporal Kriging](https://ojs.aaai.org/index.php/AAAI/article/view/16575) | [Pytorch](https://github.com/Kaimaoge/IGNNK) | AAAI 2021<br>A
| Imputation | Activity  <br> PhysioNet <br> Air Quality |         SSGAN       | [Generative Semi-supervised Learning for Multivariate Time Series Imputation](https://ojs.aaai.org/index.php/AAAI/article/view/17086) | [Pytorch](https://github.com/zjuwuyy-DL/Generative-Semi-supervised-Learning-for-Multivariate-Time-Series-Imputation) | AAAI 2021<br>A
| Imputation | PhysioNet  <br> Air Quality  |         CSDI       | [CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation](https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html) | [Pytorch](https://github.com/ermongroup/CSDI) | NIPS 2021<br>A
| Imputation & <br> Prediction  | VevoMusic  <br> WikiTraffic <br> Los-Loop <br> SZ-Taxi |         Radflow       | [Radflow: A Recurrent, Aggregated, and Decomposable Model for Networks of Time Series](https://dl.acm.org/doi/10.1145/3442381.3449945) | [Pytorch](https://github.com/alasdairtran/radflow) | WWW 2021<br>A
| Imputation | PhysioNet  <br> Air Quality <br> Gas Sensor |         STING       | [STING: Self-attention based Time-series Imputation Networks using GAN](https://ieeexplore.ieee.org/abstract/document/9679183) | None | ICDM 2021 <br> B



# [Time Series Anomaly Detection](#content)
|  Task  |    Data |   Model  | Paper   |    Code    |   Publication    |
| :-: | :-: | :-: | :-: | - | - |
| Paper Nums: 11  | <img width=90/> |      |     |     |  <img width=320/> | 
|  Anomaly Detection | SMD <br> PSM <br> MSL&SMAP <br> SWaT NeurIPS-TS <br> |         Anomaly Transformer        | [Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy](https://openreview.net/forum?id=LzQQ89U1qm_) | [Pytorch](https://github.com/spencerbraun/anomaly_transformer_pytorch) | ICLR 2022 <br> None But Top 
| Density Estimation & Anomaly Detection | PMU-B <br> PMU-C <br> SWaT <br> METR-LA |         GANF        | [Graph-Augmented Normalizing Flows for Anomaly Detection of Multiple Time Series](https://openreview.net/forum?id=45L_dgP48Vd) | [Pytorch](https://github.com/EnyanDai/GANF) | ICLR 2022 <br> None But Top 
|  Anomaly Detection |    |          | [Anomaly Detection for Tabular Data with Internal Contrastive Learning](https://openreview.net/forum?id=_hszZbt46bT) | None | ICLR 2022 <br> None But Top 
|  Anomaly Detection |     |       AnomalyKiTS   | [AnomalyKiTS-Anomaly Detection Toolkit for Time Series](https://aaai-2022.virtualchair.net/poster_dm318) | None | AAAI 2022<br>A
|  Anomaly Detection |  SWaT <br> WADI <br> MSL <br> SMAP <br> SMD  |       PA   | [Towards a Rigorous Evaluation of Time-Series Anomaly Detection](https://aaai-2022.virtualchair.net/poster_aaai2239) |  None  | AAAI 2022 <br> A
|  Anomaly Detection |  SMD  |       FDRC   | [Online false discovery rate control for anomaly detection in time series](https://dl.acm.org/doi/10.1145/3447548.3467075) | None  | NIPS 2021 <br> A
|  Anomaly Detection |  SWaT <br> WADI <br> SMD <br> ASD  |       InterFusion   | [Multivariate Time Series Anomaly Detection and Interpretation using Hierarchical Inter-Metric and Temporal Embedding](https://papers.nips.cc/paper/2021/hash/def130d0b67eb38b7a8f4e7121ed432c-Abstract.html) |  [TF](https://github.com/zhhlee/InterFusion)  | KDD 2021 <br> A
|  Anomaly Detection |  SMD <br> SWaT <br> PSM <br> BKPI  |       RANSynCoders   | [Practical Approach to Asynchronous Multivariate Time Series Anomaly Detection and Localization](https://doi.org/10.1145/3447548.3467174) |  [TF](https://github.com/eBay/RANSynCoders)  | KDD 2021 <br> A
|  Anomaly Detection |  PUMP <br> WADI <br> SWaT  |       NSIBF   | [Time Series Anomaly Detection for Cyber-physical Systems via Neural System Identification and Bayesian Filtering](https://doi.org/10.1145/3447548.3467137) |  [TF](https://github.com/NSIBF/NSIBF)  | KDD 2021 <br> A
|  Anomaly Detection |  SWaT <br> WADI   |       GDN   | [Graph Neural Network-Based Anomaly Detection in Multivariate Time Series](https://ojs.aaai.org/index.php/AAAI/article/view/16523) |  [Pytorch](https://github.com/d-ailin/GDN)  | AAAI 2021 <br> A
|  Anomaly Detection |  KPI <br> Yahoo   |      FluxEV   | [FluxEV: A Fast and Effective Unsupervised Framework for Time-Series Anomaly Detection](https://doi.org/10.1145/3437963.3441823) |  None   | WSDM 2021<br> B
|  Anomaly Detection |  NYC  |       CHAT   | [Cross-interaction hierarchical attention networks for urban anomaly prediction](https://dl.acm.org/doi/abs/10.5555/3491440.3492041) |  [Pytorch](https://github.com/d-ailin/GDN)  | IJCAI 2020 <br> A
|  Anomaly Detection |  MORE  |       MORE   | [https://github.com/ZIYU-DEEP/IJCAI-Paper-List-of-Anomaly-Detection](https://github.com/ZIYU-DEEP/IJCAI-Paper-List-of-Anomaly-Detection) |  MORE   | IJCAI  <br> A
|  Anomaly Detection |  MORE  |       MORE   | [DeepTimeSeriesModel](https://github.com/drzhang3/DeepTimeSeriesModel) |  MORE   | MORE  <br> A







# [Demand Prediction](#content)
|  Task  |    Data |   Model  | Paper   |    Code    |   Publication    |
| :-: | :-: | :-: | :-: | - | - |
| Paper Nums: 12 | <img width=150/> | <img width=220/>  |   |   |   <img width=300/> | 
| Supply & <br> Demand | JONAS-NYC <br> JONAS-DC  <br>  COVID-CHI <br>  COVID-US |         EAST-Net | [Event-Aware Multimodal Mobility Nowcasting](https://aaai-2022.virtualchair.net/poster_aaai10914) | [Pytorch](https://github.com/underdoc-wang/EAST-Net) | AAAI 2022<br>A
| Health Demand | Family Van  |         framework        | [Using Public Data to Predict Demand for Mobile Health Clinics](https://aaai-2022.virtualchair.net/poster_emer91) | None | AAAI 2022<br>A
| Traffic Demand | NYC Bike <br> NYC Taxi  |         CCRNN        | [Coupled Layer-wise Graph Convolution for Transportation Demand Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/16591) | [Pytorch](https://github.com/Essaim/CGCDemandPrediction) | AAAI 2021<br>A
| Traffic Demand | Beijing  <br> Shanghai  |         Ada-MSTNet        | [Community-Aware Multi-Task Transportation Demand Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/16107) | None | AAAI 2021<br>A
| Job Demand | Online |         TDAN       | [Talent Demand Forecasting with Attentive Neural Sequential Model](https://dl.acm.org/doi/abs/10.1145/3447548.3467131) | None | KDD 2021<br>A
| Ambulance Demand | Tokyo |         EMS-Pred       | [Forecasting Ambulance Demand with Profiled Human Mobility via Heterogeneous Multi-Graph Neural Networks](https://ieeexplore.ieee.org/abstract/document/9458623) |  [Pytorch](https://github.com/underdoc-wang/EMS-Pred-ICDE-21)  | ICDE 2021<br>A
| Traffic  Demand | Chengdu <br> NYC |         DAGNN       | [Dynamic Auto-structuring Graph Neural Network-A Joint Learning Framework for Origin-Destination Demand Prediction](https://ieeexplore.ieee.org/abstract/document/9657493) | None   | TKDE 2021<br>A
| Market Demand  |  Juhuasuan  <br> Tiantiantemai     |        RMLDP    | [Relation-aware Meta-learning for E-commerce Market Segment Demand Prediction with Limited Records](https://doi.org/10.1145/3437963.3441750) |    None  | WSDM 2021<br> B
| Metro  Demand | MetroBJ2016 <br> MetroBJ2018 |         CAS       | [Short-term origin-destination demand prediction in urban rail transit systems: A channel-wise attentive split-convolutional neural network method](https://doi.org/10.1016/j.trc.2020.102928) |  None   |  Transportation Research Part C  <br> SCI 1 Top
| Metro  Demand | MetroBJ2016 <br> MetroBJ2018 |         ST-ED       | [Predicting origin-destination ride-sourcing demand with a spatio-temporal encoder-decoder residual multi-graph convolutional network](https://doi.org/10.1016/j.trc.2020.102858) |  None   |  Transportation Research Part C <br> SCI 1 Top
| Traffic Demand |  Seattle bikeshare  |       FairST      | [Fairness-Aware Demand Prediction for New Mobility](https://ojs.aaai.org/index.php/AAAI/article/view/5458) | None | AAAI 2021<br>A
| Drug Demand  |  Wikipedia  |        None          | [Predicting Drug Demand with Wikipedia Views: Evidence from Darknet Markets](https://www.ijcai.org/proceedings/2020/610) | None  | WWW 2020<br>A




# [Travel Time Estimation](#content)
|  Task  |    Data |   Model  | Paper   |    Code    |   Publication    |
| :-: | :-: | :-: | :-: | - | - |
| Paper Nums:6 | <img width=150/> | <img width=220/>  |   |   |   <img width=300/> | 
| TTE | Taiyuan <br> Huizhou <br> Hefei|         SSML        | [SSML: Self-Supervised Meta-Learner for En Route Travel Time Estimation at Baidu Maps](https://dl.acm.org/doi/10.1145/3447548.3467060) | [Paddle](https://github.com/PaddlePaddle/Research/tree/master/ST_DM/KDD2021-SSML)  | KDD 2021<br>A
| TTE | Shenyang     |     HetETA        | [HetETA: Heterogeneous Information Network Embedding for Estimating Time of Arrival](https://dl.acm.org/doi/10.1145/3394486.3403294) | [TF](https://github.com/didi/heteta)  | KDD 2020<br>A
| TTE | Beijing <br> Suzhou <br> Shenyang   |     CompactETA        | [CompactETA: A Fast Inference System for Travel Time Prediction](https://dl.acm.org/doi/10.1145/3394486.3403386) | None | KDD 2020<br>A
| TTE | Beijing <br> Shanghai    |     TTPNet        | [TTPNet: A Neural Network for Travel Time Prediction Based on Tensor Decomposition and Graph Embedding](https://ieeexplore.ieee.org/abstract/document/9261122) |   [Pytorch](https://github.com/YibinShen/TTPNet)  | TKDE 2020<br>A
| TTE | Porto <br> Chengdu    |     DeepIST        | [DeepIST: Deep Image-based Spatio-Temporal Network for Travel Time Estimation](https://dl.acm.org/doi/abs/10.1145/3357384.3357870) |   [TF](https://github.com/csiesheep/deepist)  | CIKM 2019<br>B
| TTE | Chengdu <br> Beijing    |     DeepTTE        | [When Will You Arrive? Estimating Travel Time Based on Deep Neural Networks](https://ojs.aaai.org/index.php/AAAI/article/view/11877) |   [Pytorch](https://github.com/UrbComp/DeepTTE)  | AAAI 2018<br>A





# [Traffic Location Prediction](#content)
|  Task  |    Data |   Model  | Paper   |    Code    |   Publication    |
| :-: | :-: | :-: | :-: | - | - |
| Paper Nums:5 | <img width=150/> | <img width=220/>  |   |   |   <img width=300/> | 
| Location | ETH+UCY <br> SDD <br> nuScenes <br> SportVU |              | [You Mostly Walk Alone: Analyzing Feature Attribution in Trajectory Prediction](https://openreview.net/forum?id=POxF-LEqnF) | None | AAAI 2022<br>A
| Location | Gowalla <br> Foursquare <br> WiFi-Trace  |     GCDAN         | [Predicting Human Mobility via Graph Convolutional Dual-attentive Networks](https://dl.acm.org/doi/10.1145/3488560.3498400) |  [Pytorch](https://github.com/GCDAN/GCDAN) | WSDM 2022<br> B
| Location | MI <br> SIP   |     CMT-Net         | [CMT-Net: A Mutual Transition Aware Framework for Taxicab Pick-ups and Drop-offs Co-Prediction](https://dl.acm.org/doi/10.1145/3488560.3498394) | None | WSDM 2022 <br>B
| Location | ETH <br> HODEL  <br> UNIV <br> ZARA1 <br> ZARA2   |     CARPe       | [CARPe Posterum: A Convolutional Approach for Real-Time Pedestrian Path Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/16335) | [Pytorch](https://github.com/TeCSAR-UNCC/CARPe_Posterum) | AAAI 2021<br>A
| Location | ETH <br> HODEL  <br> UNIV <br> ZARA1 <br> ZARA2   |     DMRGCN       | [Disentangled Multi-Relational Graph Convolutional Network for Pedestrian Trajectory Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/16174) | [Pytorch](https://github.com/TeCSAR-UNCC/CARPe_Posterum) | AAAI 2021<br>A
| Location |  MORE  |     MORE       | [https://github.com/Pursueee/Trajectory-Paper-Collation](https://ojs.aaai.org/index.php/AAAI/article/view/16174) | [Pytorch](https://github.com/Pursueee/Trajectory-Paper-Collation) | MORE<br>A







# [Traffic Accident Prediction](#content)
|  Task  |    Data |   Model  | Paper   |    Code    |   Publication    |
| :-: | :-: | :-: | :-: | - | - |
| Paper Nums:5 | <img width=150/> | <img width=220/>  |   |   |   <img width=300/> | 
| Accident | PeMS  |         AGWN        | [Early Forecast of Traffc Accident Impact Based on a Single-Snapshot Observation (Student Abstract)](https://aaai-2022.virtualchair.net/poster_sa103) | [Pytorch](https://github.com/gm3g11/AGWN) | AAAI 2022<br>A
| Accident | NYC <br> Chicago |         GSNet        | [GSNet: Learning Spatial-Temporal Correlations from Geographical and Semantic Aspects for Traffic Accident Risk Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/16566) | [Pytorch](https://github.com/Echohhhhhh/GSNet) | AAAI 2021<br>A
| Accident | Beijing <br> Suzhou <br> Shenyang |         RiskOracle        | [RiskOracle: A Minute-Level Citywide Traffic Accident Forecasting Framework](https://ojs.aaai.org//index.php/AAAI/article/view/5480) | [TF](https://github.com/zzyy0929/AAAI2020-RiskOracle/) | AAAI 2020<br>A
| Accident | NYC <br> SIP  |         RiskSeq        | [Foresee Urban Sparse Traffic Accidents: A Spatiotemporal Multi-Granularity Perspective](https://ieeexplore.ieee.org/document/9242313) | None| AAAI 2020<br>A
| Accident |   |         Hetero-ConvLSTM        | [Hetero-ConvLSTM: A Deep Learning Approach to Traffic Accident Prediction on Heterogeneous Spatio-Temporal Data](https://ieeexplore.ieee.org/document/9242313) | None| KDD 2018<br>A






# [Other Forecasting](#content)
|  Task  |    Data |   Model  | Paper   |    Code    |   Publication    |
| :-: | :-: | :-: | :-: | - | - |
| Paper Nums:8 | <img width=150/> | <img width=220/>  |   |   |   <img width=300/> | 
| Crop Yield  Prediction | American Crop |         GNN-RNN        | [A GNN-RNN Approach for Harnessing Geospatial and Temporal Information: Application to Crop Yield Prediction](https://aaai-2022.virtualchair.net/poster_aisi6416) | None  | AAAI 2022<br> A
| Epidemic Prediction | Globe <br> US-State  <br> US-County |         CausalGNN     | [CausalGNN: Causal-based Graph Neural Networks for Spatio-Temporal](https://aaai-2022.virtualchair.net/poster_aisi6475) | Future | AAAI 2022<br>A
| Career Trajectory Prediction  | Company  <br> Position   |      TACTP          | [Variable Interval Time Sequence Modeling for Career Trajectory Prediction: Deep Collaborative Perspective](https://dl.acm.org/doi/10.1145/3442381.3449959) | None | WWW 2021<br>A
| Health Prediction | NASH <br> AD  |        UNITE    | [UNITE: Uncertainty-based Health Risk Prediction Leveraging Multi-sourced Data](https://doi.org/10.1145/3442381.3450087) |     [Pytorch](https://github.com/Chacha-Chen/UNITE)   | WWW 2021<br>A
| Crime Prediction | NYC <br> Chicago  |         ST-SHN     | [Spatial-Temporal Sequential Hypergraph Network for Crime Prediction with Dynamic Multiplex Relation Learning](https://www.ijcai.org/proceedings/2021/0225.pdf) |     [Pytorch](https://github.com/akaxlh/ST-SHN)   | IJCAI 2021<br>A
| Event  Prediction  |  DJIA30   <br> WebTraffic   <br> NetFlow  <br> ClockErr  <br>   AbServe  |        EvoNet    | [Time-Series Event Prediction with Evolutionary State Graph](https://doi.org/10.1145/3437963.3441827) |   None   | WSDM 2021 <br> B
| Parking Prediction | Beijing <br> Shanghai |         SHARE       | [Semi-Supervised Hierarchical Recurrent Graph Neural Network for City-Wide Parking Availability Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/5471) |  [Pytorch](https://github.com/Vvrep/SHARE-parking_availability_prediction-Pytorch)  | AAAI 2020<br>A
| Parking Prediction | Ningbo <br> Changsha |         PewLSTM       | [PewLSTM: Periodic LSTM with Weather-Aware Gating Mechanism for Parking Behavior Prediction](https://www.ijcai.org/proceedings/2020/610) |  [Pytorch](https://github.com/NingxuanFeng/PewLSTM)  | IJCAI 2020<br>A




# [Conferences](#content)


❗ 建议使用 [dblp](https://dblp.uni-trier.de/) 和 [Aminer](https://www.aminer.cn/conf)查询

❗ It is highly recommended to utilize the [dblp](https://dblp.uni-trier.de/) and [Aminer](https://www.aminer.cn/conf)(in Chinese) to search.


## Some Useful Websites
> 会议之眼 https://www.conferenceeye.cn/#/layout/home
> 
> Call4Papers http://123.57.137.208/ccf/ccf-8.jsp
> 
> Conference List http://www.conferencelist.info/upcoming.html

## Table of Conferences

* [AAAI](#AAAI) (to 2022)
* [IJCAI](#IJCAI) (to 2021)
* [KDD](#KDD) (to 2021)
* [WWW](#WWW) (to 2022)
* [ICLR ](#ICLR)(to 2022)
* [ICML](#ICML) (to 2021)
* [NeurIPS](#NeurIPS) (to 2021)
* [CIKM](#CIKM) (to 2021)
* [WSDM](#WSDM) (to 2022)


## AAAI

| Conference | Source                                                       | Deadline          | Notification      |
| ---------- | ------------------------------------------------------------ | ----------------- | ----------------- |
|AAAI-22|[Link](https://aaai.org/Conferences/AAAI-22/wp-content/uploads/2021/12/AAAI-22_Accepted_Paper_List_Main_Technical_Track.pdf)|September 8, 2021|November 29, 2021|
| AAAI-21    | [Link](https://aaai.org/Conferences/AAAI-21/wp-content/uploads/2020/12/AAAI-21_Accepted-Paper-List.Main_.Technical.Track_.pdf) |                   |                   |
| AAAI-20    | [Link](https://aaai.org/Conferences/AAAI-20/wp-content/uploads/2020/01/AAAI-20-Accepted-Paper-List.pdf) |                   |                   |
| AAAI-19    | [Link](https://aaai.org/Conferences/AAAI-19/wp-content/uploads/2018/11/AAAI-19_Accepted_Papers.pdf) |                   |                   |
| AAAI-18    | [Link](https://aaai.org/Conferences/AAAI-18/wp-content/uploads/2017/12/AAAI-18-Accepted-Paper-List.Web_.pdf) |                   |                   |
| AAAI-17    | [Link](https://www.aaai.org/Conferences/AAAI/2017/aaai17accepted-papers.pdf) |                   |                   |
| AAAI-16    | [Link](https://www.aaai.org/Conferences/AAAI/2016/aaai16accepted-papers.pdf) |                   |                   |
| AAAI-15    | [Link](https://www.aaai.org/Conferences/AAAI/2015/iaai15accepted-papers.pdf) |                   |                   |
| AAAI-14    | [Link](https://www.aaai.org/Conferences/AAAI/2014/aaai14accepts.php) |                   |                   |
| AAAI-13    | [Link](https://www.aaai.org/Conferences/AAAI/2013/aaai13accepts.php) |                   |                   |



## [IJCAI](https://www.ijcai.org/past_proceedings)

| Conference | Source                                                      | Deadline | Notification |
| ---------- | ----------------------------------------------------------- | ---------- | ---------- |
|IJCAI-22|| January 14, 2022 | April 20, 2022 |
|IJCAI-21|[Link](https://ijcai-21.org/program-main-track/)|  |  |
| IJCAI-20   | [Link](http://static.ijcai.org/2020-accepted_papers.html)   |  |  |
| IJCAI-19   | [Link](https://www.ijcai19.org/accepted-papers.html)        |  |  |
| IJCAI-18   | [Link](https://www.ijcai-18.org/accepted-papers/index.html) |  |  |
| IJCAI-17   | [Link](https://ijcai-17.org/accepted-papers.html)           |  |  |
| IJCAI-16   | [Link](https://www.ijcai.org/proceedings/2016)              |  |  |
| IJCAI-15   | [Link](https://www.ijcai.org/Proceedings/2015)              |  |  |
| IJCAI-14   | None                                                        |  |  |



## KDD

> Format : https://www.kdd.org/kdd20xx/accepted-papers

| Conference | Source                                              | Deadline | Notification |
| ---------- | --------------------------------------------------- | ---------- | ---------- |
|KDD-22|| Feb 10th, 2022 | May 19th, 2022 |
|KDD-21| [Link](https://kdd.org/kdd2021/accepted-papers)|  |  |
| KDD-20     | [Link](https://www.kdd.org/kdd2020/accepted-papers) |  |  |
| KDD-19     | [Link](https://www.kdd.org/kdd2019/accepted-papers) |  |  |
| KDD-18     | [Link](https://www.kdd.org/kdd2018/accepted-papers) |  |  |
| KDD-17     | [Link](https://www.kdd.org/kdd2017/accepted-papers) |  |  |



## WWW

TheWebConf

| Conference | Source                                                     | Deadline | Notification |
| ---------- | ---------------------------------------------------------- | ---------- | ---------- |
|WWW-22| [Link](https://www2022.thewebconf.org/accepted-papers/)| 2021-10-21 ... | 2022-01-13 ... |
|WWW-21| [Link](https://www2021.thewebconf.org/program/papers/)|  |  |
| WWW-20     | [Link](https://dl.acm.org/doi/proceedings/10.1145/3366423) |  |  |
| WWW-19     | [Link](https://www2019.thewebconf.org/accepted-papers)     |  |  |
| WWW-18     | [Link](https://dl.acm.org/doi/proceedings/10.5555/3178876) |  |  |
| WWW-17     | [Link](https://dl.acm.org/doi/proceedings/10.1145/3308558) |  |  |

## ICLR

FInding it on openreview:


> Homepage https://openreview.net/group?id=ICLR.cc
>

| Conference | Source                                                     | Deadline | Notification |
| ---------- | ---------------------------------------------------------- | ---------- | ---------- |
|ICLR 2022|https://openreview.net/group?id=ICLR.cc/2022/Conference|Oct 06 '21|Jan 24 '22|
| ICLR 2021  | [https://openreview.net/group?id=ICLR.cc/2021/Conference](https://openreview.net/group?id=ICLR.cc/2021/Conference) |  |  |
| ICLR 2020     | [https://openreview.net/group?id=ICLR.cc/2020/Conference](https://openreview.net/group?id=ICLR.cc/2020/Conference)     |  |  |



## ICML

>  Homepage https://icml.cc

| Conference | Source                                                     | Deadline | Notification |
| ---------- | ---------------------------------------------------------- | ---------- | ---------- |
|ICML 2022|https://icml.cc/Conferences/2022/Schedule| Jan 27, 2022 |  |
|ICML 2021| [https://icml.cc/Conferences/2021/Schedule](https://icml.cc/Conferences/2021/Schedule)|  |  |
| ICML 2020 | [https://icml.cc/Conferences/2020/Schedule](https://icml.cc/Conferences/2020/Schedule) |  |  |
| ICML 2019  | [https://icml.cc/Conferences/2019/Schedule](https://icml.cc/Conferences/2019/Schedule)  |  |  |



## NeurIPS

[All Links](https://papers.nips.cc/)


## CIKM

[All Links](https://dl.acm.org/conference/cikm)


## WSDM

[All Links](https://dl.acm.org/conference/wsdm)


