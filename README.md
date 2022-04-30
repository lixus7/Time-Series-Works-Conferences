# Time-Series Work and Conference

<div align="center">
<img border="0" src="https://camo.githubusercontent.com/54fdbe8888c0a75717d7939b42f3d744b77483b0/687474703a2f2f6a617977636a6c6f76652e6769746875622e696f2f73622f69636f2f617765736f6d652e737667" />
<img border="0" src="https://camo.githubusercontent.com/1ef04f27611ff643eb57eb87cc0f1204d7a6a14d/68747470733a2f2f696d672e736869656c64732e696f2f7374617469632f76313f6c6162656c3d254630253946253843253946266d6573736167653d496625323055736566756c267374796c653d7374796c653d666c617426636f6c6f723d424334453939" />
<a href="https://github.com/TatsuyaD7">     <img border="0" src="https://camo.githubusercontent.com/41e8e16b771d56dd768f7055354613254961d169/687474703a2f2f6a617977636a6c6f76652e6769746875622e696f2f73622f6769746875622f677265656e2d666f6c6c6f772e737667" /> </a> 
<a href="https://github.com/TatsuyaD7/Time-Series-Work-Conference/issues">     <img border="0" src="https://img.shields.io/github/issues/TatsuyaD7/Time-Series-Work-Conference" /> </a>
<a href="https://github.com/TatsuyaD7/Time-Series-Work-Conference/network/members">     <img border="0" src="https://img.shields.io/github/forks/TatsuyaD7/Time-Series-Work-Conference" /> </a>
<a href="https://github.com/TatsuyaD7/Time-Series-Work-Conference/stargazers">     <img border="0" src="https://img.shields.io/github/stars/TatsuyaD7/Time-Series-Work-Conference" /> </a>
<a href="https://github.com/TatsuyaD7/Time-Series-Work-Conference/blob/master/WeChat.md">     <img border="0" src="https://camo.githubusercontent.com/013c283843363c72b1463af208803bfbd5746292/687474703a2f2f6a617977636a6c6f76652e6769746875622e696f2f73622f69636f2f7765636861742e737667" /> </a>
</div>

I am under the guidance of Professor [Renhe Jiang](https://www.renhejiang.com/), [Dr. Jinliang Deng](https://scholar.google.com/citations?user=oaoJ2AYAAAAJ&hl=zh-CN&oi=ao), and [Professor Xuan Song](https://scholar.google.com/citations?user=_qCSLpMAAAAJ&hl=zh-CN&oi=ao).

We have basically completed the sorting of the task part, and will gradually mark the main methodology used for each job in the future. Meanwhile, I also want to organize all the datasets. However, my free time is limited because I'm applying for a phd offer recently. Thus, welcome those Dalaos who are also interested in sorting out time series in Top CS Conf to join this work!

All papers group by task and method (including the papers that have not been included in this github) will be placed on the Google Drive for everyone to use:

https://drive.google.com/drive/folders/17bILWdDxUrufRp3yilYfoU5VKywwS1g6?usp=sharing

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

# Recent Time Series Work Group by Task

- <a href = "#Multivariable-Time-Series-Forecasting">Multivariable Time Series Forecasting</a>
- <a href = "#Multivariable-Probabilistic-Time-Series-Forecasting">Multivariable Probabilistic Time Series Forecasting</a>
- <a href = "#Time-Series-Imputation">Time Series Imputation</a>
- <a href = "#Time-Series-Anomaly-Detection">Time Series Anomaly Detection</a>
- <a href = "#Demand-Prediction">Demand Prediction</a>
- <a href = "#Travel-Time-Estimation">Travel Time Estimation</a>
- <a href = "#Traffic-Location-Prediction">Traffic Location Prediction</a>
- <a href = "#Traffic-Event-Prediction">Traffic Event Prediction</a>
- <a href = "#Stock-Prediction">Stock Prediction</a>
- <a href = "#Other-Forecasting">Other Forecasting</a>

"A, B, C" in the publication denotes the CCF Rank;

According to the average quality of the papers and codes, we rank the conferences as follows (Don't be rude, I'm talking about the average): 

NIPS>ICML>ICLR>KDD>AAAI>IJCAI>WWW>CIKM>ICDM>WSDM;

Note that: AISTAT is CCFC but is top in computational mathematics (such as probabilistic problem);


# [Multivariable Time Series Forecasting](#content)
|  Task  |    Data |   Model  | Paper   |    Code    |   Publication    |
| :-: | :-: | :-: | :-: | :-: | - |
| Paper Nums:100+ | <img width=150/> | <img width=220/>  |   |   |   <img width=300/> | 
| Traffic Speed | NAVER-Seoul <br> METR-LA |         PM-MemNet         | [Learning to Remember Patterns: Pattern Matching Memory Networks for Traffic Forecasting](https://openreview.net/forum?id=wwDg3bbYBIq) | [Pytorch](https://github.com/HyunWookL/PM-MemNet) | ICLR 2022  <br>None But Top 
| Multivariable | PeMSD3 <br> PeMSD4 <br> PeMSD8 <br> COVID-19,etc |         TAMP-S2GCNets         | [TAMP-S2GCNets: Coupling Time-Aware Multipersistence Knowledge Representation with Spatio-Supra Graph Convolutional Networks for Time-Series Forecasting](https://openreview.net/forum?id=wv6g8fWLX2q) | [Pytorch](https://www.dropbox.com/sh/n0ajd5l0tdeyb80/AABGn-ejfV1YtRwjf_L0AOsNa?dl=0) | ICLR 2022 <br> None But Top 
| Multivariable | ETT <br> Electricity <br> Weather |         CoST         | [CoST: Contrastive Learning of Disentangled Seasonal-Trend Representations for Time Series Forecasting](https://openreview.net/forum?id=PilZY3omXV2) | [Pytorch](https://github.com/salesforce/CoST) | ICLR 2022 <br> None But Top 
| Multivariable | Electricity <br> Traffic <br> M4 <br> CASIO <br> NP |         DEPTS         | [DEPTS: Deep Expansion Learning for Periodic Time Series Forecasting](https://openreview.net/forum?id=AJAR-JgNw__) | [Pytorch](https://github.com/weifantt/DEPTS) | ICLR 2022 <br> None But Top 
| Multivariable | ETT <br> Electricity <br> Wind <br> App Flow |         Pyraformer     | [Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting](https://openreview.net/forum?id=0EXmFzUn5I) | [Pytorch](https://github.com/alipay/Pyraformer) | ICLR 2022 <br> None But Top 
| Multivariable | ETT <br> ECL  <br> M4 <br> Air Quality <br> Nasdaq |         RevIN     | [Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift](https://openreview.net/forum?id=cGDAkQo1C0p) | [Pytorch](https://github.com/ts-kim/RevIN) | ICLR 2022 <br> None But Top 
| Multivariable | Temperature <br> Cloud cover  <br> Humidity <br> Wind |         CLCRN     | [Conditional Local Convolution for Spatio-temporal Meteorological Forecasting](https://aaai-2022.virtualchair.net/poster_aaai1716) | [Pytorch](https://github.com/BIRD-TAO/CLCRN) | AAAI 2022<br>A
| Traffic Flow | PeMSD3 <br> PeMSD4 <br> PeMSD7 <br> PeMSD8 <br> PeMSD7(M) <br> PeMSD7(L) |         STG-NCDE     | [Graph Neural Controlled Differential Equations for Traffic Forecasting](https://aaai-2022.virtualchair.net/poster_aaai1716) | [Pytorch](https://github.com/jeongwhanchoi/STG-NCDE) | AAAI 2022<br>A
| Multivariable | GT-221 <br> WRS-393 <br> ZGC-564 |         STDEN     | [STDEN: Towards Physics-guided Neural Networks for Traffic Flow Prediction](https://aaai-2022.virtualchair.net/poster_aaai211) | [Pytorch](https://github.com/Echo-Ji/STDEN)   | AAAI 2022<br>A
| Multivariable | Electricity <br> Traffic <br> PeMSD7(M) <br> METR-LA  |         CATN     | [CATN: Cross Attentive Tree-Aware Network for Multivariate Time Series Forecasting](https://aaai-2022.virtualchair.net/poster_aaai7403) | None | AAAI 2022<br>A
| Multivariable | ETT <br> Electricity  |         TS2Vec     | [TS2Vec: Towards Universal Representation of Time Series](https://aaai-2022.virtualchair.net/poster_aaai8809) | [Pytorch](https://github.com/yuezhihan/ts2vec) | AAAI 2022<br>A
| Multivariable | GoogleSymptoms  <br> Covid19  <br> Power <br> Tweet |         CAMul     | [EXIT: Extrapolation and Interpolation-based Neural Controlled Differential Equations for Time-series Classification and Forecasting](https://doi.org/10.1145/3485447.3512037) |  [Pytorch](https://github.com/AdityaLab/CAMul)  | WWW 2022<br>A
| Multivariable | Electricity <br> Stock  |         MRLF     | [Multi-Granularity Residual Learning with Confidence Estimation for Time Series Prediction](https://doi.org/10.1145/3485447.3512056) | [Pytorch](https://github.com/CMLF-git-dev/MRLF) | WWW 2022<br>A
| Multivariable <br> Classification <br> Forecasting | MuJoCo  <br> Google Stock  |         EXIT     | [EXIT: Extrapolation and Interpolation-based Neural Controlled Differential Equations for Time-series Classification and Forecasting](https://doi.org/10.1145/3485447.3512030) | None | WWW 2022<br>A
| Traffic Flow | TaxiBJ <br> BikeNYC |         ST-GSP     | [ST-GSP: Spatial-Temporal Global Semantic Representation Learning for Urban Flow Prediction](https://dl.acm.org/doi/abs/10.1145/3488560.3498444) | [Pytorch](https://github.com/k51/STGSP) | WSDM 2022 <br> B
| Multivariable | M4 <br> Electricity <br> car-parts  |         TopAttn     | [Topological Attention for Time Series Forecasting](https://nips.cc/Conferences/2021/ScheduleMultitrack?event=26763) | [Pytorch](https://github.com/plus-rkwitt/TAN)<br> Future | NIPS 2021<br>A
| Multivariable | Rossmann <br> M5 <br> Wiki  |         MisSeq     | [MixSeq: Connecting Macroscopic Time Series Forecasting with Microscopic Time Series Data](https://proceedings.neurips.cc/paper/2021/hash/6b5754d737784b51ec5075c0dc437bf0-Abstract.html) | None | NIPS 2021<br>A
| Multivariable | ETT <br> Electricity <br> Exchange <br> Traffic <br> Weather <br> ILI |         Autoformer     | [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://openreview.net/forum?id=J4gRj6d5Qm) | [Pytorch](https://github.com/thuml/Autoformer) | NIPS 2021<br>A
| Multivariable | PeMSD4 <br> PeMSD8 <br> Traffic <br> ADI <br> M4 ,etc |         Error     | [Adjusting for Autocorrelated Errors in Neural Networks for Time Series](https://openreview.net/forum?id=tJ_CO8orSI) | [Pytorch](https://github.com/Daikon-Sun/AdjustAutocorrelation) | NIPS 2021<br>A
| Multivariable | Bytom <br> Decentraland <br>  PeMSD4 <br> PeMSD8|         Z-GCNETs     | [Z-GCNETs: Time Zigzags at Graph Convolutional Networks for Time Series Forecasting](http://proceedings.mlr.press/v139/chen21o.html) | [Pytorch](https://github.com/Z-GCNETs/Z-GCNETs) | ICML 2021<br>A
| Multivariable | PeMSD7(M) <br> METR-LA <br>  PeMS-BAY  |         Cov     | [Conditional Temporal Neural Processes with Covariance Loss](http://proceedings.mlr.press/v139/yoo21b.html) | None | ICML 2021<br>A
| Multivariable | METR-LA <br>  PeMS-BAY  <br>  PMU |         GTS     | [Discrete Graph Structure Learning for Forecasting Multiple Time Series](https://openreview.net/forum?id=WEHSlH5mOk) | [Pytorch](https://github.com/chaoshangcs/GTS) | ICLR 2021  <br>None But Top 
| Multivariable | Benz <br> Air Quality <br> FuelMoisture  |         framework     | [A Transformer-based Framework for Multivariate Time Series Representation Learning](https://doi.org/10.1145/3447548.3467401) | [Pytorch](https://github.com/gzerveas/mvts_transformer)  | KDD 2021<br>A
| Federated Multivariable | PeMS-BAY <br>  METR-LA  |         CNFGNN     | [Cross-Node Federated Graph Neural Network for Spatio-Temporal Data Modeling](https://doi.org/10.1145/3447548.3467371) | [Pytorch](https://github.com/mengcz13/KDD2021_CNFGNN)  | KDD 2021<br>A
| Traffic Speed  | PeMSD4 <br>  PeMSD8 <br>  England |         DMSTGCN     | [Dynamic and Multi-faceted Spatio-temporal Deep Learning for Traffic Speed Forecasting](https://doi.org/10.1145/3447548.3467275) | [Pytorch](https://github.com/liangzhehan/DMSTGCN)  | KDD 2021<br>A
| Traffic Flow  | PeMSD7(M) <br>  PeMSD7(L) <br> PeMS03 <br> PeMS04 <br> PeMS07 <br> PeMS08 |         STGODE     | [Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting](https://doi.org/10.1145/3447548.3467430) | [Pytorch](https://github.com/square-coder/STGODE)  | KDD 2021<br>A
| Multivariable  | BikeNYC <br>  PeMSD7(M) <br> Electricity |        ST-Norm     | [ST-Norm: Spatial and Temporal Normalization for Multi-variate Time Series Forecasting](https://doi.org/10.1145/3447548.3467330) | [Pytorch](https://github.com/JLDeng/ST-Norm)  | KDD 2021<br>A
| Multivariable  | DiDiXiamen <br>  DiDiChengdu |       TrajNet    | [TrajNet: A Trajectory-Based Deep Learning Model for Traffic Prediction](https://doi.org/10.1145/3447548.3467236) | None | KDD 2021<br>A
| Multivariable  | Guangzhou <br> Seattle <br> HZMetro , etc. |      DSARF    | [Deep Switching Auto-Regressive Factorization: Application to Time Series Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/16907) |  None | AAAI 2021<br>A
|Traffic Speed   |  METR-LA  <br> PeMS-BAY |      FC-GAGA    | [FC-GAGA: Fully Connected Gated Graph Architecture for Spatio-Temporal Traffic Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/17114) |  [TF](https://github.com/boreshkinai/fc-gaga) | AAAI 2021<br>A
|Traffic Speed   |  DiDiJiNan  <br> DiDiXiAn |     HGCN   | [Hierarchical Graph Convolution Network for Traffic Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/16088) | [Pytorch](https://github.com/guokan987/HGCN) | AAAI 2021<br>A
|  Multivariable   |  ETT  <br> Weather <br> ECL  |     Informer   | [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/17325) | [Pytorch](https://github.com/zhouhaoyi/Informer2020) | AAAI 2021<br>A
|  Traffic Flow    |  NYCMetro  <br> NYC Bike <br> NYC Taxi  |     MOTHER   | [Modeling Heterogeneous Relations across Multiple Modes for Potential Crowd Flow Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/16603) |  None  | AAAI 2021<br>A
|  Multivariable  |  METR-LA  <br> PeMS-BAY  <br> PeMSD7(M) <br>  PeMSD7(L) <br> PeMS03 <br> PeMS04 <br> PeMS07 <br> PeMS08  |     STFGNN   | [Spatial-Temporal Fusion Graph Neural Networks for Traffic Flow Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/16542) | [Mxnet](https://github.com/MengzhangLI/STFGNN) | AAAI 2021<br>A
|  Multivariable  | BJ Taxi <br> NYC Taxi  <br> NYC Bike1  <br> NYC Bike2 |     STGDN   | [Traffic Flow Forecasting with Spatial-Temporal Graph Diffusion Network](https://ojs.aaai.org/index.php/AAAI/article/view/17761) | [Mxnet](https://github.com/nimingniming/gdn) | AAAI 2021<br>A
|   Traffic Flow     |  SG-TAXI   |     TrGNN   | [Traffic Flow Prediction with Vehicle Trajectories](https://ojs.aaai.org/index.php/AAAI/article/view/16104) | [Pytorch](https://github.com/mingqian000/TrGNN) | AAAI 2021<br>A
|  Multivariable  | Road <br> POIs <br> SIGtraf |     DMLM   | [Predicting Traffic Congestion Evolution: A Deep Meta Learning Approach](https://www.ijcai.org/proceedings/2021/0417.pdf) | [Future](https://github.com/HelenaYD/DMLM) | IJCAI 2021<br>A
|  Multivariable  | Motes <br> Soil  <br> Revenue  <br> Traffic  <br> 20CR |     NET   | [Network of Tensor Time Series](https://doi.org/10.1145/3442381.3449969) | [Pytorch](https://github.com/baoyujing/NET3) | WWW 2021<br>A
|  Multivariable  | VevoMusic <br> WikiTraffic  <br> LOS-LOOP  <br> SZ-taxi  |     Radflow   | [Radflow: A Recurrent, Aggregated, and Decomposable Model for Networks of Time Series](https://doi.org/10.1145/3442381.3449945) | [Pytorch](https://github.com/alasdairtran/radflow) | WWW 2021<br>A
|  Multivariable  |  METR-LA  <br> Wiki-EN    |     REST   | [REST: Reciprocal Framework for Spatiotemporal-coupled Predictions](https://ieeexplore.ieee.org/document/9346058) | None | WWW 2021<br>A
|  Multivariable  |  PeMS03 <br> PeMS04 <br> PeMS07 <br> PeMS08   <br> HZMetro  |     ASTGNN   | [Learning Dynamics and Heterogeneity of Spatial-Temporal Graph Data for Traffic Forecasting](https://doi.org/10.1145/3442381.3449928) | None | TKDE 2021<br>A
|  Multivariable  |  PeMS03 <br> PeMS04 <br> PeMS07 <br> PeMS08   <br> HZMetro  |     ASTGNN   | [Learning Dynamics and Heterogeneity of Spatial-Temporal Graph Data for Traffic Forecasting](https://doi.org/10.1145/3442381.3449928) | None | TKDE 2021<br>A
| Multivariable | TaxiBJ  <br> BikeNYC-I  <br> BikeNYC-II <br> TaxiNYC <br> METR-LA  <br> PeMS-BAY  <br> PeMSD7(M)   |        DL-Traff     | [DL-Traff: Survey and Benchmark of Deep Learning Models for Urban Traffic Prediction](https://doi.org/10.1145/3459637.3482000) | Graph:[PyTorch](https://github.com/deepkashiwa20/DL-Traff-Graph) <br> Grid:[TF](https://github.com/deepkashiwa20/DL-Traff-Grid)  | CIKM 2021 <br> B
| Multivariable | METR-LA  <br> PeMS-BAY  <br> PeMSD7(M)   |        TorchGeoTem  | [PyTorch Geometric Temporal: Spatiotemporal Signal Processing with Neural Machine Learning Models](https://doi.org/10.1145/3459637.3482000) | [PyTorch](https://github.com/benedekrozemberczki/pytorch_geometric_temporal)  | CIKM 2021 <br> B
| Traffic Flow | TaxiBJ <br> BikeNYC |         LLF     | [Learning to Learn the Future: Modeling Concept Drifts in Time Series Prediction](https://doi.org/10.1145/3459637.3482271) | None | CIKM 2021 <br> B
| Multivariable | ETT <br> Electricity |         HI     | [Historical Inertia: A Neglected but Powerful Baseline for Long Sequence Time-series Forecasting](https://doi.org/10.1145/3459637.3482120) | None | CIKM 2021 <br> B
| Multivariable | ETT <br> ELE  |         AGCNT     | [AGCNT: Adaptive Graph Convolutional Network for Transformer-based Long Sequence Time-Series Forecasting](https://doi.org/10.1145/3459637.3482054) | None | CIKM 2021 <br> B
| Cellular Traffic | cellular   |         MPGAT     | [Multivariate and Propagation Graph Attention Network for Spatial-Temporal Prediction with Outdoor Cellular Traffic](https://doi.org/10.1145/3459637.3482152) | [Pytorch](https://github.com/cylin-cmlab/MPNet)  <br> Future | CIKM 2021 <br> B
| Traffic Speed | METR-LA <br> PeMS-BAY <br> Simulated |         STNN     | [Space Meets Time: Local Spacetime Neural Network For Traffic Flow Forecasting](https://ieeexplore.ieee.org/abstract/document/9679008/) | [Pytorch](https://github.com/songyangco/STNN) | ICDM 2021<br>  B
| Traffic Speed | DiDiChengdu <br> DiDiXiAn  |         T-wave     | [Trajectory WaveNet: A Trajectory-Based Model for Traffic Forecasting](https://ieeexplore.ieee.org/abstract/document/9679147) | [Pytorch](https://github.com/songyangco/STNN) | ICDM 2021<br>  B
| Multivariable | Sanyo <br> Hanergy <br> Solar <br> Electricity  <br> Exchange  |         SSDNet     | [SSDNet: State Space Decomposition Neural Network for Time Series Forecasting](https://ieeexplore.ieee.org/abstract/document/9679135/) | [Pytorch](https://github.com/YangLIN1997/SSDNet-ICDM2021) | ICDM 2021 <br> B
| Traffic Volumn | HangZhou City <br> JiNan City |         CTVI     | [Temporal Multi-view Graph Convolutional Networks for Citywide Traffic Volume Inference](https://ieeexplore.ieee.org/abstract/document/9679045/) | [Pytorch](https://github.com/dsj96/CTVI-master) | ICDM 2021 <br>  B
| Traffic Volumn | Uber Movements <br>  Grab-Posisi |         TEST-GCN     | [TEST-GCN: Topologically Enhanced Spatial-Temporal Graph Convolutional Networks for Traffic Forecasting](https://ieeexplore.ieee.org/abstract/document/9679077) | None | ICDM 2021<br> B
| Multivariable | Air Quality City <br> Meterology |         ATGCN     | [Modeling Inter-station Relationships with Attentive Temporal Graph Convolutional Network for Air Quality Prediction](https://doi.org/10.1145/3437963.3441731) | None | WSDM 2021 <br>  B
| Traffic Flow |  WalkWLA  <br>  BikeNYC   <br>  TaxiNYC |         PDSTN     | [Predicting Crowd Flows via Pyramid Dilated Deeper Spatial-temporal Network](https://doi.org/10.1145/3437963.3441785) | None | WSDM 2021 <br>  B
| Traffic Flow | PeMSD4 <br> PeMSD8    |         AGCRN        | [Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting](https://proceedings.neurips.cc/paper/2020/hash/ce1aad92b939420fc17005e5461e6f48-Abstract.html) | [Pytorch](https://github.com/LeiBAI/AGCRN) | NIPS 2020 <br> A
| Multivariable | Electricity <br> Traffic  <br>  Wind <br>  Solar <br>  M4-Hourly  |         AST        | [Adversarial Sparse Transformer for Time Series Forecasting](https://proceedings.neurips.cc/paper/2020/hash/c6b8c8d762da15fa8dbbdfb6baf9e260-Abstract.html) | [Pytorch](https://github.com/hihihihiwsf/AST) | NIPS 2020 <br> A
| Multivariable |  METR-LA <br> PeMS-BAY  <br>  PeMS07 <br>  PeMS03 <br> PeMS04 ,etc |         StemGNN        | [Adversarial Sparse Transformer for Time Series Forecasting](https://proceedings.neurips.cc/paper/2020/hash/cdf6581cb7aca4b7e19ef136c6e601a5-Abstract.html) | [Pytorch](https://github.com/microsoft/StemGNN) | NIPS 2020 <br> A
| Multivariable | M4 <br> M3 <br> Tourism |         N-BEATS         | [N-BEATS: Neural basis expansion analysis for interpretable time series forecasting](https://openreview.net/forum?id=r1ecqn4YwB) | [Pytorch+Keras](https://github.com/philipperemy/n-beats) | ICLR 2020 <br> None But Top 
| Traffic Flow | Traffic <br> Energy <br> Electricity <br> Exchange  <br> METR-LA <br> PeMS-BAY   |         MTGNN        | [Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks](https://doi.org/10.1145/3394486.3403046) | [Pytorch](https://github.com/nnzhan/MTGNN) | KDD 2020 <br> A
| Traffic Flow | Taxi-NYC <br> Bike-NYC <br> CTM |         DSAN        | [Preserving Dynamic Attention for Long-Term Spatial-Temporal Prediction](https://doi.org/10.1145/3394486.3403118) | [TF](https://github.com/haoxingl/DSAN) | KDD 2020 <br> A
| Traffic Speed <br> Traffic Flow | Shenzhen  |         Curb-GAN        | [Curb-GAN: Conditional Urban Traffic Estimation through Spatio-Temporal Generative Adversarial Networks](https://doi.org/10.1145/3394486.3403127) | [Pytorch](https://github.com/Curb-GAN/Curb-GAN) | KDD 2020 <br> A
| Traffic Flow | TaxiBJ <br> CrowdBJ  <br> TaxiJN  <br> TaxiGY |        AutoST        | [AutoST: Efficient Neural Architecture Search for Spatio-Temporal Prediction](https://doi.org/10.1145/3394486.3403122) | None | KDD 2020 <br> A
| Traffic Volumn | W3-715 <br> E5-2907 |         HSTGCN        | [Hybrid Spatio-Temporal Graph Convolutional Network: Improving Traffic Prediction with Navigation Data](https://doi.org/10.1145/3394486.3403358) | None | KDD 2020 <br> A
| Multivariable| Xiamen <br> PeMS-BAY  |        GMAN        | [GMAN: A Graph Multi-Attention Network for Traffic Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/5477) | [TF](https://github.com/zhengchuanpan/GMAN)<br>  [Pytorch](https://github.com/VincLee8188/GMAN-PyTorch) | AAAI 2020 <br> A
| Multivariable | PeMS03 <br> PeMS04 <br> PeMS07 <br> PeMS08 |      STSGCN       | [Spatial-temporal synchronous graph convolutional networks: A new framework for spatial-temporal network data forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/5438) |  [Mxnet](https://github.com/Davidham3/STSGCN) <br>  [Pytorch](https://github.com/SmallNana/STSGCN_Pytorch) | AAAI 2020 <br> A
| Multivariable |  Traffic  <br>  Energy  <br> NASDAQ  |      MLCNN       | [Towards Better Forecasting by Fusing Near and Distant Future Visions](https://ojs.aaai.org/index.php/AAAI/article/view/5466) |  [Pytorch](https://github.com/smallGum/MLCNN-Multivariate-Time-Series) | AAAI 2020 <br> 
| Multivariable |  PeMS-S <br> PeMS-BAY <br> METR-LA  <br> BJF <br> BRF <br> BRF-L |      SLCNN       | [Spatio-temporal graph structure learning for traffic forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/5770) | None | AAAI 2020 <br> A 
| Traffic Speed | METR-LA <br> PeMS-BAY  |        MRA-BGCN        | [Multi-Range Attentive Bicomponent Graph Convolutional Network for Traffic Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/5758) | None | AAAI 2020 <br> A
| Metro Flow | HKMetro |       WDGTC     | [Tensor Completion for Weakly-Dependent Data on Graph for Metro Passenger Flow Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/5915) |  [TF](https://github.com/bonaldli/WDG_TC)  | AAAI 2020 <br> A
| Multivariable | MovingMNIST <br> TaxiBJ <br>  KTH |       SA-ConvLSTM     | [Self-Attention ConvLSTM for Spatiotemporal Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/6819) |  [TF](https://github.com/MahatmaSun1/SaConvSLTM)  [PyTorch](https://github.com/jerrywn121/TianChi_AIEarth)  | AAAI 2020 <br> A
| Metro Flow | SydneyMetro  |      MLC-PPF    | [Potential Passenger Flow Prediction-A Novel Study for Urban Transportation Development](https://ojs.aaai.org/index.php/AAAI/article/view/5819) |  None | AAAI 2020 <br> A
| Commuting Flow | Lodes <br> Pluto <br> OSRM  |     GMEL   | [Learning Geo-Contextual Embeddings for Commuting Flow Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/5425) |  [Pytorch](https://github.com/jackmiemie/GMEL)  | AAAI 2020 <br> A
| Multivariable | Traffic  <br>   Exchange  <br> Solar   |       DeepTrends  | [Tensorized LSTM with Adaptive Shared Memory for Learning Trends in Multivariate Time Series](https://ojs.aaai.org/index.php/AAAI/article/view/5496) | [TF](https://github.com/DerronXu/DeepTrends)    | AAAI 2020 <br> A
| Multivariable | Traffic  <br>   Electricity   <br> SmokeVideo   <br> PCSales <br> RawMaterials  |       BHT  | [Block Hankel Tensor ARIMA for Multiple Short Time Series Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/6032) | [Python](https://github.com/huawei-noah/BHT-ARIMA)    | AAAI 2020 <br> A
| Traffic Speed | PeMSD4 <br>  PeMSD7  <br> PeMSD8  |      LSGCN        | [LSGCN: Long Short-Term Traffic Prediction with Graph Convolutional Networks](https://dl.acm.org/doi/abs/10.5555/3491440.3491766) |  [TF](https://github.com/helanzhu/LSGCN) | IJCAI 2020 <br> A
| Traffic Flow  | BikeNYC <br> MobileBJ  |        CSCNet      | [A Sequential Convolution Network for Population Flow Prediction with Explicitly Correlation Modelling](https://dl.acm.org/doi/abs/10.5555/3491440.3491625) | None  | IJCAI 2020 <br> A
| Multivariable | USDCNY  <br>   USDKRW   <br> USDIDR   |       WATTNet  | [WATTNet: learning to trade FX via hierarchical spatio-temporal representation of highly multivariate time series](https://www.ijcai.org/proceedings/2020/0630.pdf) | [TF](https://github.com/pablovicente/keras-wattnet)    | IJCAI 2020 <br> A
| Fine-grained | CitiBikeNYC <br>  Div  <br> Metro  |      GACNN        | [Towards Fine-grained Flow Forecasting: A Graph Attention Approach for Bike Sharing Systems](https://doi.org/10.1145/3366423.3380097) | None | WWW 2020 <br> A
| Flow <br> Distribution | Austin <br>  Louisville  <br> Minneapolis  |      GCScoot        | [Dynamic Flow Distribution Prediction for Urban Dockless E-Scooter Sharing Reconfiguration](https://doi.org/10.1145/3366423.3380101) | None | WWW 2020 <br> A
|  Traffic Speed | METR-LA <br> PeMS-BAY  |      STGNN        | [Traffic Flow Prediction via Spatial Temporal Graph Neural Network](https://doi.org/10.1145/3366423.3380186) |  [Pytorch](https://github.com/LMissher/STGNN)  | WWW 2020 <br> A
| Traffic Speed | DiDiChengdu  |      STAG-GCN        | [Spatiotemporal Adaptive Gated Graph Convolution Network for Urban Traffic Flow Forecasting](https://doi.org/10.1145/3340531.3411894) |  [Pytorch](https://github.com/RobinLu1209/STAG-GCN) | CIKM 2020 <br> B
| Traffic Speed | METR-LA <br> PeMS-BAY   |     ST-GRAT       | [ST-GRAT: A Novel Spatio-temporal Graph Attention Networks for Accurately Forecasting Dynamically Changing Road Speed](https://doi.org/10.1145/3340531.3411940) |  [Pytorch](https://github.com/LMissher/ST-GRAT) | CIKM 2020 <br> B
| Traffic Flow | BJ-Taxi <br>  NYC-Taxi  <br>  NYC-Bike-1  <br> NYC-Bike-2 |    ST-CGA      | [Spatial-Temporal Convolutional Graph Attention Networks for Citywide Traffic Flow Forecasting](https://doi.org/10.1145/3340531.3411941) |  [Keras](https://github.com/jbdj-star/cga) | CIKM 2020 <br> B
| Traffic Flow | NYCBike  <br>   NYCTaxi    |       MT-ASTN  | [Multi-task Adversarial Spatial-Temporal Networks for Crowd Flow Prediction](https://doi.org/10.1145/3340531.3412054) | [Pytorch](https://github.com/MiaoHaoSunny/MT-ASTN)    | CIKM 2020 <br> B
| Traffic Speed | SFO  <br>   NYC    |     DIGC  | [Deep Graph Convolutional Networks for Incident-Driven Traffic Speed Prediction](https://doi.org/10.1145/3340531.3411873) |  None   | CIKM 2020 <br> B
| Metro Flow | SZMetro <br> HZMetro  |       STP-TrellisNets   | [STP-TrellisNets: Spatial-Temporal Parallel TrellisNets for Metro Station Passenger Flow Prediction](https://doi.org/10.1145/3340531.3411874) | None | CIKM 2020 <br> B
| Multivariable | Air Quality  <br>  BikeNYC  <br>  METR-LA |   AGSTN   | [AGSTN: Learning Attention-adjusted Graph Spatio-Temporal Networks for Short-term Urban Sensor Value Forecasting](https://ieeexplore.ieee.org/abstract/document/9338255) |  [Keras](https://github.com/l852888/AGSTN) | ICDM 2020 <br> B
| Traffic Speed | METR-LA <br> PeMS-BAY  |   FreqST   | [FreqST: Exploiting Frequency Information in Spatiotemporal Modeling for Traffic Prediction](https://ieeexplore.ieee.org/abstract/document/9338305) |  None | ICDM 2020 <br> B
| Traffic Flow | PEMSD3 <br>  PEMSD7 |   TSSRGCN   | [Tssrgcn: Temporal spectral spatial retrieval graph convolutional network for traffic flow forecasting](https://ieeexplore.ieee.org/abstract/document/9338393) |  None | ICDM 2020 <br> B
| Multivariable | Air Quality  <br>   DarkSky <br>    Geographic   |     DeepLATTE   | [Building Autocorrelation-Aware Representations for Fine-Scale Spatiotemporal Prediction](https://ieeexplore.ieee.org/abstract/document/9338402) | [Pytorch](https://github.com/spatial-computing/deeplatte-fine-scale-prediction)    | ICDM 2020 <br> B
| Traffic Flow  | XATaxi  <br>   BJTaxi <br>    PortoTaxi   |     ST-PEFs   | [Interpretable Spatiotemporal Deep Learning Model for Traffic Flow Prediction Based on Potential Energy Fields](https://ieeexplore.ieee.org/abstract/document/9338315) | None  | ICDM 2020 <br> B
| Traffic Speed <br> Flow   | SZSpeed  <br>   SZTaxi   |     cST-ML   | [cST-ML: Continuous Spatial-Temporal Meta-Learning for Traffic Dynamics Prediction](https://ieeexplore.ieee.org/abstract/document/9338315) | [Pytorch](https://github.com/yingxue-zhang/cST-ML)    | ICDM 2020 <br> B
| Multivariable | Electricity <br> Traffic  <br> Wiki <br> PeMSD7(M) |         DeepGLO       | [Think Globally, Act Locally: A Deep Neural Network Approach to High-Dimensional Time Series Forecasting](https://proceedings.neurips.cc/paper/2019/hash/3a0844cee4fcf57de0c71e9ad3035478-Abstract.html) | [Pytorch](https://github.com/rajatsen91/deepglo/tree/master/DeepGLO) | NIPS 2019 <br> A
| Multivariable | Electricity <br> Traffic  <br> Solar <br> M4 <br> Wind |         LogSparse       | [Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting](https://proceedings.neurips.cc/paper/2019/hash/6775a0635c302542da2c32aa19d86be0-Abstract.html) | [Pytorch](https://github.com/mlpotter/Transformer_Time_Series) | NIPS 2019 <br> A
| Multivariable  | Synthetic <br> ECG5000  <br> Traffic  |        DILATE      | [Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models](https://proceedings.neurips.cc/paper/2019/hash/466accbac9a66b805ba50e42ad715740-Abstract.html) | [Pytorch](https://github.com/vincent-leguen/DILATE)  | NIPS 2019 <br> A
| Traffic Flow  | Earthquake  |        DeepUrbanEvent      | [DeepUrbanEvent: A System for Predicting Citywide Crowd Dynamics at Big Events](https://doi.org/10.1145/3292500.3330996) | [Keras](https://github.com/deepkashiwa/DeepUrbanEvent)  | KDD 2019 <br> A
| Traffic Flow <br> Speed | TDrive <br>  METR-LA   |         ST-MetaNet        | [Urban Traffic Prediction from Spatio-Temporal Data Using Deep Meta Learning](https://doi.org/10.1145/3292500.3330884) | [Mxnet](https://github.com/panzheyi/ST-MetaNet) | KDD 2019 <br> A
| Multivariable  | Rossman  <br> Walmart <br> Electricity <br> Traffic <br> Parts  |        ARU      | [Streaming Adaptation of Deep Forecasting Models using Adaptive Recurrent Units](https://doi.org/10.1145/3292500.3330996) | [TF](https://github.com/pratham16cse/ARU)  | KDD 2019 <br> A
| Multivariable  | Air Quality   |        AccuAir      | [AccuAir: Winning Solution to Air Quality Prediction for KDD Cup 2018](https://doi.org/10.1145/3292500.3330787) | None | KDD 2019 <br> A
| Traffic Flow  | Simulated  <br> RoadTraffic <br>  Wikipedia |        ERMreg      | [Regularized Regression for Hierarchical Forecasting Without Unbiasedness Conditions](https://doi.org/10.1145/3292500.3330976) | None | KDD 2019 <br> A
| Multivariable <br> under event | Climate  <br> Stock <br>  Pseudo |        EVL      | [Modeling Extreme Events in Time Series Prediction](https://doi.org/10.1145/3292500.3330896) |None | KDD 2019 <br> A
| Traffic Flow | PeMSD4 <br>  PeMSD8 <br> METR-LA   |         ASTGCN        | [Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/3881) | [Mxnet](https://github.com/Davidham3/ASTGCN) | AAAI 2019 <br> A
| Traffic Flow <br> Speed | NYC <br>  PeMSD(M)  |         DGCNN        | [Dynamic spatial-temporal graph convolutional neural networks for traffic forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/3877) | None  | AAAI 2019 <br> A
| Traffic Speed | METR-LA  <br>  PeMS-BAY  |         Res-RGNN        | [Gated Residual Recurrent Graph Neural Networks for Traffic Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/3821) | None  | AAAI 2019 <br> A
| Traffic FLow | NYC-Taxi <br>  NYC-Bike  |        STDN      | [Revisiting Spatial-Temporal Similarity: A Deep Learning Framework for Traffic Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/4511) | [Keras](https://github.com/tangxianfeng/STDN)  | AAAI 2019 <br> A
| Traffic Flow   | MobileBJ  <br> BikeNYC  |        DeepSTN+      | [DeepSTN+: context-aware spatial-temporal neural network for crowd flow prediction in metropolis](https://doi.org/10.1609/aaai.v33i01.33011020) | [TF](https://github.com/tsinghua-fib-lab/DeepSTN)  | AAAI 2019 <br> A
| Traffic Flow   | NYC-Taxi <br>  NYC-Bike |       STDN   | [Revisiting spatial-temporal similarity: a deep learning framework for traffic prediction](https://doi.org/10.1609/aaai.v33i01.33015668) | [Keras](https://github.com/tangxianfeng/STDN)  | AAAI 2019 <br> A
| Traffic Speed   | METR-LA  <br>  PeMS-BAY |       Res-RGNN    | [Gated residual recurrent graph neural networks for traffic prediction](https://doi.org/10.1609/aaai.v33i01.3301485) | None  | AAAI 2019 <br> A
| Traffic FLow | MetroBJ  <br>  BusBJ  <br> TaxiBJ |        GSTNet      | [GSTNet: Global Spatial-Temporal Network for Traffic Flow Prediction](https://www.ijcai.org/Proceedings/2019/0317.pdf) | [Pytorch](https://github.com/WoodSugar/GSTNet)  | IJCAI 2019 <br> A
| Traffic Speed  | METR-LA <br> PeMS-BAY  |        GWN      | [Graph WaveNet for Deep Spatial-Temporal Graph Modeling](https://doi.org/10.24963/ijcai.2019/264) | [Pytorch](https://github.com/nnzhan/Graph-WaveNet)  | IJCAI 2019 <br> A
| Traffic Flow  | DidiSY <br> BikeNYC <br>  TaxiBJ |        STG2Seq      | [STG2Seq: Spatial-Temporal Graph to Sequence Model for Multi-step Passenger Demand Forecasting](https://openreview.net/forum?id=Ein6fZbizNZ) | [TF](https://github.com/LeiBAI/STG2Seq)  | IJCAI 2019 <br> A
| Multivariable | GHL <br>  Electricity  <br>TEP |       DyAt   | [DyAt Nets: Dynamic Attention Networks for State Forecasting in Cyber-Physical Systems](https://www.ijcai.org/Proceedings/2019/0441.pdf) | [Pytorch](https://github.com/nmuralid1/DynamicAttentionNetworks)  | IJCAI 2019 <br> A
| Multivariable | Air Quality |       MGED   | [Multi-Group Encoder-Decoder Networks to Fuse Heterogeneous Data for Next-Day Air Quality Prediction](https://www.ijcai.org/proceedings/2019/0603.pdf) | None | IJCAI 2019 <br> A
| Traffic Volumn  | Chicago <br> Boston  |        MetaST      | [Learning from Multiple Cities: A Meta-Learning Approach for Spatial-Temporal Prediction](https://doi.org/10.1145/3308558.3313577) | [TF](https://github.com/huaxiuyao/MetaST)  | WWW 2019 <br> A
| TrafficPred <br> imputation |GZSpeed <br> HZMetro <br> Seattle <br> London |       BTF   | [Bayesian Temporal Factorization for Multidimensional Time Series Prediction](https://ieeexplore.ieee.org/abstract/document/9380704) | [Python](https://github.com/nmuralid1/DynamicAttentionNetworks)  | TPAMI 2019 <br> A
| Multivariable | Gas Station |       DSANet   | [DSANet: Dual Self-Attention Network for Multivariate Time Series Forecasting](https://doi.org/10.1145/3357384.3358132) | [Pytorch](https://github.com/bighuang624/DSANet)  | CIKM 2019 <br> B
| Multivariable | Solar <br> Traffic <br> Exchange <br> Electricity <br> PeMS ,etc |       Study   | [Experimental Study of Multivariate Time Series Forecasting Models](https://doi.org/10.1145/3357384.3357826) | None | CIKM 2019 <br> B
| Traffic Speed | DiDiCD <br> DiDiXA  |   BTRAC   | [Boosted Trajectory Calibration for Traffic State Estimation](https://ieeexplore.ieee.org/abstract/document/8970880) | None  | ICDM 2019 <br> B
| Multivariable | Photovoltaic  |       MTEX-CNN   | [MTEX-CNN: Multivariate Time Series EXplanations for Predictions with Convolutional Neural Networks](https://ieeexplore.ieee.org/abstract/document/8970899) | [Pytorch](https://github.com/duyanhpham-brs/XAI-Multivariate-Time-Series)  | ICDM 2019 <br> B
| Traffic Speed | BJER4 <br> PeMSD7(M)  <br>  PeMSD7(L)  |        STGCN      | [Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting](https://openreview.net/forum?id=SkNeyVzOWB) | [TF](https://github.com/VeritasYin/STGCN_IJCAI-18) [Mxnet](https://github.com/Davidham3/STGCN)  [Pytorch1](https://github.com/FelixOpolka/STGCN-PyTorch)  [Pytorch2](https://github.com/hazdzz/STGCN) [Pytorch3](https://github.com/Aguin/STGCN-PyTorch)   | IJCAI 2018 <br> A
| Traffic Speed | METR-LA <br> PeMS-BAY  |      DCRNN  | [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting](https://openreview.net/forum?id=SJiHXGWAZ) | [TF](https://github.com/liyaguang/DCRNN) [Pytorch](https://github.com/chnsh/DCRNN_PyTorch)  |ICLR 2018 <br> None But Top 




# [Multivariable Probabilistic Time Series Forecasting](#content)
|  Task  |    Data |   Model  | Paper   |    Code    |   Publication    |
| :-: | :-: | :-: | :-: | :-: | - |
| Paper Nums:39 | <img width=150/> | <img width=220/>  |   |   |   <img width=300/> | 
| probability & <br> Point & <br> Others |   electricity  <br>  Yacht <br> Boston, etc |         AQF        | [Autoregressive Quantile Flows for Predictive Uncertainty Estimation](https://openreview.net/forum?id=z1-I6rOKv1S) | None | ICLR 2022 <br> None But Top 
| probability  | IRIS <br> Digits <br> EightSchools    |         EMF        | [Embedded-model flows: Combining the inductive biases of model-free deep learning and explicit probabilistic modeling](https://openreview.net/forum?id=9pEJSVfDbba) | [Pytorch](https://github.com/gisilvs/EmbeddedModelFlows) | ICLR 2022 <br> None But Top 
| probability  | Bike Sharing <br> UCI <br> NYU Depth v2  |         NatPN        | [Natural Posterior Network: Deep Bayesian Predictive Uncertainty for Exponential Family Distributions](https://www.in.tum.de/daml/natpn/) | [Pytorch](https://github.com/borchero/natural-posterior-network) | ICLR 2022 <br> None But Top 
| probability  | Carbon <br> Concrete <br> Energy <br> Housing,etc  |      β−NLL      | [On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks](https://openreview.net/forum?id=aPOpXlnV1T) | [Pytorch](https://github.com/martius-lab/beta-nll) | ICLR 2022 <br> None But Top
| probability & Point | Sichuan <br> Panama |         PrEF        | [PrEF: Probabilistic Electricity Forecasting via Copula-Augmented State Space Model](https://aaai-2022.virtualchair.net/poster_aisi7128) | None | AAAI 2022<br>A
| probability  | Electricity <br> Traffic <br> Wiki  <br> M4   |     ISQF     | [Learning Quantile Function without Quantile Crossing for Distribution-free Time Series Forecasting](https://arxiv.org/abs/2111.06581) | [GluonTS](https://github.com/awslabs/gluon-ts/blob/4d73911f6aae5079ad228b504ab8edaa369ad04c/src/GluonTS/mx/distribution/isqf.py) | AISTAT 2022 <br> C But Top
| probability  |  M4 <br> Traffic <br>  Electricity    |     Robust     | [Robust Probabilistic Time Series Forecasting](https://arxiv.org/abs/2202.11910) | [GluonTS](https://github.com/tetrzim/robust-probabilistic-forecasting)  | AISTAT 2022 <br> C But Top
| probability  | Electricity  <br> Traffic <br> M4     |     MQF     | [Multivariate Quantile Function Forecaster](https://arxiv.org/pdf/2202.11316.pdf) | [GluonTS](https://github.com/awslabs/gluon-ts/tree/master/src/GluonTS/torch/model/mqf2)  | AISTAT 2022 <br> C But Top
| probability | MIMIC-III <br> EEG <br> COVID-19  |        CF-RNN      | [Conformal Time-series Forecasting](https://proceedings.neurips.cc/paper/2021/hash/312f1ba2a72318edaaa995a67835fad5-Abstract.html) |  [Pytorch](https://github.com/kamilest/conformal-rnn) | NIPS 2021<br>A
| probability | CDC Flu  |       EPIFNP     | [When in Doubt: Neural Non-Parametric Uncertainty Quantification for Epidemic Forecasting](https://proceedings.neurips.cc/paper/2021/hash/a4a1108bbcc329a70efa93d7bf060914-Abstract.html) |  None | NIPS 2021<br>A
| probability | Basketball  <br>  Weather|       GLIM     | [Probability Paths and the Structure of Predictions over Time](https://proceedings.neurips.cc/paper/2021/hash/7f53f8c6c730af6aeb52e66eb74d8507-Abstract.html) |   [R](https://github.com/ItsMrLin/probability-paths) | NIPS 2021<br>A
| probability | Facebook  <br>  Meps <br> Star <br> Bike ,etc |       LSF     | [Probabilistic Forecasting: A Level-Set Approach](https://proceedings.neurips.cc/paper/2021/hash/32b127307a606effdcc8e51f60a45922-Abstract.html) |   [GluonTS](https://github.com/awslabs/gluon-ts/tree/master/src/GluonTS/model/rotbaum) | NIPS 2021<br>A
| probability | Solar  <br>  Electricity <br> Traffic  <br> Taxi <br> Wikipedia |       ProTran     | [Probabilistic Transformer For Time Series Analysis](https://proceedings.neurips.cc/paper/2021/hash/c68bd9055776bf38d8fc43c0ed283678-Abstract.html) |   None | NIPS 2021<br>A
| probability | Exchange <br> Solar <br> Electricity <br> Traffic <br>  Taxi  <br>   Wiki  |        TimeGrad      | [Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting](http://proceedings.mlr.press/v139/rasul21a.html) |  [Pytorch](https://github.com/zalandoresearch/pytorch-ts) | ICML 2021<br>A
| probability & Point | PeMSD3 <br> PeMSD4 <br> PeMSD7 <br> PeMSD8 <br>  Electricity  <br>   Traffic , etc |         AGCGRU        | [RNN with Particle Flow for Probabilistic Spatio-temporal Forecasting](https://proceedings.mlr.press/v139/pal21b.html) |  [TF](https://github.com/networkslab/rnn_flow) | ICML 2021<br>A
| probability | Tourism <br> Labour <br> Traffic <br> Wiki <br>  Electricity  <br>   Traffic , etc |         Hier-E2E        | [End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series](http://proceedings.mlr.press/v139/rangapuram21a.html) |  [MXNet](https://github.com/rshyamsundar/GluonTS-hierarchical-ICML-2021) | ICML 2021<br>A
| probability | Sine <br> MNIST <br> Billiards <br> S&P <br>  Stock   |        Whittle      | [Whittle Networks: A Deep Likelihood Model for Time Series](http://proceedings.mlr.press/v139/yu21c.html) | [TF](https://github.com/ml-research/WhittleNetworks) | ICML 2021<br>A
| probability | METR-LA <br> PeMS-BAY <br> PMU   |        GTS      | [Discrete Graph Structure Learning for Forecasting Multiple Time Series](https://openreview.net/forum?id=WEHSlH5mOk) | [Pytorch](https://github.com/chaoshangcs/GTS) | ICLR 2021 <br> None But Top 
| probability & Point| Exchange <br>Solar <br> Electricity <br> Traffic <br> Taxi  <br> Wikipedia |        flow      | [Multivariate Probabilistic Time Series Forecasting via Conditioned Normalizing Flows](https://openreview.net/forum?id=WiGQBFuVRv) | [Pytorch](https://github.com/zalandoresearch/pytorch-ts) | ICLR 2021 <br> None But Top 
| probability | MNIST <br> PhysioNet2012  |        PNCNN      | [Probabilistic Numeric Convolutional Neural Networks](https://openreview.net/forum?id=T1XmO8ScKim) | None  | ICLR 2021 <br> None But Top 
| probability & Point | Energy <br> Wine <br> Power <br> MSD, etc |         PGBM        | [Probabilistic Gradient Boosting Machines for Large-Scale Probabilistic Regression](https://dl.acm.org/doi/10.1145/3447548.3467278) |  [Pytorch](https://github.com/elephaint/pgbm) | KDD 2021<br>A
| probability | DiDICD   |        TrajNet      | [TrajNet: A Trajectory-Based Deep Learning Model for Traffic Prediction](https://doi.org/10.1145/3447548.3467236) | None | KDD 2021 <br> A
| probability | Air Quality  <br>  METR-LA <br>  COVID-19  |        UQ      | [Quantifying Uncertainty in Deep Spatiotemporal Forecasting](https://doi.org/10.1145/3447548.3467325) | [Pytorch](https://github.com/DongxiaW/Quantifying_Uncertainty_in_Deep_Spatiotemporal_Forecasting) | KDD 2021 <br> A
| probability  | Electricity <br> Traffic <br> Environment <br> Air Quality <br> Dewpoint,etc|        VSMHN      | [Synergetic Learning of Heterogeneous Temporal Sequences for Multi-Horizon Probabilistic Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/17023) | [Pytorch](https://github.com/longyuanli/VSMHN) | AAAI 2021 <br> A
| probability & Point | Traffic <br> Electricity <br> Wiki <br> Solar <br> Taxi |        TLAE      | [Temporal Latent Auto-Encoder: A Method for Probabilistic Multivariate Time Series Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/17101) | None | AAAI 2021 <br> A
| probability  | Patient EHR <br> Public Health |        UNITE      | [UNITE: Uncertainty-based Health Risk Prediction Leveraging Multi-sourced Data](https://doi.org/10.1145/3442381.3450087) | [Pytorch](https://github.com/Chacha-Chen/UNITE) | WWW 2021 <br> A
| probability  | Exchange <br> Solar <br> Electricity  <br> Traffic  <br>  Wiki  |     ARSGLS     | [Deep Rao-Blackwellised Particle Filters for Time Series Forecasting](https://proceedings.neurips.cc/paper/2020/hash/afb0b97df87090596ae7c503f60bb23f-Abstract.html) | None | NIPS 2020 <br> A
| probability  | Electricity <br> Traffic <br> Wind  <br> Solar  <br>  M4  |     AST     | [Adversarial Sparse Transformer for Time Series Forecasting](https://proceedings.neurips.cc/paper/2020/hash/c6b8c8d762da15fa8dbbdfb6baf9e260-Abstract.html) | [Pytorch](https://github.com/hihihihiwsf/AST)  | NIPS 2020 <br> A
| probability  | Traffic <br> Electricity   |     STRIPE     | [Probabilistic Time Series Forecasting with Shape and Temporal Diversity](https://papers.nips.cc/paper/2020/hash/2f2b265625d76a6704b08093c652fd79-Abstract.html) | [Pytorch](https://github.com/vincent-leguen/STRIPE)  | NIPS 2020 <br> A
| probability  | Exchange <br> Solar <br> Electricity  <br> Wiki  <br>  Traffic  |     NKF     | [Normalizing Kalman Filters for Multivariate Time Series Analysis](https://proceedings.neurips.cc/paper/2020/hash/1f47cef5e38c952f94c5d61726027439-Abstract.html) |  None  | NIPS 2020 <br> A
| probability  |  S&P 500 <br> Electricity   |     Monte-Carlo     | [Adversarial Attacks on Probabilistic Autoregressive Forecasting Models](https://proceedings.mlr.press/v119/dang-nhu20a.html) | [Pytorch](https://github.com/eth-sri/probabilistic-forecasts-attacks)  | ICML 2020 <br> A
| probability  |  Boston <br> Concrete  <br>Energy <br> Kin8nm <br>  Naval, etc  |    NGBoost    | [NGBoost: Natural Gradient Boosting for Probabilistic Prediction](http://proceedings.mlr.press/v119/duan20a.html) | [Python](https://github.com/stanfordmlgroup/ngboost)  | ICML 2020 <br> A
| probability  |  Physionet <br> NHIS   |    DME    | [Deep Mixed Effect Model Using Gaussian Processes: A Personalized and Reliable Prediction for Healthcare](https://ojs.aaai.org/index.php/AAAI/article/view/5773) | [Pytorch](https://github.com/jik0730/Deep-Mixed-Effect-Model-using-Gaussian-Processes)  | AAAI 2020 <br> A
| probability  |  Exchange <br> Solar <br>  Electricity  <br>  Traffic  <br>  NYCTaxi <br> Wikipedia  |    copula    | [High-dimensional multivariate forecasting with low-rank Gaussian Copula Processes](https://proceedings.neurips.cc/paper/2019/hash/0b105cf1504c4e241fcc6d519ea962fb-Abstract.html) | [GluonTS](https://github.com/mbohlkeschneider/gluon-ts/tree/mv_release)  | NIPS 2019 <br> A
| probability  |  Electricity <br> Traffic <br>  NYCTaxi  <br>  Uber   |    DF    | [Deep Factors for Forecasting](https://proceedings.mlr.press/v97/wang19k.html) | None | ICML 2019 <br> A
| probability  |  Weather   |    DUQ    | [Deep Uncertainty Quantification: A Machine Learning Approach for Weather Forecasting](https://doi.org/10.1145/3292500.3330704) | [Keras](https://github.com/BruceBinBoxing/Deep_Learning_Weather_Forecasting)  | KDD 2019 <br> A
| probability  |  JD50K   |    framework    | [Multi-Horizon Time Series Forecasting with Temporal Attention Learning](https://doi.org/10.1145/3292500.3330662) | None  | KDD 2019 <br> A
| probability  |  MIMIC-III   |    TPF    | [Temporal Probabilistic Profiles for Sepsis Prediction in the ICU](https://doi.org/10.1145/3292500.3330747) | None  | KDD 2019 <br> A
| probability  | Electricity <br> Traffic <br>  Wiki  <br>  Dom    |    SQF    | [Probabilistic Forecasting with Spline Quantile Function RNNs](https://proceedings.mlr.press/v89/gasthaus19a.html) | None  | AISTAT 2019 <br> C But Top
| probability  | More |         More        | [https://github.com/zzw-zwzhang/Awesome-of-Time-Series-Prediction](https://github.com/zzw-zwzhang/Awesome-of-Time-Series-Prediction) |  More |  


<!-- 
| probability  | Electricity <br> Traffic <br>  Wiki  <br>  Dom    |    SQF    | [Probabilistic Forecasting with Spline Quantile Function RNNs](https://proceedings.mlr.press/v89/gasthaus19a.html) | None  | AISTAT 2019 <br> A

| probability  |  Electricity <br> Traffic <br>  NYCTaxi  <br>  Uber   |    framework    | [Multi-Horizon Time Series Forecasting with Temporal Attention Learning](https://doi.org/10.1145/3292500.3330662) | [Pytorch](https://github.com/mbohlkeschneider/gluon-ts/tree/mv_release)  | KDD 2019 <br> A -->



# [Time Series Imputation](#content)
|  Task  |    Data |   Model  | Paper   |    Code    |   Publication    |
| :-: | :-: | :-: | :-: | :-: | - |
| Paper Nums: 22  | <img width=150/> | <img width=220/>  |   |   |   <img width=300/> | 
| Imputation |  Air Quality <br> METR-LA <br> PeMS-BAY <br> CER-E  |         GRIN        | [Filling the G_ap_s-Multivariate Time Series Imputation by Graph Neural Networks](https://openreview.net/forum?id=kOu3-S3wJ7) |  [Pytorch](https://github.com/Graph-Machine-Learning-Group/grin) | ICLR 2022 <br> None But Top 
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
| Imputation  | Zero <br> MICE <br>  SoftImpute  <br>  GMMC <br> GAIN   |    SN    | [Why Not to Use Zero Imputation? Correcting Sparsity Bias in Training Neural Networks](https://openreview.net/forum?id=BylsKkHYvH) | [Future](https://github.com/JoonyoungYi/sparsity-normalization)  | ICLR 20 <br> None But Top
| Imputation  | Beijing Air <br> PhysioNet <br>  Porto Taxi <br>  London Weather  |   LGnet   | [Joint Modeling of Local and Global Temporal Dynamics for Multivariate Time Series Forecasting with Missing Values](https://ojs.aaai.org/index.php/AAAI/article/view/6056) | None | AAAI 20 <br>A
| Imputation  | Sydney <br> Melbourne <br>  Brisbane <br>  Perth, etc   |    SMV-NMF    | [A spatial missing value imputation method for multi-view urban statistical data](https://www.ijcai.org/Proceedings/2020/0182.pdf) | [Matlab](https://github.com/SMV-NMF/SMV-NMF)  | IJCAI 20 <br>A
| Imputation  | PhysioNet <br> Air Quality <br>  Wind  |   GANGRUI   | [Adversarial Recurrent Time Series Imputation](https://ieeexplore.ieee.org/abstract/document/9158560/) | None | TNNLS 20 <br>B
| Imputation  | Healthcare <br> Climate  |   GRU-ODE-Bayes   | [GRU-ODE-Bayes: Continuous Modeling of Sporadically-Observed Time Series](https://proceedings.neurips.cc/paper/2019/hash/455cb2657aaa59e32fad80cb0b65b9dc-Abstract.html) | [Pytorch](https://github.com/edebrouwer/gru_ode_bayes) | NIPS 19 <br>A
| Imputation  |  Toy |   LatenODE   | [Latent Ordinary Differential Equations for Irregularly-Sampled Time Series](https://proceedings.neurips.cc/paper/2019/hash/42a6845a557bef704ad8ac9cb4461d43-Abstract.html) | [Pytorch](https://github.com/YuliaRubanova/latent_ode) | NIPS 19 <br>A
| Imputation  |  Sines <br>  Stocks<br> Energy <br> Events |   TimeGAN   | [Time-series Generative Adversarial Networks](https://proceedings.neurips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html) | [TF](https://github.com/jsyoon0823/TimeGAN) | NIPS 19 <br>A
| Imputation  |  MIMIC-III  <br>  UWaveGesture  |   Inter-net   | [Interpolation-Prediction Networks for Irregularly Sampled Time Series](https://openreview.net/forum?id=r1efr3C9Ym) | [Keras](https://github.com/mlds-lab/interp-net) | ICLR 19 <br>None But Top
| Imputation  | PhysioNet  <br>  KDD2018  |  E2gan   | [E2gan: End-to-end generative adversarial network for multivariate time series imputation](https://www.ijcai.org/Proceedings/2019/0429.pdf) | [TF](https://github.com/Luoyonghong/E2EGAN) | IJCAI 19 <br>A
| Imputation  | EC  <br>  RV  |  STI   | [How Do Your Neighbors Disclose Your Information: Social-Aware Time Series Imputation](https://doi.org/10.1145/3308558.3313714) | [Pytorch](https://github.com/tomstream/STI) | WWW 19 <br>A



<!-- 
| Imputation  | EC  <br>  RV  |  STI   | [How Do Your Neighbors Disclose Your Information: Social-Aware Time Series Imputation](https://doi.org/10.1145/3308558.3313714) | [Pytorch](https://github.com/tomstream/STI) | WWW 19 <br>A

| probability  |  Electricity <br> Traffic <br>  NYCTaxi  <br>  Uber   |    framework    | [Multi-Horizon Time Series Forecasting with Temporal Attention Learning](https://doi.org/10.1145/3292500.3330662) | [Pytorch](https://github.com/edebrouwer/gru_ode_bayes)  | KDD 2019 <br> A 

-->


# [Time Series Anomaly Detection](#content)
|  Task  |    Data |   Model  | Paper   |    Code    |   Publication    |
| :-: | :-: | :-: | :-: | :-: | - |
| Paper Nums: 30  | <img width=90/> |      |     |     |  <img width=320/> | 
|  Anomaly Detection | SMD <br> PSM <br> MSL&SMAP <br> SWaT NeurIPS-TS <br> |         Anomaly Transformer        | [Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy](https://openreview.net/forum?id=LzQQ89U1qm_) | [Pytorch](https://github.com/spencerbraun/anomaly_transformer_pytorch) | ICLR 2022 <br> None But Top 
| Density Estimation & Anomaly Detection | PMU-B <br> PMU-C <br> SWaT <br> METR-LA |         GANF        | [Graph-Augmented Normalizing Flows for Anomaly Detection of Multiple Time Series](https://openreview.net/forum?id=45L_dgP48Vd) | [Pytorch](https://github.com/EnyanDai/GANF) | ICLR 2022 <br> None But Top 
|  Anomaly Detection |    |          | [Anomaly Detection for Tabular Data with Internal Contrastive Learning](https://openreview.net/forum?id=_hszZbt46bT) | None | ICLR 2022 <br> None But Top 
|  Anomaly Detection |     |       AnomalyKiTS   | [AnomalyKiTS-Anomaly Detection Toolkit for Time Series](https://aaai-2022.virtualchair.net/poster_dm318) | None | AAAI 2022<br>A
|  Anomaly Detection |  SWaT <br> WADI <br> MSL <br> SMAP <br> SMD  |       PA   | [Towards a Rigorous Evaluation of Time-Series Anomaly Detection](https://aaai-2022.virtualchair.net/poster_aaai2239) |  None  | AAAI 2022 <br> A
|  Anomaly Detection | Business|       SLA-VAE       | [A Semi-Supervised VAE Based Active Anomaly Detection Framework in Multivariate Time Series for Online Systems](https://doi.org/10.1145/3485447.3511984) | None| WWW 2022 <br> A
|  Anomaly Detection |  KDDCUP99 <br>  NSL   <br>  UNSW, etc |      MemStream       | [MemStream: Memory-Based Streaming Anomaly Detection](https://doi.org/10.1145/3485447.3511984) |  [Pytorch](https://github.com/Stream-AD/MemStream)| WWW 2022 <br> A
|  Anomaly Detection | NAB <br> UCR <br> MBA <br> SMAP <br>  MSL <br> SWaT <br> WADI <br> SMD <br> MSDS   |       TranAD     | [TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data](https://arxiv.org/abs/2201.07284) | [Pytorch](https://github.com/imperial-qore/TranAD) | VLDB 2022 <br> A
|  Anomaly Detection |  SMD  |       FDRC   | [Online false discovery rate control for anomaly detection in time series](https://dl.acm.org/doi/10.1145/3447548.3467075) | None  | NIPS 2021 <br> A
|  Anomaly Detection |  SWaT <br> WADI <br> SMD <br> ASD  |       InterFusion   | [Multivariate Time Series Anomaly Detection and Interpretation using Hierarchical Inter-Metric and Temporal Embedding](https://papers.nips.cc/paper/2021/hash/def130d0b67eb38b7a8f4e7121ed432c-Abstract.html) |  [TF](https://github.com/zhhlee/InterFusion)  | KDD 2021 <br> A
|  Anomaly Detection |  SMD <br> SWaT <br> PSM <br> BKPI  |       RANSynCoders   | [Practical Approach to Asynchronous Multivariate Time Series Anomaly Detection and Localization](https://doi.org/10.1145/3447548.3467174) |  [TF](https://github.com/eBay/RANSynCoders)  | KDD 2021 <br> A
|  Anomaly Detection |  PUMP <br> WADI <br> SWaT  |       NSIBF   | [Time Series Anomaly Detection for Cyber-physical Systems via Neural System Identification and Bayesian Filtering](https://doi.org/10.1145/3447548.3467137) |  [TF](https://github.com/NSIBF/NSIBF)  | KDD 2021 <br> A
|  Anomaly Detection |  SWaT <br> WADI   |       GDN   | [Graph Neural Network-Based Anomaly Detection in Multivariate Time Series](https://ojs.aaai.org/index.php/AAAI/article/view/16523) |  [Pytorch](https://github.com/d-ailin/GDN)  | AAAI 2021 <br> A
|  Anomaly Detection |  KPI <br> Yahoo   |      FluxEV   | [FluxEV: A Fast and Effective Unsupervised Framework for Time-Series Anomaly Detection](https://doi.org/10.1145/3437963.3441823) |  None   | WSDM 2021<br> B
|  Earthquakes Detection |  NIED   |       CrowdQuake   | [A Networked System of Low-Cost Sensors for Earthquake Detection via Deep Learning](https://doi.org/10.1145/3394486.3403378) |  [TF](https://github.com/xhuang2016/Seismic-Detection)    | KDD 2020 <br> A
|  Anomaly Detection |  SWaT  <br> WADI <br> SMD  <br>  SMAP <br> MSL <br>  Orange |       USAD   | [USAD: UnSupervised Anomaly Detection on Multivariate Time Series](https://doi.org/10.1145/3394486.3403392) |   [Pytorch](https://github.com/manigalati/usad)   | KDD 2020 <br> A
|  Anomaly Detection |  NYC  |       CHAT   | [Cross-interaction hierarchical attention networks for urban anomaly prediction](https://dl.acm.org/doi/abs/10.5555/3491440.3492041) |  None  | IJCAI 2020 <br> A
|  Anomaly Detection |  NYC-Bike  <br> NYC-Taxi <br> Weather <br>  NYC-POI <br> NYC-Anomaly |       DST-MFN   | [Deep Spatio-Temporal Multiple Domain Fusion Network for Urban Anomalies Detection](https://doi.org/10.1145/3292500.3330672) |  None  | CIKM 2020 <br> B
|  Anomaly Detection | SMAP <br> MSL <br> TSA  |      MTAD-GAT | [Multivariate Time-Series Anomaly Detection via Graph Attention Network](https://ieeexplore.ieee.org/abstract/document/9338317) |   [TF](https://github.com/mangushev/mtad-gat)  [Pytorch](https://github.com/ML4ITS/mtad-gat-pytorch) | ICDM 2020 <br> B
|  Anomaly Detection |  SMAP  <br> MSL <br> SMD  |       OmniAnomaly   | [Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network](https://doi.org/10.1145/3292500.3330672) |   [TF](https://github.com/NetManAIOps/OmniAnomaly)   | KDD 2019 <br> A
|  Anomaly Detection |  GeoLife  <br> TST   |       IRL-ADU   | [Sequential Anomaly Detection using Inverse Reinforcement Learning](https://doi.org/10.1145/3292500.3330932) |   None  | KDD 2019 <br> A
|  Anomaly Detection |  donors  <br> census  <br> fraud <br> celeba ,etc |      DevNet  | [Deep Anomaly Detection with Deviation Networks](https://doi.org/10.1145/3292500.3330871) |   [Keras](https://github.com/GuansongPang/deviation-network) [Pytorch](https://github.com/Choubo/deviation-network-image) | KDD 2019 <br> A
|  Anomaly Detection | KPI <br> Yahoo <br> Microsoft  |      SR-CNN  | [Time-Series Anomaly Detection Service at Microsoft](https://doi.org/10.1145/3292500.3330680) |   None | KDD 2019 <br> A
|  Anomaly Detection |  power plant |      MSCRED  | [A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data](https://ojs.aaai.org/index.php/AAAI/article/view/3942) |   [TF](https://github.com/7fantasysz/MSCRED) | AAAI 2019 <br> A
|  Anomaly Detection | ECG <br> Motion |      BeatGAN  | [BeatGAN: Anomalous Rhythm Detection using Adversarially Generated Time Series](https://www.ijcai.org/Proceedings/2019/0616.pdf) |   [Pytorch](https://github.com/hi-bingo/BeatGAN) | IJCAI 2019 <br> A
|  Anomaly Detection | NAB <br> ECG |      OED  | [Outlier Detection for Time Series with Recurrent Autoencoder Ensembles](https://www.ijcai.org/proceedings/2019/0378.pdf) |   [TF](https://github.com/tungk/OED) | IJCAI 2019 <br> A
|  Anomaly Detection | KPIs |      Buzz  | [Unsupervised Anomaly Detection for Intricate KPIs via Adversarial Training of VAE](https://ieeexplore.ieee.org/abstract/document/8737430) |   [TF](https://github.com/yantijin/Buzz) | INFOCOM 2019 <br> A
|  Anomaly Detection | KDDCUP <br> Thyroid <br> Arrhythmia  <br> KDDCUP-Rev |      DAGMM  | [Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection](https://openreview.net/forum?id=BJJLHbb0-) |   [Pytorch](https://github.com/danieltan07/dagmm) | ICLR 2018 <br> A
|  Anomaly Detection | SMAP <br> MSL  |      telemanom  | [Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding](https://doi.org/10.1145/3219819.3219845) |   [TF](https://github.com/khundman/telemanom) | KDD 2018 <br> A
|  Anomaly Detection | AD <br> AID362 <br> aPascal  <br>  BM , etc|      CINFO | [Sparse Modeling-Based Sequential Ensemble Learning for Effective Outlier Detection in High-Dimensional Numeric Data](https://ojs.aaai.org/index.php/AAAI/article/view/11692) |    [Matlab](https://drive.google.com/file/d/0B_GL5U7rPj1xNzNwTHpHSzZkQXM/view?resourcekey=0-HneFEhC8NUIWDfhmfaOyBQ) | AAAI 2018 <br> A
|  Anomaly Detection | KPIs |      Donut | [Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications](https://doi.org/10.1145/3178876.3185996) |    [TF](https://github.com/NetManAIOps/donut) | WWW 2018 <br> A
|  Anomaly Detection | MAWI |      DSPOT | [Anomaly Detection in Streams with Extreme Value Theory](https://doi.org/10.1145/3097983.3098144) |    [Python](https://github.com/NetManAIOps/donut) | KDD 2017 <br> A
|  Anomaly Detection | Power <br> Space <br>  Engine <br> ECG |         EncDec-AD | [	LSTM-based encoder-decoder for multi-sensor anomaly detection](https://www.semanticscholar.org/paper/LSTM-based-Encoder-Decoder-for-Multi-sensor-Anomaly-Malhotra-Ramakrishnan/e9672150c4f39ab64876e798a94212a93d1770fe) |    [Pytorch](https://github.com/jaeeun49/Anomaly-Detection/blob/main/code_practices/LSTM-based%20Encoder-Decoder%20for%20Multi-sensor%20Anomaly%20Detection.ipynb) | ICML 2016 <br> A
|  Anomaly Detection |  MORE  |       MORE   | [https://github.com/ZIYU-DEEP/IJCAI-Paper-List-of-Anomaly-Detection](https://github.com/ZIYU-DEEP/IJCAI-Paper-List-of-Anomaly-Detection) |  MORE   | IJCAI  <br> A
|  Anomaly Detection |  MORE  |       MORE   | [DeepTimeSeriesModel](https://github.com/drzhang3/DeepTimeSeriesModel) |  MORE   | MORE  <br> A
|  Anomaly Detection |  MORE  |       MORE   | [GuansongPang](https://github.com/GuansongPang/SOTA-Deep-Anomaly-Detection) |  MORE   | MORE  <br> A




# [Demand Prediction](#content)
|  Task  |    Data |   Model  | Paper   |    Code    |   Publication    |
| :-: | :-: | :-: | :-: | :-:| - |
| Paper Nums: 23 | <img width=150/> | <img width=220/>  |   |   |   <img width=300/> | 
| Supply & <br> Demand | JONAS-NYC <br> JONAS-DC  <br>  COVID-CHI <br>  COVID-US |         EAST-Net | [Event-Aware Multimodal Mobility Nowcasting](https://aaai-2022.virtualchair.net/poster_aaai10914) | [Pytorch](https://github.com/underdoc-wang/EAST-Net) | AAAI 2022<br>A
| Health Demand | Family Van  |         framework        | [Using Public Data to Predict Demand for Mobile Health Clinics](https://aaai-2022.virtualchair.net/poster_emer91) | None | AAAI 2022<br>A
| Traffic Demand | NYC Bike <br> NYC Taxi  |         CCRNN        | [Coupled Layer-wise Graph Convolution for Transportation Demand Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/16591) | [Pytorch](https://github.com/Essaim/CGCDemandPrediction) | AAAI 2021<br>A
| Traffic Demand | BaiduBJ  <br> BaiduSH  |         Ada-MSTNet        | [Community-Aware Multi-Task Transportation Demand Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/16107) | None | AAAI 2021<br>A
| Job Demand | Online |         TDAN       | [Talent Demand Forecasting with Attentive Neural Sequential Model](https://dl.acm.org/doi/abs/10.1145/3447548.3467131) | None | KDD 2021<br>A
| Ambulance Demand | Tokyo |         EMS-Pred       | [Forecasting Ambulance Demand with Profiled Human Mobility via Heterogeneous Multi-Graph Neural Networks](https://ieeexplore.ieee.org/abstract/document/9458623) |  [Pytorch](https://github.com/underdoc-wang/EMS-Pred-ICDE-21)  | ICDE 2021<br>A
| Traffic  Demand | DiDiChengdu <br> NYCTaxi |         DAGNN       | [Dynamic Auto-structuring Graph Neural Network-A Joint Learning Framework for Origin-Destination Demand Prediction](https://ieeexplore.ieee.org/abstract/document/9657493) | None   | TKDE 2021<br>A
| Traffic Demand  |  TaxiNYC <br>  CitiBikeNYC |        MultiAttConvLSTM          | [Multi-level attention networks for multi-step citywide passenger demands prediction](https://ieeexplore.ieee.org/abstract/document/8873676/) | None  | TKDE 2021<br>A
| Market Demand  |  Juhuasuan  <br> Tiantiantemai     |        RMLDP    | [Relation-aware Meta-learning for E-commerce Market Segment Demand Prediction with Limited Records](https://doi.org/10.1145/3437963.3441750) |    None  | WSDM 2021<br> B
| Metro  Demand | MetroBJ2016 <br> MetroBJ2018 |         CAS       | [Short-term origin-destination demand prediction in urban rail transit systems: A channel-wise attentive split-convolutional neural network method](https://doi.org/10.1016/j.trc.2020.102928) |  None   |  Transportation Research Part C 21  <br> SCI 1 Top
| Metro  Demand | MetroBJ2016 <br> MetroBJ2018 |         ST-ED       | [Predicting origin-destination ride-sourcing demand with a spatio-temporal encoder-decoder residual multi-graph convolutional network](https://doi.org/10.1016/j.trc.2020.102858) |  None   |  Transportation Research Part C 21<br> SCI 1 Top
| Traffic Demand |  Seattlebike  |       FairST      | [Fairness-Aware Demand Prediction for New Mobility](https://ojs.aaai.org/index.php/AAAI/article/view/5458) | None | AAAI 2020<br>A
| Drug Demand  |  Wikipedia  |        None          | [Predicting Drug Demand with Wikipedia Views: Evidence from Darknet Markets](https://doi.org/10.1145/3366423.3380022) | None  | WWW 2020<br>A
| Traffic Demand  | DiDiBJ  <br>  DiDiSH  |  MPGCN   | [Predicting Origin-Destination Flow via Multi-Perspective Graph Convolutional Network](https://ieeexplore.ieee.org/abstract/document/9101359) | [Pytorch](https://github.com/underdoc-wang/MPGCN)  | ICDE 20 <br>A
| Traffic Demand  | NYC  <br>  DiDiCD  |  MPGCN   | [Stochastic Origin-Destination Matrix Forecasting Using Dual-Stage Graph Convolutional, Recurrent Neural Networks](https://ieeexplore.ieee.org/abstract/document/9101647/) | [TF](https://github.com/hujilin1229/od-pred)  | ICDE 20 <br>A
| Traffic Demand  | Bengaluru  <br>  NYC  |  GraphLSTM   | [Grids Versus Graphs: Partitioning Space for Improved Taxi Demand-Supply Forecasts](https://ieeexplore.ieee.org/abstract/document/9099450/) | [Pytorch](https://github.com/NDavisK/Grids-versus-Graphs)  | TITS 20 <br>B
| Traffic Demand  | NYCbike  <br>  NYCtaxi  |  CoST-Net   | [Co-Prediction of Multiple Transportation Demands Based on Deep Spatio-Temporal Neural Network](https://doi.org/10.1145/3292500.3330887) | None  | KDD 19 <br>A
| Traffic Demand  | UCAR  <br>  DiDiCD  |  GEML   | [Origin-Destination Matrix Prediction via Graph Convolution: a New Perspective of Passenger Demand Modeling](https://doi.org/10.1145/3292500.3330877) | [Keras](https://github.com/Zekun-Cai/GEML-Origin-Destination-Matrix-Prediction-via-Graph-Convolution)   | KDD 19 <br>A
| Traffic Demand  | NYCbike  <br>  Meso West  |  CE-LSTM   | [Learning Heterogeneous Spatial-Temporal Representation for Bike-Sharing Demand Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/3890) | None | AAAI 19 <br>A
| Traffic Demand  | Beijing  <br>  Shanghai  |  STMGCN  | [Spatiotemporal Multi-Graph Convolution Network for Ride-Hailing Demand Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/4247) | [Pytorch](https://github.com/underdoc-wang/ST-MGCN) | AAAI 19 <br>A
| Traffic Demand  | NYC-TOD   |  CSTN  | [Contextualized Spatial–Temporal Network for Taxi Origin-Destination Demand Prediction](https://ieeexplore.ieee.org/abstract/document/8720246/) | [Keras](https://github.com/liulingbo918/CSTN) | TITS 19 <br>B
| Traffic Demand  | NYCtaxi   |  MultiConvLSTM  | [Deep Multi-Scale Convolutional LSTM Network for Travel Demand and Origin-Destination Predictions](https://ieeexplore.ieee.org/abstract/document/8758916/) |  None   | TITS 19 <br>B
| Traffic Demand  | PeMS   |  t-SNE  | [Estimating multi-year  origin-destination demand using high-granular multi-source traffic data](https://doi.org/10.1016/j.trc.2018.09.002) |  None   | Transportation Research Part C 18  <br> SCI 1 Top




<!-- | Traffic Demand  |  Electricity <br> Traffic <br>  NYCTaxi  <br>  Uber   |    framework    | [Multi-Horizon Time Series Forecasting with Temporal Attention Learning](https://doi.org/10.1145/3292500.3330662) | [Pytorch](https://github.com/underdoc-wang/ST-MGCN)  | KDD 2019 <br> A 
 -->





# [Travel Time Estimation](#content)
|  Task  |    Data |   Model  | Paper   |    Code    |   Publication    |
| :-: | :-: | :-: | :-: | :-: | - |
| Paper Nums:17 | <img width=150/> | <img width=220/>  |   |   |   <img width=300/> | 
| TTE | Baidu:<br> Taiyuan <br> Huizhou <br> Hefei|         SSML        | [SSML: Self-Supervised Meta-Learner for En Route Travel Time Estimation at Baidu Maps](https://dl.acm.org/doi/10.1145/3447548.3467060) | [Paddle](https://github.com/PaddlePaddle/Research/tree/master/ST_DM/KDD2021-SSML)  | KDD 2021<br>A
| TTE | DiDi: <br> Shenyang     |     HetETA        | [HetETA: Heterogeneous Information Network Embedding for Estimating Time of Arrival](https://dl.acm.org/doi/10.1145/3394486.3403294) | [TF](https://github.com/didi/heteta)  | KDD 2020<br>A
| TTE | DiDi: <br> Beijing <br> Suzhou <br> Shenyang   |     CompactETA        | [CompactETA: A Fast Inference System for Travel Time Prediction](https://dl.acm.org/doi/10.1145/3394486.3403386) | None | KDD 2020<br>A
| TTE | GTFS     |     BusTr        | [BusTr: Predicting Bus Travel Times from Real-Time Traffic](https://doi.org/10.1145/3394486.3403376) |   None  | KDD 2020<br>A
| TTE | Taiyuan   <br>  Hefei <br> Huizhou <br> （Baidu）   |     BusTr        | [ConSTGAT: Contextual Spatial-Temporal Graph Attention Network for Travel Time Estimation at Baidu Maps](https://doi.org/10.1145/3394486.3403320) |   None  | KDD 2020<br>A
| TTE | NYC   <br>  IST <br> TKY   |     DeepJMT        | [Context-aware Deep Model for Joint Mobility and Time Prediction](https://doi.org/10.1145/3336191.3371837) |   None  | WSDM 2020<br>B
| TTE | Beijing <br> Shanghai    |     TTPNet        | [TTPNet: A Neural Network for Travel Time Prediction Based on Tensor Decomposition and Graph Embedding](https://ieeexplore.ieee.org/abstract/document/9261122) |   [Pytorch](https://github.com/YibinShen/TTPNet)  | TKDE 2020<br>A
| TTE | DiDiBJ   |     RNML-ETA         | [Road Network Metric Learning for Estimated Time of Arrival](https://ieeexplore.ieee.org/abstract/document/9412145) |   None  | ICPR 2020<br>C
| TTE | Cainiao    |     DeepETA       | [DeepETA: A Spatial-Temporal Sequential Neural Network Model for Estimating Time of Arrival in Package Delivery System](https://ojs.aaai.org/index.php/AAAI/article/view/3856) |   None  | AAAI 2019<br>A
| TTE | Beijing  <br>  Shanghai |     CTTE       | [Aggressive driving saves more time? Multi-task learning for customized travel time estimation](https://www.ijcai.org/Proceedings/2019/0234.pdf) |   None  | IJCAI 2019<br>A
| TTE | Shanghai  <br>  Porto |     DeepI2T       | [Travel time estimation without road networks: an urban morphological layout representation approach](https://www.ijcai.org/proceedings/2019/0245.pdf) |   None  | IJCAI 2019<br>A
| TTE | Porto <br> Chengdu    |     DeepIST        | [DeepIST: Deep Image-based Spatio-Temporal Network for Travel Time Estimation](https://dl.acm.org/doi/abs/10.1145/3357384.3357870) |   [TF](https://github.com/csiesheep/deepist)  | CIKM 2019<br>B
| TTE | Singapore |     AtHy-TNet       | [Path Travel Time Estimation using Attribute-related Hybrid Trajectories Network](https://doi.org/10.1145/3357384.3357927) |   None  | CIKM 2019<br>B
| TTE | BT-Traffic <br> PeMSD7 <br>  Q-traffic |    NASF     | [Learning to Effectively Estimate the Travel Time for Fastest Route Recommendation](https://doi.org/10.1145/3357384.3357907) |   None  | CIKM 2019<br>B
| TTE | Chengdu <br> Beijing    |     DeepTTE        | [When Will You Arrive? Estimating Travel Time Based on Deep Neural Networks](https://jelly007.github.io/deepTTE.pdf) |   [Pytorch](https://github.com/UrbComp/DeepTTE)  | AAAI 2018<br>A
| TTE | PORTO <br>SANFRANCISCO  |    NoisyOR     | [Predicting Vehicular Travel Times by Modeling Heterogeneous Influences Between Arterial Roads](https://ojs.aaai.org/index.php/AAAI/article/view/11858) |   None  | AAAI 2018<br>A
| TTE |  MORE  |     MORE       | [github](https://github.com/NickHan-cs/Spatio-Temporal-Data-Mining-Survey/blob/master/Estimated-Time-of-Arrival/Paper.md) | MORE | MORE<br>A




<!-- 

| TTE | GTFS <br> Beijing    |     BusTr        | [BusTr: Predicting Bus Travel Times from Real-Time Traffic](https://doi.org/10.1145/3394486.3403320) |   [Pytorch](https://github.com/UrbComp/DeepTTE)  | AAAI 2018<br>A -->



# [Traffic Location Prediction](#content)
|  Task  |    Data |   Model  | Paper   |    Code    |   Publication    |
| :-: | :-: | :-: | :-: |:-: | - |
| Paper Nums:20 | <img width=150/> | <img width=220/>  |   |   |   <img width=310/> | 
| Location | ETH+UCY <br> SDD <br> nuScenes <br> SportVU |              | [You Mostly Walk Alone: Analyzing Feature Attribution in Trajectory Prediction](https://openreview.net/forum?id=POxF-LEqnF) | None | ICLR 2022<br>None But Top
| Location | Gowalla <br> Foursquare <br> WiFi-Trace  |     GCDAN         | [Predicting Human Mobility via Graph Convolutional Dual-attentive Networks](https://dl.acm.org/doi/10.1145/3488560.3498400) |  [Pytorch](https://github.com/GCDAN/GCDAN) | WSDM 2022<br> B
| Location | MI <br> SIP   |     CMT-Net         | [CMT-Net: A Mutual Transition Aware Framework for Taxicab Pick-ups and Drop-offs Co-Prediction](https://dl.acm.org/doi/10.1145/3488560.3498394) | None | WSDM 2022 <br>B
| Location | Gowalla <br> FS-NYC  <br> FS-TKY  |     MobTCast       | [MobTCast: Leveraging Auxiliary Trajectory Forecasting for Human Mobility Prediction](https://proceedings.neurips.cc/paper/2021/hash/fecf2c550171d3195c879d115440ae45-Abstract.html) | [Author](https://drive.google.com/drive/folders/1xfiaz9cAxKYmNWgOH986JpMVSQbt3_qu?usp=sharing) | NIPS 2021<br>A
| Location | ETH <br> Hotel  <br> Univ <br> Zara1 <br> Zara2   |     CARPe       | [CARPe Posterum: A Convolutional Approach for Real-Time Pedestrian Path Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/16335) | [Pytorch](https://github.com/TeCSAR-UNCC/CARPe_Posterum) | AAAI 2021<br>A
| Location | ETH <br> Hotel  <br> Univ <br> Zara1 <br> Zara2   |     TPNMS       | [Temporal Pyramid Network for Pedestrian Trajectory Prediction with Multi-Supervision](https://ojs.aaai.org/index.php/AAAI/article/view/16299) | [Pytorch](https://github.com/Blessinglrq/TPNMS) | AAAI 2021<br>A
| Location | ETH <br> Hotel  <br> Univ <br> Zara1 <br> Zara2   |     DMRGCN       | [Disentangled Multi-Relational Graph Convolutional Network for Pedestrian Trajectory Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/16174) | [Pytorch](https://github.com/TeCSAR-UNCC/CARPe_Posterum) | AAAI 2021<br>A
| Location | Gowalla <br> Foursquare  |     BSDA       | [Location Predicts You: Location Prediction via Bi-direction Speculation and Dual-level Association](https://www.ijcai.org/proceedings/2021/74) | None | IJCAI 2021<br>A
| Location | ETH-UCY <br> Collisions <br>  NGsim  <br>Charges   <br> NBA  |     FQA       | [Multi-agent Trajectory Prediction with Fuzzy Query Attention](https://proceedings.neurips.cc/paper/2020/hash/fe87435d12ef7642af67d9bc82a8b3cd-Abstract.html) | [Pytorch](https://github.com/nitinkamra1992/FQA) | NIPS 2020<br>A
| Location | ETH-UCY <br> Collisions <br>  NGsim  <br>Charges   <br> NBA  |     ARNN    | [An Attentional Recurrent Neural Network for Personalized Next Location Recommendation](https://ojs.aaai.org/index.php/AAAI/article/view/5337) |   None  | AAAI 2020<br>A
| Location | ETH <br> Hotel  <br> Univ <br> Zara1 <br> Zara2  |     MDNLSTM    | [Multimodal Interaction-Aware Trajectory Prediction in Crowded Space](https://ojs.aaai.org/index.php/AAAI/article/view/6874) |   None  | AAAI 2020<br>A
| Location | Atlantic|     OMuLeT    | [OMuLeT: Online Multi-Lead Time Location Prediction for Hurricane Trajectory Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/5444) |   [Matlab](https://github.com/cqwangding/OMuLeT)   | AAAI 2020<br>A
| Location | Gowalla <br> Foursquare  |     Flashback  | [OMuLeT: Online Multi-Lead Time Location Prediction for Hurricane Trajectory Forecasting](https://www.ijcai.org/Proceedings/2020/302) |  [Pytorch](https://github.com/eXascaleInfolab/Flashback_code)  | IJCAI 2020<br>A
| Location | CrowdCJ <br> TrashBins  <br>B&B  <br> MYOPIC  |     MALMCS  | [Dynamic Public Resource Allocation Based on Human Mobility Prediction](https://doi.org/10.1145/3380986) |  [Python](https://github.com/sjruan/malmcs)  | UbiCom 2020<br>A
| Location | ETH <br> Hotel  <br> Univ <br> Zara1 <br> Zara2   |     Social-BiGAT  | [Social-BiGAT: Multimodal Trajectory Forecasting using Bicycle-GAN and Graph Attention Networks](https://proceedings.neurips.cc/paper/2019/hash/d09bf41544a3365a46c9077ebb5e35c3-Abstract.html) |  None  | NIPS 2019<br>A
| Location | Foursquare <br> Gowalla    |     VANext  | [Predicting Human Mobility via Variational Attention](https://doi.org/10.1145/3308558.3313610) |  None  | WWW 2019<br>A
| Location | Flickr <br> Foursquare  <br>  Geolife  |     CATHI  | [Context-aware Variational Trajectory Encoding and Human Mobility Inference](https://doi.org/10.1145/3308558.3313608) |  None  | WWW 2019<br>A
| Location | ETH <br> Hotel  <br> Univ <br> Zara1 <br> Zara2  |     STGAT  | [STGAT: Modeling Spatial-Temporal Interactions for Human Trajectory Prediction](https://openaccess.thecvf.com/content_ICCV_2019/html/Huang_STGAT_Modeling_Spatial-Temporal_Interactions_for_Human_Trajectory_Prediction_ICCV_2019_paper.html) |  [Pytorch](https://github.com/huang-xx/STGAT)  | ICCV 2019<br>A
| Location | BaiduBJ  |     HST-LSTM  | [HST-LSTM: A Hierarchical Spatial-Temporal Long-Short Term Memory Network for Location Prediction](https://www.ijcai.org/proceedings/2018/324) |  [Pytorch](https://github.com/Logan-Lin/ST-LSTM_PyTorch)  | IJCAI 2018<br>A
| Location | Foursquare   <br> MobileAPP <br> CellularSH |     DeepMove | [DeepMove: Predicting Human Mobility with Attentional Recurrent Networks](https://doi.org/10.1145/3178876.3186058) |  [Pytorch](https://github.com/vonfeng/DeepMove)  | WWW 2018<br>A
| Location |  MORE  |     MORE       | [github](https://github.com/xuehaouwa/Awesome-Trajectory-Prediction) | [Hao Xue](https://github.com/xuehaouwa/Awesome-Trajectory-Prediction) | MORE
| Location |  MORE  |     MORE       | [https://github.com/Pursueee/Trajectory-Paper-Collation](https://github.com/Pursueee/Trajectory-Paper-Collation) | [Pytorch](https://github.com/Pursueee/Trajectory-Paper-Collation) | MORE


<!-- 
| Location | Foursquare   <br> MobileAPP <br> CellularSH |     DeepMove | [DeepMove: Predicting Human Mobility with Attentional Recurrent Networks](https://doi.org/10.1145/3178876.3186058) |  [Pytorch](https://github.com/vonfeng/DeepMove)  | WWW 2018<br>A

| Location | ETH-UCY <br> Collisions <br>  NGsim  <br>Charges   <br> NBA  |     FQA       | [An Attentional Recurrent Neural Network for Personalized Next Location Recommendation](https://ojs.aaai.org/index.php/AAAI/article/view/5337) | [Pytorch](https://github.com/huang-xx/STGAT) | AAAI 2020<br>A -->




# [Traffic Event Prediction](#content)
|  Task  |    Data |   Model  | Paper   |    Code    |   Publication    |
| :-: | :-: | :-: | :-: | :-: | - |
| Paper Nums:17 | <img width=150/> | <img width=220/>  |   |   |   <img width=300/> | 
| Event | PeMS  |         AGWN        | [Early Forecast of Traffc Accident Impact Based on a Single-Snapshot Observation (Student Abstract)](https://aaai-2022.virtualchair.net/poster_sa103) | [Pytorch](https://github.com/gm3g11/AGWN) | AAAI 2022<br>A
|  Event  |  SLA-VAE <br> E-commerce  |       RETE    | [RETE: Retrieval-Enhanced Temporal Event Forecasting on Unified Query Product Evolutionary Graph](https://doi.org/10.1145/3485447.3511974) | None| WWW 2022 <br> A
| Event | NYC <br> Chicago |         GSNet        | [GSNet: Learning Spatial-Temporal Correlations from Geographical and Semantic Aspects for Traffic Accident Risk Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/16566) | [Pytorch](https://github.com/Echohhhhhh/GSNet) | AAAI 2021<br>A
| Event | NYCIncidents <br> CHIIncidents <br>  SFIncidents   |     STCGNN       | [Spatio-Temporal-Categorical Graph Neural Networks for Fine-Grained Multi-Incident Co-Prediction](https://doi.org/10.1145/3459637.3482482) | [Pytorch](https://github.com/underdoc-wang/STC-GNN) | CIKM 2021<br>B 
| Event | Thailand <br> Egypt <br>  India  <br>Russia   <br> Covid-19  |     CMF       | [Understanding Event Predictions via Contextualized Multilevel Feature Learning](https://doi.org/10.1145/3459637.3482309) | None  | CIKM 2021<br>B 
| Event  Prediction  |  DJIA30   <br> WebTraffic   <br> NetFlow  <br> ClockErr  <br>   AbServe  |        EvoNet    | [Time-Series Event Prediction with Evolutionary State Graph](https://doi.org/10.1145/3437963.3441827) |   None   | WSDM 2021 <br> B|
| Event | NYCIncidents <br> CHIIncidents <br>  SFIncidents   |     PreView       | [Dynamic Heterogeneous Graph Neural Network for Real-time Event Prediction](https://doi.org/10.1145/3394486.3403373) | None | KDD 2020<br>A
| Event | Beijing <br> Suzhou <br> Shenyang |         RiskOracle        | [RiskOracle: A Minute-Level Citywide Traffic Accident Forecasting Framework](https://ojs.aaai.org//index.php/AAAI/article/view/5480) | [TF](https://github.com/zzyy0929/AAAI2020-RiskOracle/) | AAAI 2020<br>A
| Event | NYCIncidents <br> CHIIncidents  |     STrans       | [Hierarchically Structured Transformer Networks for Fine-Grained Spatial Event Forecasting](https://doi.org/10.1145/3366423.3380296) | None  | WWW 2020<br>A
| Event | FewEvent  |     DMB-PN       | [Meta-Learning with Dynamic-Memory-Based Prototypical Network for Few-Shot Event Detection](https://dl.acm.org/doi/10.1145/3336191.3371796) | [dataset](https://github.com/231sm/Low_Resource_KBP)  | WSDM 2020<br>B
| Event | NYC <br> SIP  |         RiskSeq        | [Foresee Urban Sparse Traffic Accidents: A Spatiotemporal Multi-Granularity Perspective](https://ieeexplore.ieee.org/document/9242313) | None| TKDE 2020<br>A
| Event | MemeTracker  <br>  Weibo  |    LANTERN      | [Learning Latent Process from High-Dimensional Event Sequences via Efficient Sampling](https://proceedings.neurips.cc/paper/2019/hash/a29d1598024f9e87beab4b98411d48ce-Abstract.html) | [Pytorch](https://github.com/zhangzx-sjtu/LANTERN-NeurIPS-2019)  | NIPS 2019<br>A
| Event | Graph  <br>  Stack  <br> SmartHome <br> CarIndicators |    WGP-LN, <br>  FD-Dir     | [Uncertainty on Asynchronous Time Event Prediction](https://proceedings.neurips.cc/paper/2019/hash/78efce208a5242729d222e7e6e3e565e-Abstract.html) | [TF](https://github.com/sharpenb/Uncertainty-Event-Prediction)  | NIPS 2019<br>A
| Event | Thailand <br> Egypt <br>  India  <br>Russia |    DynamicGCN    | [Learning Dynamic Context Graphs for Predicting Social Events](https://doi.org/10.1145/3292500.3330919) |    [Pytorch](https://github.com/amy-deng/DynamicGCN)  | KDD 2019<br>A
| Event | NYCCollision  <br>  ChicagoCrime  <br> NYCTaxi |    DMPP    | [Deep Mixture Point Processes: Spatio-temporal Event Prediction with Rich Contextual Information](https://dl.acm.org/doi/10.1145/3292500.3330937) |    None  | KDD 2019<br>A
| Event | Civil <br> Air Quality  |    SIMDA    | [Incomplete Label Multi-Task Deep Learning for Spatio-Temporal Event Subtype Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/4245) |  None  | AAAI 2019<br>A
| Crime Prediction  | NYC Crime <br> NYC Anomaly   <br>  Chicago Crime  |        MiST      | [MiST: A Multiview and Multimodal Spatial-Temporal Learning Framework for Citywide Abnormal Event Forecasting](https://doi.org/10.1145/3308558.3313730) | None  | WWW 2019 <br> A
| Event | NYCAccident <br> NYCEvent  |    DFN    | [Deep Dynamic Fusion Network for Traffic Accident Forecasting](https://doi.org/10.1145/3357384.3357829) |  None  | CIKM 2019<br>B
| Event |   |         Hetero-ConvLSTM        | [Hetero-ConvLSTM: A Deep Learning Approach to Traffic Accident Prediction on Heterogeneous Spatio-Temporal Data](https://ieeexplore.ieee.org/document/9242313) | None| KDD 2018<br>A





<!-- 
| Event | NYCIncidents <br> CHIIncidents  |     PreView       | [Hierarchically Structured Transformer Networks for Fine-Grained Spatial Event Forecasting](https://doi.org/10.1145/3366423.3380296) | [Pytorch](https://github.com/amy-deng/DynamicGCN) | WWW 2020<br>A -->




# [Stock Prediction](#content)
|  Task  |    Data |   Model  | Paper   |    Code    |   Publication    |
| :-: | :-: | :-: | :-: | :-: | - |
| Paper Nums:29 | <img width=150/> | <img width=220/>  |   |   |   <img width=300/> | 
| Stock Movement <br> Prediction | Calls |         NumHTML      | [NumHTML: Numeric-Oriented Hierarchical Transformer Model for Multi-task Financial Forecasting](https://aaai-2022.virtualchair.net/poster_aaai4799) | [Future,Author](https://github.com/YangLinyi) | AAAI 2022<br>A   
| Stock Movement<br> Prediction | CSI800 |    TRA  |   [Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport](https://doi.org/10.1145/3447548.3467358) | [Pytorch](https://github.com/microsoft/qlib/tree/main/examples/benchmarks/TRA) | KDD 2021<br>A
| Stock Movement <br> Prediction | ACL18  <br> KDD17 <br> NDX100  <br> CSI300  <br> NI225<br> FTSE100 |     DTML | [Accurate Multivariate Stock Movement Prediction via Data-Axis Transformer with Multi-Level Contexts](https://doi.org/10.1145/3447548.3467297) | None| KDD 2021<br>A
| Stock  <br> Prediction | Self-defined |     AD-GAT | [Modeling the Momentum Spillover Effect for Stock Prediction via Attribute-Driven Graph Attention Networks](https://ojs.aaai.org/index.php/AAAI/article/view/16077) | [Pytorch](https://github.com/RuichengFIC/ADGAT) | AAAI 2021<br>A
| Stock Selection | NASDAQ  <br> NYSE <br> TSE|         STHAN-SR        | [Stock Selection via Spatiotemporal Hypergraph Attention Network: A Learning to Rank Approach](https://ojs.aaai.org/index.php/AAAI/article/view/16127) | None | AAAI 2021<br>A
| Stock Movement<br> Prediction | TPX500 |    CGM  |   [Long-term, Short-term and Sudden Event: Trading Volume Movement Prediction with Graph-based Multi-view Modeling](https://www.ijcai.org/proceedings/2021/0518.pdf) | [Pytorch](https://github.com/lancopku/CGM) | IJCAI 2021<br>A
| Stock Trend<br> Prediction |  CSI300 <br> SPX <br>  TOPIX-100   |    HATR  |   [Hierarchical Adaptive Temporal-Relational Modeling for Stock Trend Prediction](https://www.ijcai.org/proceedings/2021/0508.pdf) |  None  | IJCAI 2021<br>A
| Stock Movement<br> Prediction | NASDAQ <br> NYSE <br> TSE <br> China & HK |    HyperStockGAT  |   [Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport](https://doi.org/10.1145/3442381.3450095) | None| WWW 2021<br>A
| Stock Trend<br> Prediction |  CSI300 <br> CSI500    |    REST  |   [REST: Relational Event-driven Stock Trend Forecasting](https://doi.org/10.1145/3442381.3450032) | None | WWW 2021<br>A
| Stock Trend <br> Prediction | CSI300  <br> CSI800 <br> NASDAQ100|        CMLF       | [Stock Trend Prediction with Multi-granularity Data: A Contrastive Learning Approach with Adaptive Fusion](https://doi.org/10.1145/3459637.3482483) | [Pytorch](https://github.com/CMLF-git-dev/CMLF)  | CIKM 2021<br>B
| Stock Movement <br> Prediction | Self-defined |        MFN  | [Incorporating Expert-Based Investment Opinion Signals in Stock Prediction: A Deep Learning Framework](https://ojs.aaai.org/index.php/AAAI/article/view/5445) | None | AAAI 2020<br>A
| Stock Movement <br> Prediction | TPX500 <br> TPX100 |       LSTM-RGCN | [Modeling the Stock Relation with Graph Network for Overnight Stock Movement Prediction](https://www.ijcai.org/proceedings/2020/0626.pdf) | [Pytorch](https://github.com/liweitj47/overnight-stock-movement-prediction) | IJCAI 2020<br>A
| Stock Movement <br> Prediction | NASDAQ <br> ChinaAShare |      HMG-TF | [Hierarchical Multi-Scale Gaussian Transformer for Stock Movement Prediction](https://www.ijcai.org/proceedings/2020/0640.pdf) | None | IJCAI 2020<br>A
| Stock Trend<br> Prediction |  FI-2010 <br> CSI-2016    |    MTDNN  |   [Multi-scale Two-way Deep Neural Network for Stock Trend Prediction](https://www.ijcai.org/proceedings/2020/0628.pdf) | [Future](https://github.com/marscrazy/MTDNN) | IJCAI 2020<br>A
| Stock Price <br> Forecasting |  Self-defined |         Dandelion       | [Domain adaptive multi-modality neural attention network for financial forecasting](https://doi.org/10.1145/3366423.3380288) | [Sklearn](https://github.com/Leo02016/Dandelion)  | WWW 2020<br>A
| Stock Volatility <br> Forecasting |  Calls |         HTML       | [Hierarchical Transformer-based Multi-task Learning for Volatility Prediction](https://doi.org/10.1145/3366423.3380128) | [Pytorch](https://github.com/YangLinyi/HTML-Hierarchical-Transformer-based-Multi-task-Learning-for-Volatility-Prediction)  | WWW 2020<br>A
| Quantitative <br> Investments  | Self-defined |        KGEEF  | [Knowledge Graph-based Event Embedding Framework for Financial Quantitative Investments](https://doi.org/10.1145/3397271.3401427) | None | SIGIR 2020<br>A
| Stock Price <br> Prediction |  TAQ  |        GARCH-LSTM  | [Price Forecast with High-Frequency Finance Data: An Autoregressive Recurrent Neural Network Model with Technical Indicators](https://doi.org/10.1145/3340531.3412738) |  None   | CIKM 2020<br>B
| Stock Movement <br> Prediction | HATS |         STHGCN       | [Spatiotemporal hypergraph convolution network for stock movement forecasting](https://ieeexplore.ieee.org/abstract/document/9338303) | [Pytorch](https://github.com/midas-research/sthgcn-icdm) | ICDM 2020<br>B
| Stock Market <br> Prediction | Nikkei |         GNNs  | [Exploring Graph Neural Networks for Stock Market Predictions with Rolling Window Analysis](https://arxiv.org/abs/1909.10660) | None | NIPSw 2019<br>A
| Stock Trend <br> Prediction |  ChineseStock |         IMTR       | [Investment Behaviors Can Tell What Inside: Exploring Stock Intrinsic Properties for Stock Trend Prediction](https://doi.org/10.1145/3292500.3330663) | None| KDD 2019<br>A
| Stock Movement <br> Prediction | CSI200 <br> CSI300  <br> CSI500 |         RNN-MRFs  | [Multi-task Recurrent Neural Networks and Higher-order Markov Random Fields for Stock Price Movement Prediction: Multi-task RNN and Higer-order MRFs for Stock Price Classification](https://doi.org/10.1145/3292500.3330983) | None | KDD 2019<br>A
| Stock Movement <br> Prediction | Self-defined |         TTIO  | [Individualized Indicator for All: Stock-wise Technical Indicator Optimization with Stock Embedding](https://doi.org/10.1145/3292500.3330833) | None | KDD 2019<br>A
| Stock Movement <br> Prediction | ACL18  <br> KDD17 <br> NDX100  <br> CSI300  <br> NI225<br> FTSE100|         Adv-ALSTM  | [Enhancing stock movement prediction with adversarial training](https://www.ijcai.org/proceedings/2019/0810.pdf) | [TF](https://github.com/fulifeng/Adv-ALSTM) | IJCAI 2019<br>A
| Stock  Prediction | NASDAQ  <br> NYSE  |         RSR  | [Temporal Relational Ranking for Stock Prediction](https://doi.org/10.1145/3292500.3330833) | [TF](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking) | TOIS 2019<br>A
| Stock Movement <br> Prediction | Self-defined  |        StockNet   | [Stock Movement Prediction from Tweets and Historical Prices](https://aclanthology.org/P18-1183) | [TF](https://github.com/yumoxu/stocknet-code) | ACL 2018<br>A
| Stock Trend <br> Prediction | Self-defined  |        HAN  | [Listening to Chaotic Whispers: A Deep Learning Framework for News-oriented Stock Trend Prediction](https://doi.org/10.1145/3159652.3159690) | [TF](https://github.com/donghyeonk/han) | WSDM 2018<br>B
| Stock Price <br> Prediction | Self-defined |        SFM  | [Stock Price Prediction via Discovering Multi-Frequency Trading Patterns](https://doi.org/10.1145/3097983.3098117) | [Keras](https://github.com/z331565360/State-Frequency-Memory-stock-prediction) | KDD 2017<br>A
| Stock Movement <br> Prediction | NASDAQ  <br> NYSE  |         KGEB-CNN   | [Knowledge-Driven Event Embedding for Stock Prediction](https://aclanthology.org/C16-1201) | None | COLING 2016<br>B







# [Other Forecasting](#content)
|  Task  |    Data |   Model  | Paper   |    Code    |   Publication    |
| :-: | :-: | :-: | :-: | :-: | - |
| Paper Nums:9 | <img width=150/> | <img width=220/>  |   |   |   <img width=300/> |
| Crop Yield  Prediction | American Crop |         GNN-RNN        | [A GNN-RNN Approach for Harnessing Geospatial and Temporal Information: Application to Crop Yield Prediction](https://aaai-2022.virtualchair.net/poster_aisi6416) | None  | AAAI 2022<br> A|
| Epidemic Prediction | Globe <br> US-State  <br> US-County |         CausalGNN     | [CausalGNN: Causal-based Graph Neural Networks for Spatio-Temporal](https://aaai-2022.virtualchair.net/poster_aisi6475) | Future | AAAI 2022<br>A|
| Disease Prediction | Disease <br> Tumors   |         PopNet     | [PopNet: Real-Time Population-Level Disease Prediction with Data Latency](https://doi.org/10.1145/3485447.3512127) | [Pytorch](https://github.com/v1xerunt/PopNet) | WWW 2022<br>A|
| FakeNews Detection | Snop <br> PolitiFact  |         GET     | [Evidence-aware Fake News Detection with Graph Neural Networks](https://doi.org/10.1145/3485447.3512122) | [Keras](https://github.com/CRIPAC-DIG/GET) | WWW 2022<br>A|
| Health Prediction | Beidian <br> Epinions  |        CFChurn    | [A Counterfactual Modeling Framework for Churn Prediction](https://doi.org/10.1145/3488560.3498468) |     [Pytorch](https://github.com/tsinghua-fib-lab/CFChurn)   | WSDM 2022<br>B|
| Career Trajectory Prediction  | Company  <br> Position   |      TACTP          | [Variable Interval Time Sequence Modeling for Career Trajectory Prediction: Deep Collaborative Perspective](https://dl.acm.org/doi/10.1145/3442381.3449959) | None | WWW 2021<br>A|
| Health Prediction | NASH <br> AD  |        UNITE    | [UNITE: Uncertainty-based Health Risk Prediction Leveraging Multi-sourced Data](https://doi.org/10.1145/3442381.3450087) |     [Pytorch](https://github.com/Chacha-Chen/UNITE)   | WWW 2021<br>A|
| Crime Prediction | NYC <br> Chicago  |         ST-SHN     | [Spatial-Temporal Sequential Hypergraph Network for Crime Prediction with Dynamic Multiplex Relation Learning](https://www.ijcai.org/proceedings/2021/0225.pdf) |     [TF](https://github.com/akaxlh/ST-SHN)   | IJCAI 2021<br>A|
| Parking Prediction | Beijing <br> Shanghai |         SHARE       | [Semi-Supervised Hierarchical Recurrent Graph Neural Network for City-Wide Parking Availability Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/5471) |  [Pytorch](https://github.com/Vvrep/SHARE-parking_availability_prediction-Pytorch)  | AAAI 2020<br>A|
| Parking Prediction | Ningbo <br> Changsha |         PewLSTM       | [PewLSTM: Periodic LSTM with Weather-Aware Gating Mechanism for Parking Behavior Prediction](https://www.ijcai.org/proceedings/2020/610) |  [Pytorch](https://github.com/NingxuanFeng/PewLSTM)  | IJCAI 2020<br>A|




<!-- | Disease Prediction | Disease <br> Tumors   |         PopNet     | [PopNet: Real-Time Population-Level Disease Prediction with Data Latency](https://doi.org/10.1145/3485447.3512127) | [Pytorch](https://github.com/v1xerunt/PopNet) | WWW 2022<br>A| -->


# [Conferences](#content)


❗ 建议使用 [dblp](https://dblp.uni-trier.de/) 和 [Aminer](https://www.aminer.cn/conf)查询

❗ It is highly recommended to utilize the [dblp](https://dblp.uni-trier.de/) and [Aminer](https://www.aminer.cn/conf)(in Chinese) to search.


## Some Useful Websites


> CCF Conference Deadlines https://ccfddl.github.io/
> 
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

