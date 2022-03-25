# Time-Series Work and Conference

# - <a href = "#Conferences">Jump to Conferences page</a>
# Deep Learning Models for Time-series Task

- <a href = "#Multivariable-Time-Series-Forecasting">Multivariable Time Series Forecasting</a>
- <a href = "#Multivariable-Probabilistic-Time-Series-Forecasting">Multivariable Probabilistic Time Series Forecasting</a>
- <a href = "#On--Demand/Original--Destination-Prediction">On-Demand/Original-Destination Prediction</a>
- <a href = "#Travel-Time-Estimation">Travel Time Estimation</a>
- <a href = "#Traffic-Accident-Prediction">Traffic Accident Prediction</a>
- <a href = "#Traffic-Location-Prediction">Traffic Location Prediction</a>
- <a href = "#Others">Others</a>


# [Multivariable Time Series Forecasting](#content)
|  Task  |    Data |   Model  | Paper   |    Code    |   Publication    |
| :-: | :-: | :-: | :-: | - | - |
| 还没数 | <img width=150/> | <img width=220/>  |   |   |   <img width=300/> | 
| Traffic Speed | NAVER-Seoul <br> METR-LA |         PM-MemNet         | [Learning to Remember Patterns: Pattern Matching Memory Networks for Traffic Forecasting](https://openreview.net/forum?id=wwDg3bbYBIq) | [Pytorch](https://github.com/HyunWookL/PM-MemNet) | ICLR 2022 / <br>None But Top 
| Multivariable | PeMSD3 <br> PeMSD4 <br> PeMSD8 <br> COVID-19,etc |         TAMP-S2GCNets         | [TAMP-S2GCNets: Coupling Time-Aware Multipersistence Knowledge Representation with Spatio-Supra Graph Convolutional Networks for Time-Series Forecasting](https://openreview.net/forum?id=wv6g8fWLX2q) | [Pytorch](https://www.dropbox.com/sh/n0ajd5l0tdeyb80/AABGn-ejfV1YtRwjf_L0AOsNa?dl=0) | ICLR 2022 / NoneButTop 
| Multivariable | ETT <br> Electricity <br> Weather |         CoST         | [CoST: Contrastive Learning of Disentangled Seasonal-Trend Representations for Time Series Forecasting](https://openreview.net/forum?id=PilZY3omXV2) | [Pytorch](https://github.com/salesforce/CoST) | ICLR 2022 /<br> None But Top 
| Multivariable | Electricity <br> traffic <br> M4 <br> CASIO <br> NP |         DEPTS         | [DEPTS: Deep Expansion Learning for Periodic Time Series Forecasting](https://openreview.net/forum?id=AJAR-JgNw__) | [Pytorch](https://github.com/weifantt/DEPTS) | ICLR 2022 /<br> None But Top 
| Multivariable | ETT <br> Electricity <br> Wind <br> App Flow |         Pyraformer     | [Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting](https://openreview.net/forum?id=0EXmFzUn5I) | [Pytorch](https://github.com/alipay/Pyraformer) | ICLR 2022 /<br> None But Top 
| Multivariable | ETT <br> ECL  <br> M4 <br> Air Quality <br> Nasdaq |         RevIN     | [Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift](https://openreview.net/forum?id=cGDAkQo1C0p) | [Pytorch](https://github.com/ts-kim/RevIN) | ICLR 2022 /<br> None But Top 
| Multivariable | Temperature <br> Cloud cover  <br> Humidity <br> Wind |         CLCRN     | [Conditional Local Convolution for Spatio-temporal Meteorological Forecasting](https://aaai-2022.virtualchair.net/poster_aaai1716) | [Pytorch](https://github.com/BIRD-TAO/CLCRN) | AAAI 2022 / A
| Traffic Flow | PeMSD3 <br> PeMSD4 <br> PeMSD7 <br> PeMSD8 <br> PeMSD7(M) <br> PeMSD7(L) |         STG-NCDE     | [Graph Neural Controlled Differential Equations for Traffic Forecasting](https://aaai-2022.virtualchair.net/poster_aaai1716) | [Pytorch](https://github.com/jeongwhanchoi/STG-NCDE) | AAAI 2022 / A
| Multivariable | GT-221 <br> WRS-393 <br> ZGC-564 |         STDEN     | [STDEN: Towards Physics-guided Neural Networks for Traffic Flow Prediction](https://aaai-2022.virtualchair.net/poster_aaai211) | [Pytorch](https://github.com/Echo-Ji/STDEN)   | AAAI 2022 / A
| Multivariable | Electricity <br> traffic <br> PeMSD7(M) <br> METR-LA  |         CATN     | [CATN: Cross Attentive Tree-Aware Network for Multivariate Time Series Forecasting](https://aaai-2022.virtualchair.net/poster_aaai7403) | None | AAAI 2022 / A
| Multivariable | ETT <br> Electricity  |         TS2Vec     | [TS2Vec: Towards Universal Representation of Time Series](https://aaai-2022.virtualchair.net/poster_aaai8809) | [Pytorch](https://github.com/yuezhihan/ts2vec) | AAAI 2022 / A
| Epidemic | Globe <br> US-State  <br> US-County |         CausalGNN     | [CausalGNN: Causal-based Graph Neural Networks for Spatio-Temporal](https://aaai-2022.virtualchair.net/poster_aisi6475) | Future | AAAI 2022 / A
| Traffic Flow | TaxiBJ <br> BikeNYC |         ST-GSP     | [ST-GSP: Spatial-Temporal Global Semantic Representation Learning for Urban Flow Prediction](https://dl.acm.org/doi/abs/10.1145/3488560.3498444) | [Pytorch](https://github.com/k51/STGSP) | WSDM 2022 / B
| Traffic Speed | METR-LA <br> PeMS-Bay <br> Simulated |         STNN     | [Space Meets Time: Local Spacetime Neural Network For Traffic Flow Forecasting](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9679008) | [Pytorch](https://github.com/songyangco/STNN) | ICDM 2021 / B
| Traffic Speed | DiDiChengdu <br> DiDiXiAn  |         T-wave     | [Trajectory WaveNet: A Trajectory-Based Model for Traffic Forecasting](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9679147) | [Pytorch](https://github.com/songyangco/STNN) | ICDM 2021 / B
| Multivariable | Sanyo <br> Hanergy <br> Solar <br> Electricity  <br> Exchange  |         SSDNet     | [SSDNet: State Space Decomposition Neural Network for Time Series Forecasting](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9679135) | [Pytorch](https://github.com/YangLIN1997/SSDNet-ICDM2021) | ICDM 2021 / B
| Traffic Volumn | Hangzhou City <br> Jinan City |         CTVI     | [Temporal Multi-view Graph Convolutional Networks for Citywide Traffic Volume Inference](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9679045) | [Pytorch](https://github.com/dsj96/CTVI-master) | ICDM 2021 / B
| Traffic Volumn | Uber Movements <br>  Grab-Posisi |         TEST-GCN     | [TEST-GCN: Topologically Enhanced Spatial-Temporal Graph Convolutional Networks for Traffic Forecasting](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9679077) | None | ICDM 2021 / B

# [Multivariable Probabilistic Time Series Forecasting](#content)
|  Task  |    Data |   Model  | Paper   |    Code    |   Publication    |
| :-: | :-: | :-: | :-: | - | - |
| 还没数 | <img width=150/> | <img width=220/>  |   |   |   <img width=300/> | 
| Traffic Speed | NAVER-Seoul <br> METR-LA |         PM-MemNet         | [Learning to Remember Patterns: Pattern Matching Memory Networks for Traffic Forecasting](https://openreview.net/forum?id=wwDg3bbYBIq) | [Pytorch](https://github.com/HyunWookL/PM-MemNet) | ICLR 2022 / <br>None But Top 


❗ 建议使用 [dblp](https://dblp.uni-trier.de/) 和 [Aminer](https://www.aminer.cn/conf)查询

❗ It is highly recommended to utilize the [dblp](https://dblp.uni-trier.de/) and [Aminer](https://www.aminer.cn/conf)(in Chinese) to search.


# [Conferences](#content)
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


> https://openreview.net/group?id=ICLR.cc/20xx/Conferenceeg:
>

| Conference | Source                                                     | Deadline | Notification |
| ---------- | ---------------------------------------------------------- | ---------- | ---------- |
|ICLR 2022|https://openreview.net/group?id=ICLR.cc/2022/Conference|Oct 06 '21|Jan 24 '22|
| ICLR 2021  | [https://openreview.net/group?id=ICLR.cc/2021/Conference](https://openreview.net/group?id=ICLR.cc/2021/Conference) |  |  |
| ICLR 2020     | [https://openreview.net/group?id=ICLR.cc/2020/Conference](https://openreview.net/group?id=ICLR.cc/2020/Conference)     |  |  |



## ICML

>  Format https://icml.cc/Conferences/20xx/Schedule

| Conference | Source                                                     | Deadline | Notification |
| ---------- | ---------------------------------------------------------- | ---------- | ---------- |
|ICML 2022|https://icml.cc/Conferences/2022/Schedule| Jan 27, 2022 |  |
|ICML 2021| [https://icml.cc/Conferences/2021/Schedule](https://icml.cc/Conferences/2021/Schedule)|  |  |
| ICML 2020 | [https://icml.cc/Conferences/2020/Schedule](https://icml.cc/Conferences/2020/Schedule) |  |  |
| ICML 2019  | [https://icml.cc/Conferences/2019/Schedule](https://icml.cc/Conferences/2019/Schedule)     |  |  |



## NeurIPS

[All Links](https://papers.nips.cc/)

## WSDM

[All Links](https://dl.acm.org/conference/wsdm)
