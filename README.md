# A-NIDS: Adaptive Network Intrusion Detection System Based on Clustering and Stacked CTGAN
Existing research often assumes that training and testing data are static and identically distributed, 
whereas in reality, data drift is inevitable. Moreover, to enhance model versatility and detection 
performance, models have become increasingly complex, posing challenges to real time deployment. 
To address these challenges, we propose an adaptive network intrusion detection system named A-NIDS, 
consisting of a main task and two bypass tasks. The main task is to develop a fully connected 
and shallow network with strong detection performance and real-time capability. The first bypass 
task is a clustering model that helps the main task detect data drift in an unsupervised manner. 
The second bypass task is a generation model to generate old data to address catastrophic forgetting 
in new model iterations and the storage cost issue caused by accumulating old data. 
![图片描述](Framework.jpg)

# Environment Setup
`pip install -r requirement.txt`

# Citation
@article{zha2025nids,  
&nbsp;&nbsp;&nbsp;&nbsp;title={A-NIDS: Adaptive Network Intrusion Detection System Based on Clustering and Stacked CTGAN},  
&nbsp;&nbsp;&nbsp;&nbsp;author={Zha, Chao and Wang, Zhiyu and Fan, Yifei and Bai, Bing and Zhang, Yinjie and Shi, Sainan and Zhang, Ruyun},  
&nbsp;&nbsp;&nbsp;&nbsp;journal={IEEE Transactions on Information Forensics and Security},  
&nbsp;&nbsp;&nbsp;&nbsp;year={2025},  
&nbsp;&nbsp;&nbsp;&nbsp;publisher={IEEE}  
}