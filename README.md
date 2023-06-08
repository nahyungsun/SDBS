이 깃허브는 https://github.com/ruotianluo/ImageCaptioning.pytorch의 코드를 기반으로 오류 문제를 해결하고 SDBS 알고리즘 및 성능 향상을 위한 알고리즘 제안 그리고 코드 최적화를 진행하였습니다.

사용 환경
Ubuntu 20.04.4 LTS
Python 3.8.13, 
PyTorch 1.12.0
CUDA 11.6, 
cuDNN 8.4.0,
NLTK 3.7.

pip install nltk==3.7  
pip install h5py==3.7.0  
pip install lmdbdict==0.2.2  
pip install scikit-image==0.19.2  
pip install matplotlib==3.5.1  
pip install gensim==4.2.0  
pip install pyemd==0.5.1  
pip install pandas  

실험 방법은 my_gen_eval_n_dbs_.sh 스크립트 파일을 다음과 같이 실행하면 됩니다. 뒤의 인자는 사용할 모델의 이름입니다.

bash my_gen_eval_n_dbs_.sh trans (or a2i2)

# 결과 예시
![image](https://github.com/nahyungsun/SDBS/assets/54011107/c9d383ae-0ab8-4ca4-a3ee-4953bd9c4414)

##### COCO Data Set Label
  ● a man with a red helmet on a small moped on a dirt road.  
  ● a man riding on the back of a motorcycle.  
  ● a dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains.  
  ● a man in a red shirt and a red hat is on a motorcycle on a hill side.  
  ● man riding a motor bike on a dirt road on the countryside.  

##### BeamSearch
● a man riding a motorcycle down a dirt road  
● a man riding a motorcycle on a dirt road  
● a man rides a motorcycle down a dirt road  
● a man is riding a motorcycle on a dirt road  
● a man riding a motorcycle down a dirt road in the mountains  

##### Diverse Beam Search
● a man riding a motorcycle down a dirt road  
● man riding a motorcycle on a dirt road  
● there are two people riding on a motorcycle  
● a person on a motor bike on a dirt road  
● two people are riding a motorcycle down a trail  

##### Semantic Diverse Beam Search
● a man riding a motorcycle down a dirt road  
● there is a man riding a motorcycle down a dirt road  
● man riding motorcycle on dirt road with mountain in background  
● the person is riding a motorcycle down the dirt road  
● an image of a man riding his motor bike  


