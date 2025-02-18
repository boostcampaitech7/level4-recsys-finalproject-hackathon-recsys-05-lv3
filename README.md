<p align="center">

  <h1> 🎞️ Cold-Start problem on your Recsys </h1>

  > 이 프로젝트는 개발된 Recsys에서 발생할 수 있는 콜드스타트 문제를 직접 정의 및 해결하고자 함을 목표로 합니다.


</p>

<br>


## 👨🏼‍💻 Members
공지원|김주은|류지훈|박세연|박재현|백상민|
:-:|:-:|:-:|:-:|:-:|:-:
<img src='https://avatars.githubusercontent.com/annakong23' height=60 width=60></img>|<img src='https://avatars.githubusercontent.com/kimjueun028' height=60 width=60></img>|<img src='https://avatars.githubusercontent.com/JihoonRyu00' height=60 width=60></img>|<img src='https://avatars.githubusercontent.com/SayOny' height=60 width=60></img>|<img src='https://avatars.githubusercontent.com/JaeHyun11' height=60 width=60></img>|<img src='https://avatars.githubusercontent.com/gagoory7' height=60 width=60></img>|
<a href="https://github.com/annakong23" target="_blank"><img src="https://img.shields.io/badge/GitHub-black.svg?&style=round&logo=github"/></a>|<a href="https://github.com/kimjueun028" target="_blank"><img src="https://img.shields.io/badge/GitHub-black.svg?&style=round&logo=github"/></a>|<a href="https://github.com/JihoonRyu00" target="_blank"><img src="https://img.shields.io/badge/GitHub-black.svg?&style=round&logo=github"/></a>|<a href="https://github.com/SayOny" target="_blank"><img src="https://img.shields.io/badge/GitHub-black.svg?&style=round&logo=github"/></a>|<a href="https://github.com/JaeHyun11" target="_blank"><img src="https://img.shields.io/badge/GitHub-black.svg?&style=round&logo=github"/>|<a href="https://github.com/gagoory7" target="_blank"><img src="https://img.shields.io/badge/GitHub-black.svg?&style=round&logo=github"/></a>

<br>

## 🛠️ 기술 스택 및 협업
  <img src="https://img.shields.io/badge/Python-3776AB?style=square&logo=Python&logoColor=white"/>&nbsp;
  <img src="https://img.shields.io/badge/Pandas-150458?style=square&logo=Pandas&logoColor=white"/>&nbsp;
  <img src="https://img.shields.io/badge/scikitlearn-F7931E?style=square&logo=scikitlearn&logoColor=white"/>&nbsp;
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white"/>&nbsp;

  <img src="https://img.shields.io/badge/Notion-000000?style=square&logo=Notion&logoColor=white"/>&nbsp;
  <img src="https://img.shields.io/badge/Slack-4A154B?style=flat-square&logo=Slack&logoColor=white"/>&nbsp;


<br>

## 📁 Directory
```bash
project
├── README.md
├── main.py
├── inference.py
├── requirements.txt
│
├── config/
│   └── baseline.yaml
│
├── data/
│   ├── raw/
│   └── preprocessed/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── utils/
│   └── __init__.py/
│
└── code/
    ├── checkpoints/
    └── runs/

```
<br>

# 🏃 How to run
## Config

baseline.yaml에서 파라미터를 설정하세요



## 전처리 & 학습 & 예측
### Training

전처리 & 학습 & 예측을 동시에 하려면 다음 명령어를 사용하세요:

```bash
python main.py -c config/baseline.yaml
```

자세한 파싱 정보는 main.py를 참고하세요.

### Inference

추론 및 결과 저장을 동시에 하려면 다음 명령어를 사용하세요:

```bash
python inference.py -c config/baseline.yaml
```

자세한 파싱 정보는 inference.py를 참고하세요.

