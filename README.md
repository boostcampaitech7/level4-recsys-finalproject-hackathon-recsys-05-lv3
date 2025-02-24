<p align="center">

  <h1> 🎞️ Cold-Start problem on your Recsys </h1>

  이 프로젝트는 개발된 Recsys에서 발생할 수 있는 **콜드스타트 문제를 직접 정의 및 해결함**을 목표로 합니다.

</p>

<br>

## 📊 Data
| Dataset                                                       | #Ratings    | #Users   | #Movies |
|--------------------------------------------------------------|-------------|----------|---------|
| [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/)| 32,000,204  | 200,948  | 87,585  |
| [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) | 1,000,209   | 6,040    | 3,900   |

**ratings.csv** : 유저 ID, 영화 ID, 평점, timestamp

**movies.csv** : 영화 ID, 영화 제목, 영화 장르

**links.csv** : 영화 ID, imdbId, tmdbId

<br>

## 🛠️ 기술 스택 및 협업
<div align="center">
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
├── code/
│   ├── checkpoints/
│   └── runs/
│
├── data/
│   ├── MovieLens1M/
│   │   ├── preprocessed/
│   │   └── raw/
│   ├── MovieLens32M/
│   │   ├── preprocessed/
│   │   └── raw/
│   └── .gitkeep
│
└── src/
    ├── data/
    │   ├── dataloader.py
    │   ├── embedding.py
    │   ├── preprocessing.py
    │   ├── sampling.cpp
    │   └── split_methods.py
    │
    ├── lightgcn_utils/
    │   ├── loss.py
    │   ├── metrics.py
    │   ├── trainer.py
    │   └── utils.py
    │
    ├── models/
    │   ├── __init__.py
    │   ├── CLCRec.py
    │   └── lightgcn.py
    │
    ├── scrap/
    │   └── scrapper.py
    │
    ├── __init__.py
    └── wandblogger.py
 

```

<br>

## ❄️ 콜드스타트 문제 정의

<img src="https://github.com/user-attachments/assets/008a1851-59ac-4846-b747-3ff06a8b060a" width="500" />




<br>

## 🏃 How to run
### Config
config/baseline.yaml에서 파라미터를 설정하세요.

baseline에서 추가 실험한 내용은 다음과 같습니다.

- **model_args**: `LightGCN`, `CLCRec` 
- **loss**: `BPRLoss`, `BPRLossWithReg`, `BPRLoss_with_coldweight`, `BPRLoss_with_alignment_similarity`
- **train.neg_sampling**: `random`, `popular`, `cold`
- **model_args.LightGCN.use_meta_embedding**:  `True`, `False`
- **use_ssl**: `True`, `False`
- **hyperparameter**: 임베딩 차원(`latent_dim_rec`), 학습률(`lr`), 레이어 수(`n_layers`), 정규화(`weight_decay`), 시드(`seed`) 등

<br>

<img src="https://github.com/user-attachments/assets/c5818e75-6cda-490b-81ef-bf7358abb2d4" width="500" />

<br>

### 전처리 & 학습 & 예측
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

<br>

## 👨🏼‍💻 Members
<div align="center">
<table>
  <tr>
    <td align="center"><a href="https://github.com/annakong23"><img src="https://avatars.githubusercontent.com/u/102771961?v=4" width="100px;" alt=""/><br /><sub><b>공지원</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/kimjueun028"><img src="https://avatars.githubusercontent.com/u/92249116?v=4" width="100px;" alt=""/><br /><sub><b>김주은</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/JihoonRyu00"><img src="https://avatars.githubusercontent.com/JihoonRyu00" width="100px;" alt=""/><br /><sub><b>류지훈</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/SayOny"><img src="https://avatars.githubusercontent.com/SayOny" width="100px;" alt=""/><br /><sub><b>박세연</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/JaeHyun11"><img src="https://avatars.githubusercontent.com/JaeHyun11" width="100px;" alt=""/><br /><sub><b>박재현</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/gagoory7"><img src="https://avatars.githubusercontent.com/u/163074222?v=4" width="100px;" alt=""/><br /><sub><b>백상민</b></sub><br />
    </td>
  </tr>
</table>
</div>

<br>

## 📝 Wrap Up Report & Presentation Deck

[Wrap Up Report](https://github.com/user-attachments/files/18938745/TVING2_RecSys_.05.pdf)

<br>

[Presentation Deck](https://github.com/user-attachments/files/18938658/RecSys_5._.2_Cold-Start.problem.on.your.Recsys.pdf)
