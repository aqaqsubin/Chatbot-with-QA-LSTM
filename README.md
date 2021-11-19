
# **AI Chatbot based on QA-LSTM🤖**

  

### **2016 ICLR LSTM-based Deep Learning Models for Non-factoid Answer Selection**

QA-LSTM과 QA-LSTM with Attention 모델 구현 (`avg_pooling`/`max_pooling`)

 
<br>

-----
  

### 👉 **Word Embedding**

&nbsp;&nbsp;&nbsp;&nbsp;**Pretrained Embedding**: KoBERT `monologg/kobert`을 사용한 BERT 임베딩

```
python main.py <생략> --embed bert
```


&nbsp;&nbsp;&nbsp;&nbsp;**Embedding Layer**: nn.Embedding을 이용한 임베딩

```
python main.py <생략> --embed nn
```
<br>
<br>

-----
### **모델 프레임워크**

<br>
<div  align=left>
<img  src="./img/QA-LSTM.png"  width=700/><br>
QA-LSTM
</div>
<br>


<br>
<div  align=left>
<img  src="./img/QA-LSTM_attn.png"  width=700/><br>
QA-LSTM with attention
</div>
<br>

  
  

👉 **Attention mechanism**  
&nbsp;&nbsp;Bahdanau Attention mechanism 사용

<br>

-----

### **모델 훈련**
**Multi GPU**
```
python main.py --cuda --gpuid [list of gpuid] --data_dir [data dir path] --method [pooling type] --embed [embedding method] --max_epochs 10 --train --model_name [model_name] --accelerator ddp --embd_size [embedding size]] --hidden_size [hidden size] --batch_size [batch_size]
```

**Single GPU**
```
python main.py --cuda --gpuid 0 --data_dir [data directory path] --method [pooling type] --embed [embedding method] --max_epochs 10 --train --model_name [model_name] --embd_size [embedding size]] --hidden_size [hidden size] --batch_size [batch_size]
```

- data_dir: `train.csv`, `val.csv`가 있는 디렉토리 경로
- method: `avg_pooling` or `max_pooling`
- gpuid: GPU ID 리스트
	- ex) `--gpuid 0 1 2`
- embed: 
	- `bert` : 사전학습된 KoBERT 임베딩 (`monologg/kobert` 사용)
	- `nn` : torch.nn.Embedding Layer  
<br>
- embd_size: 임베딩 크기 (BERT 임베딩을 사용하는 경우 768)  
- hidden_size: LSTM Layer의 hidden size  
- batch_size: batch size

<br>

### **모델 검증**
📣 검증 시 `LightningQALSTM`의 `embd_size`, `hiddend_size` 확인 필요  (훈련과 동일하게 설정)  

```
python main.py --cuda --model_pt [model path] --gpuid [gpu id] --data_dir [data directory path] --embd_size [embedding size]  --hidden_size [hidden size]
```

- data_dir: `reaction_emb.pickle`이 있는 디렉토리 경로 (없는 경우 새로 생성)
- method: `avg_pooling` or `max_pooling` (학습된 모델과 동일한 pooling method 선택)
- model_pt: 모델 체크포인트 경로 
	- ex) `--model_pt model_ckpt/qa_lstm-epoch\=04-train_loss\=0.05.ckpt`
- gpuid: 하나의 GPU ID 
- embed: 
	- `bert` : 사전학습된 KoBERT 임베딩 (`monologg/kobert` 사용)
	- `nn` : torch.nn.Embedding Layer  
<br>
- embd_size: 임베딩 크기 (BERT 임베딩을 사용하는 경우 768)   
- hidden_size: LSTM Layer의 hidden size  
<br>

#### **모델 검증 예시**


<br>

> **Model Info**   
>&nbsp;&nbsp;&nbsp;&nbsp;Model : QA-LSTM  
&nbsp;&nbsp;&nbsp;&nbsp;pooling : max_pooling  
&nbsp;&nbsp;&nbsp;&nbsp;embedding method: nn.Embedding Layer   
&nbsp;&nbsp;&nbsp;&nbsp;embedding size: 256    
&nbsp;&nbsp;&nbsp;&nbsp;hiddend size: 128     


의미있는 결과를 보이진 않는 것으로 판단... 

```
Query : 안녕? 
Candidate: ['하이! 헬로! 안녕하세요', '얼른 사과하세요', '오예오예 야호야호', '울지마 바보야~', '오예', '충분합니다', '흑흑흑', '흑흑', '아자아자 파이팅!!', '여보세요? 모시모시? 헬로우?', '체리 먹고 정신 체리세요', '어랏', '센스쟁이시네요 그런 센스는 저도 가르쳐 주세요', '오올? 센스쟁이', '피짜?', '으악', '으이구', '그냥 뭐 이것저것?', '아 맞다!', '슬프다. 흑흑 오늘도 r는 눈물을 흘린r...'] 
 
Query : 저녁 먹었어? 
Candidate: ['변태', '빙고', '으엑웩', '절레절레', '거절합니다', '다이어트?', '떡볶이??', '삼겹살?', '세상에...', '헐...', '바이바이', '안 삐졌눈뒈...', '꺄아', '냠냠냠 뇸뇸뇸', '아이고', '우웩 갑자기 매스꺼운 느낌?', '크크', '역시는 역시다', '헐', '냠냠냠냠'] 

Query : 오늘 뭐했어? 
Candidate: ['할 수 있슴다!', '오잉? 기분 탓인가?', '절레절레', '변태다. 변태가 나타났다', '오올? 센스쟁이', '먹고 싶긴 한데, 이번엔 패스!', '조심하겠슴다!', '아... 절레절레', '바이바이', '그냥 뭐 이것저것?', '누구인가? 누가 그런 소리를 내었나', '다이어트?', '짱짱!', '피짜?', '빙고', '으악', '아 맞다!', '헐...', '반사 반사도 반사입니다.', '변태'] 

Query : 나 우울해 
Candidate: ['으엑웩', '꺄아', '냠냠냠냠', '냠냠냠 뇸뇸뇸', '쳇', '쳇쳇', '크크', '힝', '변태', '어라', '메리 크리스마스', '짱짱!', '안 삐졌눈뒈...', '사과먹고 사과하세요', '절레절레', '떡볶이??', '세상에...', '어머', '헐...', '바이바이']  

Query : 하루종일 서있었어 
Candidate: ['냠냠냠 뇸뇸뇸', '으엑웩', '안 삐졌눈뒈...', '꺄아', '어라', '짱짱!', '사과먹고 사과하세요', '냠냠냠냠', '쳇', '쳇쳇', '힝', '떡볶이??', '부들부들', '절레절레', '변태', '룰루룰루', '우웩 갑자기 매스꺼운 느낌?', '크크', '메리 크리스마스', '바이바이'] 

Query : 어머머 
Candidate: ['어머어머', '변태', '누구인가? 누가 그런 소리를 내었나', '아멘', '어머', '웰컴 환영합니다', '우웩 갑자기 매스꺼운 느낌?', '하하하', '헐...', '안 삐졌눈뒈...', '조심하겠슴다!', '슬프다. 흑흑 오늘도 r는 눈물을 흘린r...', '오올? 센스쟁이', '변태다. 변태가 나타났다', '떡볶이??', '으엑웩', '아... 절레절레', '크크', '거절합니다', '먹고 싶긴 한데, 이번엔 패스!'] 

Query : 야 너 잘하는게 뭔데 
Candidate: ['으엑웩', '절레절레', '안 삐졌눈뒈...', '꺄아', '냠냠냠 뇸뇸뇸', '짱짱!', '누구인가? 누가 그런 소리를 내었나', '냠냠냠냠', '어라', '쳇', '쳇쳇', '크크', '힝', '바이바이', '변태', '사과먹고 사과하세요', '빙고', '아... 절레절레', '메리 크리스마스', '먹고 싶긴 한데, 이번엔 패스!'] 
```

<br>

> **Model Info**   
>&nbsp;&nbsp;&nbsp;&nbsp;Model : QA-LSTM with attention   
&nbsp;&nbsp;&nbsp;&nbsp;pooling : max_pooling    
&nbsp;&nbsp;&nbsp;&nbsp;embedding method: nn.Embedding Layer   
&nbsp;&nbsp;&nbsp;&nbsp;embedding size: 256    
&nbsp;&nbsp;&nbsp;&nbsp;hiddend size: 128     


응답DB에서 query와 가장 비슷한 reply가 채택되는 경향이 있음

```
Query : 안녕? 
Candidate: ['안녕하세요?', '다이어트?', '안녕안녕입니다', '안 잤어요?', '안 좋아해요', '출발했어요?', '피짜?', '안 보여요', '아직 안 자요?', '준비 다 했어요?', '안해요? 왜 안해요?', '사랑스러워요', '많이 안 좋아요?', '안 늦었어요?', '벌써 다 먹었어요?', 'TV봐요?', '어랏', '불러주면 안돼요?', '혼났어요?', '기억 나요?'] 

Query : 저녁 먹었어? 
Candidate: ['저녁 뭐 먹었어요?', '점심 뭐 먹었어요?', '늦었어요?', '푹 쉬었어요?', '벌써 다 먹었어요?', '뭐 좀 먹었어요?', '저녁 먹어야죠. 냠냠뇸뇸', '늦지 않게 출발했어요?', '뭐 먹었어요? 한식? 일식? 간식?', '잘 쉬었어요?', '밥은 먹었어요?', '안 늦었어요?', '출발했어요?', '맛있겠죠?', '늦진 않았어요?', '햄버거?', '다이어트?', '혼자 먹어요?', '준비 다 했어요?', '재밌어요?'] 

Query : 오늘 뭐했어? 
Candidate: ['피짜?', '맛있어요?', '얼마나 마셨어요?', '고기?', '오늘은 뭐했어요?', '지금 일어났어요?', '왜 웃어요?', '잘했죠?', '뭐하는데요?', '맛있겠죠?', '지금도요?', '뭐래요?', '무슨 약이요?', '언제 자요?', '혼자 먹어요?', '몇시에 일어났어요?', '저 때문에요?', '심심하죠?', '누워있어요?', '혼났어요?'] 

Query : 난 우울해
Candidate: ['우와 인정합니다', '이해해요', '이해해요', '그래도요...', '헐', '헐', '헐...', '퓽 하고 없어졌어요. 헤헷', '서운해요 흥', '충분해요', '그것이 알고 싶다....', '이상해요', '피곤해서 그래요', '아주 난리났네요.', '꺄아', '으악', '집중해요', '전화해봐요', '세상 억울해요', '피곤해서 어떡해요.'] 

Query : 하루종일 서있었어 
Candidate: ['푹 쉬었어요?', '하루 종일요?', '누워있어요?', '늦었어요?', '전 뭐든 상관없어요.', '서운해요 흥', '안 늦었어요?', '어제 몇시에 잤어요?', '뭐라도 먹어요.', '힝. 저 울 겁니다', '헉 많이 마셨네요.', '전 뭐든 다 좋아요', '깼어요? 깼으면 이제 벌떡 일어나세요!', '뭐 좀 먹었어요?', '웰컴 환영합니다', '오호 딱이네요', '오올? 센스쟁이', '그럼 되죠. 생각보다 간단하네요.', '우에웩 갑자기 속이 별로네요.', '늦네요. 무슨 일 있는 건 아니겠죠?'] 

Query : 어머머 
Candidate: ['어머', '어머어머', '어랏', '어라', '크크', '충분합니다', '바이바이', '정답입니다', '어색해요', '변태', '다이어트?', '어쩌겠어요', '심쿵', '열심히 해야죠', '얼마든지요', '덜덜', '덜덜', '수업 열심히 들어요~', '심심하죠?', '진짜네요'] 

Query : 야 너 잘하는게 뭔데 
Candidate: ['오예오예 야호야호', '안 삐졌눈뒈...', '오와 정말 잘나왔네요!', '안녕 잘가요. 빠이빠~이 빠이빠이야', '잘 다녀와요~', '원할 때 언제든지요', '한숨 푹 자요~', '다행이네요!', '오호 딱이네요', '제가 잘못했네요', '지인짜 보고싶어요', '오올~ 좋은 자세예요. 베리굿', '오예오예 저 지금 약간 신이 났어요.', '우와 시간 진짜 빠르네요', '얼른 사과하세요', '잘 쉬었어요?', '오~ 똑똑하네요', '너무 길어요. 전 짧은게 좋던데...', '오올? 센스쟁이', '신경써줘서 고마워요']
```


