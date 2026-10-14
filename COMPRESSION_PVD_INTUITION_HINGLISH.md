# Compression aur PVD: code kya karta hai, intuition kya hai

Code review date: 7 October 2026. Ye current implementation ki explanation hai. Neeche diye combination ideas abhi hypotheses hain; unka result measure karna baaki hai.

## 1. Teen signals ka simple difference

| Signal | Code ka main sawaal | Direction ka role |
|---|---|---|
| Absorption, hamara current expected-profit v2 analysis | Buy/sell delta extreme hai, par net price response chhota hai? | Buy absorption se DOWN/short hypothesis; sell absorption se UP/long hypothesis |
| Compression | Range aur recent price movement apne history se unusually chhote hain? | UP/DOWN choose nahi karta; baad mein movement badhne ki hypothesis |
| PVD exhaustion | Is volume ke liye model jitna price response expect karta hai, actual response usse unusually kam hai? | Recent net move UP/DOWN se signal ka naam banta hai |
| PVD expansion | Is volume ke liye model jitna response expect karta hai, actual response usse unusually zyada hai? | Recent net move UP/DOWN se signal ka naam banta hai |

**Absorption aur compression related ho sakte hain, par dono ki definition alag hai.** Absorption net displacement aur signed buy-minus-sell delta dekhta hai. Compression high-low range, recent movement aur total volume dekhta hai.

Example: price bahut upar-neeche ghoomkar starting price par close ho gaya. Net response chhota hai, par range badi hai. Absorption qualify kar sakta hai; compression zaroori nahi.

## 2. Compression ka algo

Main function: `signals/compression/calculate.py -> calculate_compression()`.

### Step A: 4 cheezein measure hoti hain

| Component | Actual calculation | Simple intuition | Weight |
|---|---|---|---|
| Range | Current high - low | Candle ke andar price kitna phaila? Chhota range compression ko support karta hai | 30% |
| ATR | Recent true range ka average | Recent candles overall shaant hain? Gaps bhi true range mein aate hain | 25% |
| RV | Recent close changes ke squares ka average, phir square root | Recent close-to-close movement chhota hai? | 25% |
| Volume-range efficiency | `log(1 + volume) / log(1 + range)` | Range ke comparison mein activity zyada hai? | 20% |

- Current config: range baseline 20 bars; ATR 5 bars; RV 5 bars.
- RV yahan price changes ka RMS jaisa measure hai. Mean change subtract nahi hota; ye standard deviation nahi hai.
- Total volume use hota hai. Buy/sell delta aur direction use nahi hote.
- High volume compulsory nahi: range + ATR + RV se confirmation aa sakti hai, efficiency weak hone par bhi.

### Step B: apni recent history se compare karo

- Range, ATR aur RV par `log(1 + value)` lagta hai.
- Har feature ka current value previous baseline ke median aur MAD se compare hota hai.
- MAD ka matlab: median se usual absolute distance.
- Formula: `(current - past median) / (1.4826 × past MAD + epsilon)`.
- Baseline mein current row nahi hoti: rolling history `shift(1)` hoti hai.
- Range/ATR/RV ka z negative ho toh usual se shaant. Efficiency ka z positive ho toh usual se zyada activity per range.

### Step C: z ko component score banao

- Range/ATR/RV: `clip(-z / 2.5, 0, 1)`.
- Efficiency: `clip(z / 2.5, 0, 1)`.
- Example: range z = -1.25 -> range component 0.50.
- 2.5 ya usse bada relevant deviation -> component 1 par cap ho jata hai.
- Phir 4 components ka weighted average banta hai.

### Step D: confirmation gate aur final state

- Range component ko minimum level pass karna zaroori hai.
- ATR/RV/efficiency mein se kam se kam 2 ko bhi wahi minimum pass karna hai.
- Minimum level = `compression_threshold × (0.40 / 0.60)`.
- Current config threshold 0.30 hai, toh component minimum 0.20 hai.
- Gate fail -> score 0. Gate pass -> weighted score retain hota hai.
- `compression_state = 1` tab hai jab final score configured threshold se bada ya equal ho.
- Gate-pass positive score aur final compressed state ek hi cheez nahi hain: positive score threshold se neeche bhi ho sakta hai.
- `compression_persistence` recent compressed states ka average hai. Ye final score mein extra weight add nahi karta.

Intuition: ek chhoti candle ko turant compression nahi bolna; range ke saath doosri measurements bhi shaant honi chahiye.

## 3. Compression ke multiday tracks

Function: `signals/compression/calculate_multiday.py -> build_scored_candles()`.

- k-day candle: first open, maximum high, minimum low, last close, summed volume.
- Har trading bar par ek naya trailing candle ban sakta hai.
- **`track="candle"`:** previous overlapping k-day candles baseline mein aati hain. Recent history milti hai, par adjacent candles ka data bahut shared hota hai.
- **`track="custom"`:** har k-th candle ki separate phase banti hai. Us phase ki candles non-overlapping hoti hain. Baseline ke 20 candles ke liye roughly 20 × k raw bars, plus indicator warmup, chahiye.
- Example 5D + 20 baseline: approximately 100 trading bars sirf baseline scale par; warmup aur bhi lag sakta hai.
- Custom track mein ATR/RV/persistence windows bhi phase bars par chalte hain. Unka calendar span 1D se bada ho jata hai.
- Candle track mein baseline current aggregate row exclude karta hai, par previous aggregate rows current candle ke kuch constituent days share kar sakti hain. Ye future data nahi hai, lekin independent comparison bhi nahi hai.

Intuition choice: recent overlapping distribution vs longer non-overlapping distribution. Hamare absorption baseline ki tarah current constituent window ko poori tarah exclude karne ka rule yahan automatically same nahi hai.

## 4. Compression success label kya bolta hai?

Do labeling files hain; dono ko mix mat samajhna.

### ATR label: `label_atr.py`

- Signal close se next configured bars ke closes ka maximum absolute move lo.
- Move `k_atr × signal-time ATR` se strictly bada -> SUCCESS, warna FAILURE.
- UP aur DOWN dono movement count hota hai. Profit/stop ka trade model nahi hai.
- Future window incomplete -> CENSORED.
- Agle lookahead bars mein compression dobara aaye toh CONTINUATION ho sakta hai.
- Continuation ko exclude/success/failure/ignore karna configurable hai; default exclude hai.
- Important: full series par future move calculate hota hai; zero-score rows baad mein filter hoti hain.
- Continuation future se bana outcome tag hai. Live entry ke waqt ise feature nahi banana.

### Tick label: `label.py`

- Intended target: next `forward_days` mein maximum absolute close move > `k × tick_size`.
- **Actual code pehle zero-score rows hata deta hai, phir forward scan karta hai.**
- Isliye next 10 rows zaroori nahi next 10 trading days hon; wo next 10 retained signal rows ho sakti hain.
- `runner.py` aur `finding_threshold/compression_threshold.py` abhi isi tick label ko call karte hain.
- Dono labeling variants intraday high/low touches nahi, closes measure karte hain.
- Multiday phase series ko seedha label doge toh forward row ka time meaning alag hoga. Joint trade analysis ke liye outcomes raw 1D par align karna chahiye.

Ye labeling distinction score intuition se alag issue hai, par comparison results par seedha effect dalta hai.

## 5. PVD ka algo

Main function: `signals/pvd/calculate.py -> calculate_pvd()`.

Current config: price horizon 3 bars; volatility window 15; model window 15; label ATR period 10.

### Step A: actual price response

- Price change = current close - 3 bars pehle ka close.
- Normal price scale = previous 1-bar close changes ka rolling standard deviation, `sigma`.
- Normalized price response = `abs(price change) / sigma`.
- Absolute value magnitude ke liye hai; original price change ka sign separately retain hota hai.
- Ye 3-bar net move hai. Code full trend structure, monotonic movement ya trend ki speed girna explicitly check nahi karta.

### Step B: relative total volume

- Last 3 bars ka total volume sum karo.
- Usko previous horizon-volume values ke rolling average se divide karo.
- Ratio > 1 -> recent normal se zyada volume; ratio < 1 -> kam volume.
- Yahan total volume hai, signed volume delta nahi.

### Step C: volume se expected price response

- `x = log(1 + volume ratio)`.
- `y = log(1 + normalized price response)`.
- Previous observations par straight-line OLS fit hota hai: `predicted y = alpha + beta × x`.
- Current x par prediction lo, phir inverse log se expected normalized response nikalo.
- Fit current observation exclude karta hai.
- **Beta positive hona compulsory nahi.** Recent data mein fitted volume-response relation weak ya negative bhi ho sakta hai.

Intuition: "Is instrument ki recent history mein itne volume par usually kitni net price movement milti thi?"

### Step D: actual vs expected difference

- Residual = actual log-response - predicted log-response.
- Current residual ko previous residuals ke mean aur standard deviation se compare karke `pvd_z` banta hai.
- Negative z: response recent residual pattern se unusually kam.
- Positive z: response recent residual pattern se unusually zyada.
- Z ka sign price UP/DOWN nahi batata. Price direction original price change se aati hai.
- Strictly speaking, z residual ke historical mean se deviation hai. Historical mean nonzero ho toh z ka sign aur raw actual-minus-model difference ka sign hamesha same hona zaroori nahi.

### Step E: signal aur score

| Condition, current config | Net price direction | Signal name | Intuition/hypothesis |
|---|---|---|---|
| z <= -1.5 | UP | UP_TREND_EXHAUSTION | UP net move ko volume ke comparison mein weak response mil raha hai |
| z <= -1.5 | DOWN | DOWN_TREND_EXHAUSTION | DOWN net move ko weak response mil raha hai |
| z >= +1.5 | UP | UP_PRICE_EXPANSION | UP response model-relative unusually bada hai |
| z >= +1.5 | DOWN | DOWN_PRICE_EXPANSION | DOWN response model-relative unusually bada hai |

- `pvd_score = min(1, abs(z) / 2)`.
- Exhaustion aur expansion ko same magnitude score mil sakta hai; unko identify karne ke liye signed z/signal name retain karna zaroori hai.
- Strength STRONG at |z| >= 1.5, EXTREME at |z| >= 2. Ye strength cutoffs fixed hain, configurable signal thresholds se independently chalte hain.
- |z| >= 2 par score saturate hota hai: z=2 aur z=5 dono score=1.

**Naam exhaustion hai, par measured fact relative low response hai.** Trend rukega ya reversal aayega, ye outcome hypothesis hai. High volume ya already established trend detection ki compulsory condition nahi hai.

## 6. PVD labels aur threshold file

### Exhaustion labels: `signals/pvd/label.py`

- UP exhaustion par DOWN reversal target; DOWN exhaustion par UP reversal target.
- Target aur opposite failure distance dono signal-time ATR × configured multiplier hain; current multiplier 1.5.
- Next 10 raw daily closes scan hote hain: target first -> SUCCESS; failure level first -> FAILURE.
- Neither -> UNRESOLVED. Intraday high/low touch count nahi hota.
- Incomplete future window bhi UNRESOLVED hai; separate CENSORED label nahi hai. Code full horizon unavailable ho toh partial window mein already reached target/stop bhi evaluate nahi karta.
- Actual `pvd_atr` true-range average hai, gaps include karta hai. Kuch comments abhi sirf high-low mean likhte hain; executable code ko follow karna hai.
- Expansion signals ka future success/failure label is file mein nahi banta.

### Threshold analysis: `finding_threshold/pvd_threshold.py`

- Har instrument separate; exhaustion mein UP+DOWN pool hote hain.
- Negative threshold grid ka least extreme cutoff lekar candidate exhaustion rows banti hain.
- Phir |z| thresholds sweep hote hain: counts, precision, recall, F1, MCC, lift aur bootstrap precision interval.
- Ye metrics initial candidate exhaustion population ke andar hain; sab market dates ki overall detection accuracy nahi hain.
- **Current config mein `exhaustion_merge_unresolved=True` hai:** UNRESOLVED ko sweep/distribution mein SUCCESS count karta hai.
- Isse reversal na milne wale cases, aur incomplete-window cases, bhi successful count ho sakte hain. Ye label ke "opposite target reached = success" meaning se alag hai.
- Exhaustion ko trend rukna, reversal, ya sirf no continuation kis meaning mein judge karna hai, pehle define karna zaroori hai.
- Expansion section buckets/scatter dikhata hai; future reversion performance ka threshold sweep nahi karta.

### Expansion ka "real residual" naam misleading hai

- File `_real_res_atr = pvd_z × pvd_atr` banati hai.
- **Ye actual extra price move nahi nikalta.** Standardized log residual ko ATR se multiply karna original log model ka inverse nahi hai.
- Product ki unit price hai; ise directly "itna ATR overshoot" ya "itni correction expected" nahi bol sakte.
- Positive value price UP direction bhi nahi batati: DOWN expansion ka z bhi positive hota hai.
- Actual model-relative overshoot ke liye normalized actual aur predicted response retain karo; difference × same sigma price-unit comparison de sakta hai. ATR mein express karna ho toh us price difference ko ATR se divide karo. Ye bhi fitted-model comparison hoga, future correction target nahi.

## 7. Log/distribution assumptions: kya test karna useful hoga?

Ye choices current code mein hain; code review se unka best hona prove nahi hota.

| Choice | Intuition | Useful comparison |
|---|---|---|
| Compression ke logs | Large values ka influence soften karo | Raw/ATR-normalized features vs log features |
| `log(volume) / log(range)` style efficiency | Activity ke badle kitna range mil raha hai? | Normalized volume/range, percentile, ya fitted volume-range mismatch |
| Median/MAD aur 1.4826 | Outliers ka effect kam; usual scale se compare karo | Rank/quantile scores; low-MAD days alag inspect karo |
| Compression weights, /2.5 cap, confirmation gate | Multiple components se confirmation | Component-wise value; weights/gate/cap change karke stable result dekho |
| PVD log-log OLS | Recent volume se usual response estimate karo | Simple volume buckets; robust fit; fit stability aur beta sign dekho |
| PVD mean/std residual z | Model mismatch unusually large hai? | Median/MAD residual ya empirical residual percentile |
| PVD horizon response / 1-bar sigma | Movement ko recent volatility scale par compare karo | Same-horizon historical movement scale se compare karo |

- Z-score banana normal distribution prove nahi karta. -1.5/+1.5 ko fixed probability samajhna sahi nahi.
- 1.4826 normal-distribution reference ki scale calibration hai; compression actual Gaussian fit nahi karta.
- ATR/RV/range related features hain. Teen votes ko teen independent confirmations mat count karna.
- Chhota MAD/std epsilon se divide hokar very large z bana sakta hai; stale/illiquid instruments alag inspect karo.
- `log1p` price units par depend karta hai: points ko ticks mein convert karne se transformed feature badal sakta hai.
- PVD fitted log-response negative ho sakta hai; inverse-log expected response negative bhi ho sakta hai. Nonnegative magnitude prediction ke liye ise inspect/handle karna future improvement hai.
- Rolling overlapping windows ka sample count independent observations ke barabar nahi hota.

## 8. Absorption ke saath overlap ka idea

Tumhara intuition plausible hai: buying/selling pressure ko chhota response mil raha ho, toh absorption, compression aur exhaustion ek phase ke alag views ho sakte hain.

**Lekin absorption aate hi PVD/compression aana compulsory nahi hai.** Features, horizons aur baselines alag hain. Same horizon par low response exhaustion se naturally related ho sakta hai; expansion aksar pehle ke bade move ki context ho sakta hai.

| Combination | Test karne wali intuition |
|---|---|
| Buy absorption + compression | Buy pressure ke bawajood narrow movement; possible DOWN trade context, direction absorption se aayegi |
| Sell absorption + compression | Sell pressure ke bawajood narrow movement; possible UP trade context |
| Buy absorption + UP exhaustion | Positive recent net move ko weak response; DOWN reversal hypothesis aligned ho sakti hai |
| Sell absorption + DOWN exhaustion | Negative recent net move ko weak response; UP reversal hypothesis aligned ho sakti hai |
| Pehle UP expansion, phir buy absorption | Pehle unusually bada UP move, phir buy pressure par progress kam; correction setup ka possible sequence |
| Pehle DOWN expansion, phir sell absorption | Pehle unusually bada DOWN move, phir sell pressure par progress kam; bounce setup ka possible sequence |
| Absorption + opposite PVD direction | Automatic extra confirmation nahi; conflicting context ki separate category |

PVD exhaustion mein low response aur absorption mein low response shared information hai. Dono aane ko independent proof samajhkar score double karna jaldi hoga.

## 9. Joint analysis ke liye simple next approach

1. Pehle absorption alone ko baseline rakho.
2. Same instrument par signal timestamp, detection window aur direction retain karo.
3. Same-day compression/PVD aur previous configured N days ke compression/PVD ko separate flags banao. Sequence aur simultaneous overlap alag compare karo.
4. Decision ke waqt available signals hi features hon. Baad mein aaya signal earlier trade ko confirm nahi kar sakta.
5. Absorption alone vs +compression vs +aligned exhaustion vs prior expansion vs conflicting PVD compare karo.
6. Har group mein unique signal/phase count, signals/month, average success size, average failure size aur expected ticks dekho. Same phase ke repeated flags ko extra trades mat count karo.
7. Outcomes raw 1D bars par consistent rakho; combined absorption ka longest-window preference retrospective hai, use live entry rule samajhkar apply mat karo.
8. Chhote samples aur unresolved/censored counts retain karo. Sab confirmations compulsory karne se signals bahut kam ho sakte hain.
9. Pehle extra signal ka incremental benefit measure karo; uske baad combined score/weights decide karo.

## Code references

- [Compression calculation](signals/compression/calculate.py)
- [Compression multiday tracks](signals/compression/calculate_multiday.py)
- [Compression ATR labels](signals/compression/label_atr.py)
- [Compression tick labels](signals/compression/label.py)
- [Compression threshold analysis](finding_threshold/compression_threshold.py)
- [PVD calculation](signals/pvd/calculate.py)
- [PVD exhaustion labels](signals/pvd/label.py)
- [PVD threshold analysis](finding_threshold/pvd_threshold.py)
- [Current config](config.py)
- [Current absorption expected-profit implementation](finding_threshold/expected_profit_analysis.py)

Is review mein signal/label code change nahi kiya gaya; ye algo, intuition aur future comparison ideas ka document hai.
