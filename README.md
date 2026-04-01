# XGboost_sky
full-scratch implementation by sky

## XGboostとは

### データ定義からコスト関数まで
データ集合を
```math
\mathcal{D}={(\mathbf{x}_n, y_n)}\left(|D| = N, \mathbf{x} \in \mathbb{R}^M, y \in \mathbb{R}\right)
```
と定義する。
この時、 $\mathbf{x}_n$ を入力した時のモデルによる出力値 $\hat{y}_n$ は
```math
\hat{y}_{n} = \phi(\mathbf{x}_n) = \sum_{j=1}^{J}f_j(\mathbf{x}_n)
```
である。
ここで $f_j$ は個々の決定木の出力で、 $`f_j \in \mathcal{F}, \mathcal{F} = \{ f_j(\mathbf{x}_n) \}_{j=1}^J`$ である。

今、 $j$ 番目の出力時点での木の葉の数を $T_j$ とし、 $t \in \{1,...,T_j\}$ の葉の重みを ${w_t^{(j)}}$ とすると、正則化付きの目的関数（後では、単に目的関数と呼ぶ）は
```math
\mathcal{L} = \sum_{n=1}^{N} l(y_n, \hat{y}_n) + \sum_{j=1}^{J}\Omega(f_j) \cdots (1)
```
であり，ここで
```math
\Omega(f_j) = \gamma T_j + \frac{1}{2} \lambda \sum_{t=1}^{T_j}\left(w_t^{(j)}\right)^2
```
である。この元で、次のようなモデルの最適化を考える。
1. $(j-1)$ 番目までの（１）式を最小化
2. 予測値 $\hat{y}^{(j-1)}$ を固定（この予測値は $(j-1)$ 番目の決定木の予測値ではなく、 $(j-1)$ 番目までのすべての予測値の累積和であることに注意）
2. $(j)$ 番目の出力を追加して、（１）式を最小化

つまり、（１）式を
```math
\mathcal{L}^{(j)} = \sum_{n=1}^{N} l(y_n, y_n^{(j-1)}+f_j(\mathbf{x}_n))+\Omega(f_j)
```
のような目的関数として考える。
この式を直接微分して最小化するのはかなり計算コストがかかる（らしい）ので、 $l(y_n, y_n^{(j-1)}+f_j(\mathbf{x}_n))$ のテイラー展開による２次近似を考える。
この２次近似は
```math
l(y_n, y_n^{(j-1)}+f_j(\mathbf{x}_n)) = l(y_n, y_n^{(j-1)}) + g_n f_j(\mathbf{x}_n) + \frac{1}{2}h_n \{f_j(\mathbf{x}_n)\}^2
```
で与えられ、 $g_n=\frac{\partial }{\partial y_n^{(j-1)}}l$ 、 $h_n=\left(\frac{\partial}{\partial y_n^{(j-1)}}\right)^2 l$ である。

今、 $(j-1)$ 番目までの決定木は固定化しているので $l(y_n, y_n^{(j-1)})=const.$ である。
よって、 $j$ 番目の決定木を追加した時に最小化される目的関数は
```math
\begin{align}
\mathcal{L}^{(j)} &\simeq \tilde{\mathcal{L}}^{(j)} \\
&= \sum_{n=1}^{N} \left[ l(y_n, y_n^{(j-1)}) + g_n f_j(\mathbf{x}_n) + \frac{1}{2}h_n \{f_j(\mathbf{x}_n)\}^2\right] + \Omega(f_j) \\
&= \sum_{n=1}^{N} \left[g_n f_j(\mathbf{x}_n) + \frac{1}{2}h_n \{f_j(\mathbf{x}_n)\}^2\right] + \Omega(f_j) + const.
\end{align} \cdots (2)
```
となる。

ここで、分割 $\mathcal{N}$ を $\mathcal{N}=\{N_t\}_{t=1}^{T_j}$ 、 $N_t=\{n|f_j(\mathbf{x}_n)=w_t^{(j)}\}$ で与える（ $N_t$ は簡単に言えば、ある葉の重み $w_t$ を出力するデータ $\mathbf{x}_n$ のインデックス集合のこと）。
これを用いて、（２）式は
```math
\begin{align}
\tilde{\mathcal{L}}^{(j)} &= \sum_{n=1}^{N} \left[g_n f_j(\mathbf{x}_n) + \frac{1}{2}h_n \{f_j(\mathbf{x}_n)\}^2\right] + \gamma T_j + \frac{1}{2} \lambda \sum_{t=1}^{T_j}\left(w_t^{(j)}\right)^2 + const. \\
&= \sum_{t=1}^{T_j}\left[w_t^{(j)}\sum_{n \in N_t}g_n + \frac{1}{2}\left(w_t^{(j)}\right)^2 \sum_{n \in N_t}h_n \right] + \gamma T_j + \frac{1}{2} \lambda \sum_{t=1}^{T_j}\left(w_t^{(j)}\right)^2 + const.\\
&= \sum_{t=1}^{T_j}\left[w_t^{(j)}\sum_{n \in N_t}g_n + \frac{1}{2}{(w_t^{j})}^2 \sum_{n \in N_t}h_n +\frac{1}{2} \lambda \left(w_t^{(j)}\right)^2 \right] + \gamma T_j +const. \\
&= \sum_{t=1}^{T_j}\left[w_t^{(j)}\sum_{n \in N_t}g_n + \frac{1}{2}\left(w_t^{(j)}\right)^2 \{\sum_{n \in N_t}h_n + \lambda \}  \right] + \gamma T_j +const. 
\end{align}
```
となる。
これを $w_t^{(j)}$ について微分して最小値を求めると、
```math
w_t^{(j)} = \frac{\sum_{n \in N_t}g_n}{\sum_{n \in N_t}(h_n + \lambda)}
```
の時、
```math
\tilde{\mathcal{L}}^{(j)}_{\rm{min}} = -\frac{1}{2}\sum_{t=1}^{T_j} \frac{\left(\sum_{n \in N_t}g_n\right)^2}{\sum_{n \in N_t}(h_n + \lambda)}+\gamma T
```
となる。また、個々の木分割に関する最適化も行うので、前節で導出した最小損失 $`\tilde{L}_{\mathrm{min}}^{(j)}`$ は、木の構造（どのデータがどの葉に属するか）が固定されている場合の指標である。しかし、可能なすべての木構造を網羅的に探索することは計算量的に不可能である。そのため、実際には1つの根ノードから始めて、**貪欲法（Greedy Algorithm）**を用いて再帰的にノードを分割していく。
あるノードを分割した際の「良さ」を評価するため、分割前後の損失の減少量を 利得 (Gain) として定義する。ノード N に属するデータの集合を、ある特徴量としきい値によって $N_L​$ と $N_R$ と $N=N_L \cap N_R$ に分割したとき、その分割による利得 $`L_{\mathrm{split}}`$ は以下の式で与えられる。
```math
L_{\rm{split}} = \frac{1}{2} \left[ \frac{\left(\sum_{n \in N_L} g_n\right)^2}{\sum_{n \in N_L} h_n + \lambda} + \frac{\left(\sum_{n \in N_R} g_n\right)^2}{\sum_{n \in N_R} h_n + \lambda} - \frac{\left(\sum_{n \in N} g_n\right)^2}{\sum_{n \in N} h_n + \lambda} \right] - \gamma
```
ここで、各項は以下の意味を持つ：

1. 第1項・第2項: 分割後の左の子ノードおよび右の子ノードにおけるスコア。
2. 第3項: 分割前の元のノードにおけるスコア。
3. $\gamma$ :新たに葉を1つ増やすことに対するペナルティ（正則化項）。

学習プロセスにおいては、すべての特徴量およびすべての分割候補点についてこの$`L_{\mathrm{split}}`$を計算し、利得が最大となる分割を採択する。さらに、木の学習においては、上記アルゴリズムに加えて以下の手法が併用される。
* . Shrinkage（縮小）: 各ステップで追加される決定木の重みに学習率 η を乗じることで、個々の木の影響を抑え、後続の木がモデルを改善する余地を残す。
* Column Subsampling（特徴量サブサンプリング）: 各ノードの分割時にすべての特徴量を使わず、ランダムに選択した一部の特徴量のみを候補とすることで、多様性を確保し過学習を防ぐ。
* Early Stopping（早期終了）: 検証データに対する損失が一定回数改善しなくなった時点で学習を打ち切り、最適な木の総数 $J$ を決定する。
## References
([1]XGBoost: A Scalable Tree Boosting System. T.Chen, C.Guestrin, 2016)[https://arxiv.org/abs/1603.02754]
