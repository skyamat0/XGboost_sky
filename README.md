# XGboost_sky
full-scratch implementation by sky

## XGboostとは

### データ定義からコスト関数まで
データ集合を
```math
\mathcal{D}={(\bm{x}_n, y_n)}\left(|D| = N, \bm{x} \in \mathbb{R}^M, y \in \mathbb{R}\right)
```
と定義する。
この時、$\bm{x}_n$を入力した時のモデルによる出力値$\hat{y}_n$は
```math
\hat{y}_{n} = \phi(\bm{x}_n) = \sum_{j=1}^{J}f_j(\bm{x}_n)
```
である。
ここで$f_j$は個々の決定木の出力で、$f_j\in \mathcal{F}, \mathcal{F}=\{f_j(\bm{x}_n)\}_{j=1}^{J}$である。

今、$j$番目の出力時点での木の葉の数を$T_j$とし、$t \in \{1,...,T_j\}$の葉の重みを${w_t^{(j)}}$とすると、正則化付きの目的関数（後では、単に目的関数と呼ぶ）は
```math
\mathcal{L} = \sum_{n=1}^{N} l(y_n, \hat{y}_n) + \sum_{j=1}^{J}\Omega(f_j) \cdots (1)
```
であり，ここで
```math
\Omega(f_j) = \gamma T_j + \frac{1}{2} \lambda \sum_{t=1}^{T_j}w_t^{(j)}^2
```
である。この元で、次のようなモデルの最適化を考える。
1. $(j-1)$番目までの（１）式を最小化
2. 予測値$\hat{y}^{(j-1)}$を固定（この予測値は$(j-1)$番目の決定木の予測値ではなく、$(j-1)$番目までのすべての予測値の累積和であることに注意）
2. $(j)$番目の出力を追加して、（１）式を最小化

つまり、（１）式を
```math
\mathcal{L}^{(j)} = \sum_{n=1}^{N} l(y_n, y_n^{(j-1)}+f_j(\bm{x}_n))+\Omega(f_j)
```
のような目的関数として考える。
この式を直接微分して最小化するのはかなり計算コストがかかる（らしい）ので、$ l(y_n, y_n^{(j-1)}+f_j(\bm{x}_n))$のテイラー展開による２次近似を考える。
この２次近似は
```math
l(y_n, y_n^{(j-1)}+f_j(\bm{x}_n)) = l(y_n, y_n^{(j-1)}) + g_n f_j(\bm{x}_n) + \frac{1}{2}h_n \{f_j(\bm{x}_n)\}^2
```
で与えられ、$g_n=\frac{\partial }{\partial y_n^{(j-1)}}l$、$h_n=\left(\frac{\partial}{\partial y_n^{(j-1)}}\right)^2 l$である。

今、$(j-1)$番目までの決定木は固定化しているので$l(y_n, y_n^{(j-1)})=const.$である。
よって、$j$番目の決定木を追加した時に最小化される目的関数は
```math
\begin{align}
\mathcal{L}^{(j)} &\simeq \tilde{\mathcal{L}}^{(j)} \\
&= \sum_{n=1}^{N} \left[ l(y_n, y_n^{(j-1)}) + g_n f_j(\bm{x}_n) + \frac{1}{2}h_n \{f_j(\bm{x}_n)\}^2\right] + \Omega(f_j) \\
&= \sum_{n=1}^{N} \left[g_n f_j(\bm{x}_n) + \frac{1}{2}h_n \{f_j(\bm{x}_n)\}^2\right] + \Omega(f_j) + const.
\end{align} \cdots (2)
```
となる。

ここで、分割$\mathcal{N}$を$\mathcal{N}=\{N_t\}_{t=1}^{T_j}$、$N_t=\{n|f_j(\bm{x}_n)=w_t^{(j)}\}$で与える（$N_t$は簡単に言えば、ある葉の重み$w_t$を出力するデータ$\bm{x}_n$のインデックス集合）。
これを用いて、（２）式は
```math
\begin{align}
\tilde{\mathcal{L}}^{(j)} &= \sum_{n=1}^{N} \left[g_n f_j(\bm{x}_n) + \frac{1}{2}h_n \{f_j(\bm{x}_n)\}^2\right] + \gamma T_j + \frac{1}{2} \lambda \sum_{t=1}^{T_j}w_t^{(j)}^2 + const. \\
&= \sum_{t=1}^{T_j}\left[w_t^{(j)}\sum_{n \in N_t}g_n + \frac{1}{2}{(w_t^{j})}^2 \sum_{n \in N_t}h_n \right] + \gamma T_j + \frac{1}{2} \lambda \sum_{t=1}^{T_j}w_t^{(j)}^2 + const.\\
&= \sum_{t=1}^{T_j}\left[w_t^{(j)}\sum_{n \in N_t}g_n + \frac{1}{2}{(w_t^{j})}^2 \sum_{n \in N_t}h_n +\frac{1}{2} \lambda w_t^{(j)}^2 \right] + \gamma T_j +const. \\
&= \sum_{t=1}^{T_j}\left[w_t^{(j)}\sum_{n \in N_t}g_n + \frac{1}{2}(w_t^{(j)})^2 \{\sum_{n \in N_t}h_n + \lambda \}  \right] + \gamma T_j +const. 
\end{align}
```
となる。
これを$w_t^{(j)}$について微分して最小値を求めると、
```math
w_t^{(j)} = 
```
## References
([1]XGBoost: A Scalable Tree Boosting System. T.Chen, C.Guestrin, 2016)[https://arxiv.org/abs/1603.02754]