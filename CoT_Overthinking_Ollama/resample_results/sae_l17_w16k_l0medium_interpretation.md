# Interpreting the top right->wrong features

Model: `google/gemma-3-4b-it`, SAE `gemma-scope-2-4b-it-res/layer_17_width_16k_l0_medium`.
Contrast: 108 right->wrong vs 396 right->right MMLU-Pro math questions.

## How to read these numbers

- **Cohen's d** is the between-group effect size in units of pooled standard deviation. Rough interpretation: 0.2 = small, 0.5 = medium, 0.8 = large. Our top feature is d = +0.80, and ~10 features sit in the medium-large band (0.5-0.7).
- **active_RW / active_RR** is the fraction of questions in each group where the feature fires at all. This separates two very different feature profiles:
    - *Always-on amplitude shifters* (e.g. feat 279, 589): fire in ~100% of both groups, just harder in right->wrong. Probably generic `math-problem-ness` directions that are slightly stronger on the harder questions.
    - *Sparse trigger features* (e.g. feat 4103 — 11% vs 1%, feat 2072 — 17% vs 3%, feat 1659 — 7% vs 1%): fire only on a subset of questions, and that subset is heavily skewed to right->wrong. These are the interesting candidates for a semantic interpretation, because they mark specific kinds of question.
- **Multiple-testing caveat**: we tested 16384 features, so some separation is expected by chance. Cohen's d > 0.5 with N=504 is unlikely to be noise, but for individual features you should treat the d value as a *ranking* rather than a calibrated p-value.
- **Class imbalance caveat**: n(RW)=108 vs n(RR)=396. Means for RR are therefore more stable; a rarely-firing feature with only ~1% active rate in RR has very few samples to estimate its mean from — big relative differences can be partly noise.
- **Not evidence of causation**: these features *correlate with* questions the model gets right early then abandons. They don't prove that suppressing the feature would fix the flip. Test that with an activation patching / steering experiment.

## Patterns in the top 12

The top-12 right->wrong features split into roughly three types (eyeballing the table):

1. **Universal-but-stronger** (active_RW ≈ active_RR ≈ 1.00): feats 279, 589, 337, 166, 6055. These are always on; the model simply activates them harder on harder questions. Low interpretive value on their own — they probably encode 'this is a math word problem' or similar.
2. **Moderately selective** (active_RW ~ 0.4-0.6, active_RR ~ 0.2-0.4): feats 471, 346, 565, 598, 10753. These fire on about half of right->wrong questions and half as often on right->right. They're the best candidates for topic-level features (e.g. a specific type of algebra / probability wording).
3. **Sharp selective triggers** (active_RW < 0.2, but 5-10× the RR rate): feats 4103, 2072, 1659, 4551, 3871. These fire rarely, but when they do the question is much more likely to be a right->wrong case. These are the best targets for Neuronpedia inspection.

## Top questions per feature

For each feature, the right->wrong questions where it fires hardest. Compare the question prompts — shared vocabulary / structure across the top questions is your interpretation signal.

### Feature 279 — d=+0.804, active 1.00 vs 1.00, [Neuronpedia](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/279)

Top right->wrong activations:

- **1385.8** — qid=8433 [math]: Find all positive integers $n<2^{250}$ for which simultaneously $n$ divides $2^n$, $n-1$ divides $2^n-1$, and $n-2$ divides $2^n - 2$. Return all positive integers as an ascending list.
- **1341.7** — qid=7703 [math]: An urn contains four balls numbered 1 through 4 . The balls are selected one at a time without replacement. A match occurs if the ball numbered $m$ is the $m$ th ball selected. Let the event $A_i$ denote a match on the $...
- **1294.1** — qid=8448 [math]: A mass weighing $2 \mathrm{lb}$ stretches a spring 6 in. If the mass is pulled down an additional 3 in. and then released, and if there is no damping, determine the position $u$ of the mass at any time $t$. Find the freq...
- **1286.6** — qid=8485 [math]: Consider a hypothesis test with H0 : μ = 70 and Ha : μ < 70. Which of the following choices of significance level and sample size results in the greatest power of the test when μ = 65?
- **1265.6** — qid=8126 [math]: Let X_2,X_3,... be independent random variables such that $P(X_n=n)=P(X_n=-n)=1/(2n\log (n)), P(X_n=0)=1-1/(n*\log(n))$. Does $n^{-1}\sum_{i=2}^n X_i$ converges in probability? Does $n^{-1}\sum_{i=2}^n X_i$ converges in ...

For contrast, top right->right activations of the same feature:

- 1395.6 — qid=8338: Given the following equation: x^4 - x - 10 = 0. determine the initial approximations for finding the smallest positive root. Use these to find the root correct to three decimal pla...
- 1391.3 — qid=7679: What is the smallest number of vertices in a graph that guarantees the existence of a clique of size 3 or an independent set of size 2?
- 1332.0 — qid=8391: Let S be a compact topological space, let T be a topological space, and let f be a function from S onto T. Of the following conditions on f, which is the weakest condition sufficie...

### Feature 4103 — d=+0.655, active 0.11 vs 0.01, [Neuronpedia](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/4103)

Top right->wrong activations:

- **112.8** — qid=8002 [math]: A cylindrical tank of height 4 m and radius 1 m is filled with water. Water drains through a square hole of side 2 cm in the bottom. How long does it take for the tank to go from full to empty?
- **110.9** — qid=8110 [math]: compute the integral $\iint_{\Sigma} x^3 dy*dz +y^3 dz*dx+z^3 dx*dy$, where is the outward of the ellipsoid x^2+y^2+z^2/4=1. Round the answer to the thousands decimal.
- **99.9** — qid=8322 [math]: What is 3^(3^(3^(...))) mod 100? There are 2012 3's in the expression.
- **94.2** — qid=7690 [math]: Apply the Graeffe's root squaring method to find the roots of the following equation x^3 - 2x + 2 = 0 correct to two decimals. What's the sum of these roots?
- **91.7** — qid=8123 [math]: In how many ways can a group of 10 people be divided into 3 non-empty subsets?

For contrast, top right->right activations of the same feature:

- 91.8 — qid=8021: Please solve x^3 + 2*x = 10 using newton-raphson method.
- 90.2 — qid=7781: suppose I=[0,1]\times[0,1], where exp is the exponential function. What is the numeric of the double integral of the function f(x,y)=x*y^3 exp^{x^2+y^2} over I?

### Feature 589 — d=+0.625, active 1.00 vs 1.00, [Neuronpedia](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/589)

Top right->wrong activations:

- **651.3** — qid=7703 [math]: An urn contains four balls numbered 1 through 4 . The balls are selected one at a time without replacement. A match occurs if the ball numbered $m$ is the $m$ th ball selected. Let the event $A_i$ denote a match on the $...
- **637.3** — qid=7892 [math]: How many ways are there to color the faces of a cube with three colors, up to rotation?
- **603.9** — qid=7774 [math]: What is the smallest positive integer $n$ such that $\frac{1}{n}$ is a terminating decimal and $n$ contains the digit 9?
- **584.1** — qid=8126 [math]: Let X_2,X_3,... be independent random variables such that $P(X_n=n)=P(X_n=-n)=1/(2n\log (n)), P(X_n=0)=1-1/(n*\log(n))$. Does $n^{-1}\sum_{i=2}^n X_i$ converges in probability? Does $n^{-1}\sum_{i=2}^n X_i$ converges in ...
- **580.1** — qid=8230 [math]: Evaluate $\lim _{x \rightarrow 1^{-}} \prod_{n=0}^{\infty}(\frac{1+x^{n+1}}{1+x^n})^{x^n}$?

For contrast, top right->right activations of the same feature:

- 637.4 — qid=7877: How many ways are there to color the vertices of a cube with two colors, up to rotation?
- 631.8 — qid=7992: Four packages are delivered to four houses, one to each house. If these packages are randomly delivered, what is the probability that exactly two of them are delivered to the corre...
- 626.5 — qid=8391: Let S be a compact topological space, let T be a topological space, and let f be a function from S onto T. Of the following conditions on f, which is the weakest condition sufficie...

### Feature 471 — d=+0.585, active 0.49 vs 0.28, [Neuronpedia](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/471)

Top right->wrong activations:

- **281.3** — qid=8433 [math]: Find all positive integers $n<2^{250}$ for which simultaneously $n$ divides $2^n$, $n-1$ divides $2^n-1$, and $n-2$ divides $2^n - 2$. Return all positive integers as an ascending list.
- **269.2** — qid=7708 [math]: Statement 1 | Some abelian group of order 45 has a subgroup of order 10. Statement 2 | A subgroup H of a group G is a normal subgroup if and only if thenumber of left cosets of H is equal to the number of right cosets of...
- **249.0** — qid=8126 [math]: Let X_2,X_3,... be independent random variables such that $P(X_n=n)=P(X_n=-n)=1/(2n\log (n)), P(X_n=0)=1-1/(n*\log(n))$. Does $n^{-1}\sum_{i=2}^n X_i$ converges in probability? Does $n^{-1}\sum_{i=2}^n X_i$ converges in ...
- **223.2** — qid=8085 [math]: Statement 1 | Suppose f : [a, b] is a function and suppose f has a local maximum. f'(x) must exist and equal 0? Statement 2 | There exist non-constant continuous maps from R to Q.
- **211.8** — qid=8467 [math]: Statement 1 | A factor group of a non-Abelian group is non-Abelian. Statement 2 | If K is a normal subgroup of H and H is a normal subgroup of G, then K is a normal subgroup of G.

For contrast, top right->right activations of the same feature:

- 309.1 — qid=8207: Let $M=4a^2 - 2b^2 +a$. Let $j$ be the value of $M$ when $a=5$ and $b=3$, and let $k$ be the value of $M$ when $a=-1$ and $b=4$. Calculate $j+2k$.
- 220.4 — qid=7509: Statement 1 | Every group of order 159 is cyclic. Statement 2 | Every group of order 102 has a nontrivial proper normal subgroup.
- 220.3 — qid=7819: Statement 1 | In a finite dimensional vector space every linearly independent set of vectors is contained in a basis. Statement 2 | If B_1 and B_2 are bases for the same vector spa...

### Feature 2072 — d=+0.572, active 0.17 vs 0.03, [Neuronpedia](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/2072)

Top right->wrong activations:

- **204.6** — qid=8433 [math]: Find all positive integers $n<2^{250}$ for which simultaneously $n$ divides $2^n$, $n-1$ divides $2^n-1$, and $n-2$ divides $2^n - 2$. Return all positive integers as an ascending list.
- **196.2** — qid=8230 [math]: Evaluate $\lim _{x \rightarrow 1^{-}} \prod_{n=0}^{\infty}(\frac{1+x^{n+1}}{1+x^n})^{x^n}$?
- **174.2** — qid=7604 [math]: Determine whether the polynomial in Z[x] satisfies an Eisenstein criterion for irreducibility over Q. 8x^3 + 6x^2 - 9x + 24
- **166.7** — qid=8036 [math]: Use Stokes' Theorem to evaluate $\int_C \mathbf{F} \cdot d \mathbf{r}$, where $\mathbf{F}(x, y, z)=x y \mathbf{i}+y z \mathbf{j}+z x \mathbf{k}$, and $C$ is the triangle with vertices $(1,0,0),(0,1,0)$, and $(0,0,1)$, or...
- **166.6** — qid=8110 [math]: compute the integral $\iint_{\Sigma} x^3 dy*dz +y^3 dz*dx+z^3 dx*dy$, where is the outward of the ellipsoid x^2+y^2+z^2/4=1. Round the answer to the thousands decimal.

For contrast, top right->right activations of the same feature:

- 194.7 — qid=7508: Find all c in Z_3 such that Z_3[x]/(x^3 + x^2 + c) is a field.
- 170.3 — qid=7509: Statement 1 | Every group of order 159 is cyclic. Statement 2 | Every group of order 102 has a nontrivial proper normal subgroup.
- 168.0 — qid=7994: What is the greatest common divisor of $2^{1001}-1$ and $2^{1012}-1$?

### Feature 346 — d=+0.564, active 0.58 vs 0.32, [Neuronpedia](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/346)

Top right->wrong activations:

- **398.5** — qid=8433 [math]: Find all positive integers $n<2^{250}$ for which simultaneously $n$ divides $2^n$, $n-1$ divides $2^n-1$, and $n-2$ divides $2^n - 2$. Return all positive integers as an ascending list.
- **295.8** — qid=8453 [math]: Consider the initial value problem $$ y^{\prime}+\frac{2}{3} y=1-\frac{1}{2} t, \quad y(0)=y_0 . $$ Find the value of $y_0$ for which the solution touches, but does not cross, the $t$-axis.
- **285.4** — qid=8448 [math]: A mass weighing $2 \mathrm{lb}$ stretches a spring 6 in. If the mass is pulled down an additional 3 in. and then released, and if there is no damping, determine the position $u$ of the mass at any time $t$. Find the freq...
- **283.2** — qid=7774 [math]: What is the smallest positive integer $n$ such that $\frac{1}{n}$ is a terminating decimal and $n$ contains the digit 9?
- **275.7** — qid=8110 [math]: compute the integral $\iint_{\Sigma} x^3 dy*dz +y^3 dz*dx+z^3 dx*dy$, where is the outward of the ellipsoid x^2+y^2+z^2/4=1. Round the answer to the thousands decimal.

For contrast, top right->right activations of the same feature:

- 348.2 — qid=7555: Suppose that $f(x)$ is a polynomial that has degree $6$ and $g(x)$ is a polynomial that has degree $3$. If $h(x)$ is also a polynomial such that $f(g(x)) + g(h(x)) + h(f(x))$ is a ...
- 319.4 — qid=7558: Given that $a$ and $b$ are real numbers such that $-3\leq a\leq1$ and $-2\leq b\leq 4$, and values for $a$ and $b$ are chosen at random, what is the probability that the product $a...
- 307.7 — qid=7862: Let $C$ be the circle with equation $x^2+12y+57=-y^2-10x$. If $(a,b)$ is the center of $C$ and $r$ is its radius, what is the value of $a+b+r$?

### Feature 1659 — d=+0.519, active 0.07 vs 0.01, [Neuronpedia](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/1659)

Top right->wrong activations:

- **172.3** — qid=8433 [math]: Find all positive integers $n<2^{250}$ for which simultaneously $n$ divides $2^n$, $n-1$ divides $2^n-1$, and $n-2$ divides $2^n - 2$. Return all positive integers as an ascending list.
- **151.7** — qid=7802 [math]: Please solve the equation sin(4*x) + x = 54 and provide all the roots using newton-raphson method.
- **149.2** — qid=8110 [math]: compute the integral $\iint_{\Sigma} x^3 dy*dz +y^3 dz*dx+z^3 dx*dy$, where is the outward of the ellipsoid x^2+y^2+z^2/4=1. Round the answer to the thousands decimal.
- **140.4** — qid=7882 [math]: Assuming we are underground, and the only thing we can observe is whether a person brings an umbrella or not. The weather could be either rain or sunny. Assuming the P(rain)=0.6 and P(sunny)=0.4. Assuming the weather on ...
- **131.0** — qid=7690 [math]: Apply the Graeffe's root squaring method to find the roots of the following equation x^3 - 2x + 2 = 0 correct to two decimals. What's the sum of these roots?

For contrast, top right->right activations of the same feature:

- 126.5 — qid=8338: Given the following equation: x^4 - x - 10 = 0. determine the initial approximations for finding the smallest positive root. Use these to find the root correct to three decimal pla...
- 112.1 — qid=7509: Statement 1 | Every group of order 159 is cyclic. Statement 2 | Every group of order 102 has a nontrivial proper normal subgroup.

### Feature 337 — d=+0.483, active 1.00 vs 0.99, [Neuronpedia](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/337)

Top right->wrong activations:

- **395.9** — qid=8230 [math]: Evaluate $\lim _{x \rightarrow 1^{-}} \prod_{n=0}^{\infty}(\frac{1+x^{n+1}}{1+x^n})^{x^n}$?
- **394.5** — qid=8453 [math]: Consider the initial value problem $$ y^{\prime}+\frac{2}{3} y=1-\frac{1}{2} t, \quad y(0)=y_0 . $$ Find the value of $y_0$ for which the solution touches, but does not cross, the $t$-axis.
- **370.5** — qid=7703 [math]: An urn contains four balls numbered 1 through 4 . The balls are selected one at a time without replacement. A match occurs if the ball numbered $m$ is the $m$ th ball selected. Let the event $A_i$ denote a match on the $...
- **363.2** — qid=8485 [math]: Consider a hypothesis test with H0 : μ = 70 and Ha : μ < 70. Which of the following choices of significance level and sample size results in the greatest power of the test when μ = 65?
- **357.3** — qid=7804 [math]: A model for the surface area of a human body is given by $S=0.1091 w^{0.425} h^{0.725}$, where $w$ is the weight (in pounds), $h$ is the height (in inches), and $S$ is measured in square feet. If the errors in measuremen...

For contrast, top right->right activations of the same feature:

- 447.6 — qid=7555: Suppose that $f(x)$ is a polynomial that has degree $6$ and $g(x)$ is a polynomial that has degree $3$. If $h(x)$ is also a polynomial such that $f(g(x)) + g(h(x)) + h(f(x))$ is a ...
- 379.8 — qid=7558: Given that $a$ and $b$ are real numbers such that $-3\leq a\leq1$ and $-2\leq b\leq 4$, and values for $a$ and $b$ are chosen at random, what is the probability that the product $a...
- 375.9 — qid=8329: Let N be a spatial Poisson process with constant intensity $11$ in R^d, where d\geq2. Let S be the ball of radius $r$ centered at zero.  Denote |S| to be the volume of the ball. Wh...

### Feature 166 — d=+0.474, active 0.95 vs 0.89, [Neuronpedia](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/166)

Top right->wrong activations:

- **382.7** — qid=7808 [math]: Consider the initial value problem (see Example 5) $$ y^{\prime \prime}+5 y^{\prime}+6 y=0, \quad y(0)=2, \quad y^{\prime}(0)=\beta $$ where $\beta>0$. Determine the smallest value of $\beta$ for which $y_m \geq 4$.
- **361.6** — qid=8110 [math]: compute the integral $\iint_{\Sigma} x^3 dy*dz +y^3 dz*dx+z^3 dx*dy$, where is the outward of the ellipsoid x^2+y^2+z^2/4=1. Round the answer to the thousands decimal.
- **348.5** — qid=7575 [math]: Compute $\int_C dz / (z * (z-2)^2)dz$, where C: |z - 2| = 1. The answer is Ai with i denoting the imaginary unit, what is A?
- **335.6** — qid=8448 [math]: A mass weighing $2 \mathrm{lb}$ stretches a spring 6 in. If the mass is pulled down an additional 3 in. and then released, and if there is no damping, determine the position $u$ of the mass at any time $t$. Find the freq...
- **331.0** — qid=8453 [math]: Consider the initial value problem $$ y^{\prime}+\frac{2}{3} y=1-\frac{1}{2} t, \quad y(0)=y_0 . $$ Find the value of $y_0$ for which the solution touches, but does not cross, the $t$-axis.

For contrast, top right->right activations of the same feature:

- 383.8 — qid=7555: Suppose that $f(x)$ is a polynomial that has degree $6$ and $g(x)$ is a polynomial that has degree $3$. If $h(x)$ is also a polynomial such that $f(g(x)) + g(h(x)) + h(f(x))$ is a ...
- 345.3 — qid=8007: Compute $\int_{|z| = 2} (5z - 2) / (z * (z - 1)) dz$. The answer is Ai with i denoting the imaginary unit, what is A?
- 344.0 — qid=7782: Let $A=\{n+\sum_{p=1}^{\infty} a_p 2^{-2p}: n \in \mathbf{Z}, a_p=0 or 1 \}$. What is the Lebesgue measure of A?

### Feature 6055 — d=+0.473, active 0.97 vs 0.91, [Neuronpedia](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/6055)

Top right->wrong activations:

- **192.3** — qid=7701 [math]: 9.6-111. Let $X$ and $Y$ have a bivariate normal distribution with correlation coefficient $\rho$. To test $H_0: \rho=0$ against $H_1: \rho \neq 0$, a random sample of $n$ pairs of observations is selected. Suppose that ...
- **183.8** — qid=7798 [math]: what is the limit of $2/\sqrt{\pi}*\sqrt{n}\int_0^1(1-x^2)^n dx$ as n goes to infinity?
- **179.1** — qid=8039 [math]: Statement 1 | Every homomorphic image of a group G is isomorphic to a factor group of G. Statement 2 | The homomorphic images of a group G are the same (up to isomorphism) as the factor groups of G.
- **178.7** — qid=7892 [math]: How many ways are there to color the faces of a cube with three colors, up to rotation?
- **177.1** — qid=7882 [math]: Assuming we are underground, and the only thing we can observe is whether a person brings an umbrella or not. The weather could be either rain or sunny. Assuming the P(rain)=0.6 and P(sunny)=0.4. Assuming the weather on ...

For contrast, top right->right activations of the same feature:

- 186.2 — qid=7571: How many triangles are there whose sides are all integers and whose maximum side length equals 11?
- 184.1 — qid=7844: Use the equation below to answer the question. 14 × 3 = 42 Which statement correctly interprets the expression?
- 180.8 — qid=8303: Let $f(x) = 3x^2-2$ and $g(f(x)) = x^2 + x +1$. Find the sum of all possible values of $g(25)$.

### Feature 4551 — d=+0.455, active 0.11 vs 0.02, [Neuronpedia](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/4551)

Top right->wrong activations:

- **271.3** — qid=8467 [math]: Statement 1 | A factor group of a non-Abelian group is non-Abelian. Statement 2 | If K is a normal subgroup of H and H is a normal subgroup of G, then K is a normal subgroup of G.
- **257.0** — qid=7708 [math]: Statement 1 | Some abelian group of order 45 has a subgroup of order 10. Statement 2 | A subgroup H of a group G is a normal subgroup if and only if thenumber of left cosets of H is equal to the number of right cosets of...
- **243.6** — qid=7614 [math]: There are two games involving flipping a fair coin. In the first game you win a prize if you can throw between 45% and 55% heads. In the second game you win if you can throw more than 80% heads. For each game would you r...
- **239.4** — qid=8085 [math]: Statement 1 | Suppose f : [a, b] is a function and suppose f has a local maximum. f'(x) must exist and equal 0? Statement 2 | There exist non-constant continuous maps from R to Q.
- **236.2** — qid=8354 [math]: Statement 1 | The set of 2 x 2 matrices with integer entries and nonzero determinant is a group under matrix multiplication. Statement 2 | The set of 2 x 2 matrices with integer entries and determinant 1 is a group under...

For contrast, top right->right activations of the same feature:

- 232.8 — qid=7509: Statement 1 | Every group of order 159 is cyclic. Statement 2 | Every group of order 102 has a nontrivial proper normal subgroup.
- 220.8 — qid=8250: Statement 1 | Every free abelian group is torsion free. Statement 2 | Every finitely generated torsion-free abelian group is a free abelian group.
- 219.2 — qid=8031: For the following functions, which are bounded entire functions? 1. f(x)=0; 2. f(x)= 1+i; 3. f(x)=sin(x); 4. f(x)=min{|cos(x)|,1}. Here i=\sqrt{-1} and $|\cdot|$ is the norm of a c...

### Feature 3871 — d=+0.422, active 0.08 vs 0.02, [Neuronpedia](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/3871)

Top right->wrong activations:

- **155.1** — qid=7808 [math]: Consider the initial value problem (see Example 5) $$ y^{\prime \prime}+5 y^{\prime}+6 y=0, \quad y(0)=2, \quad y^{\prime}(0)=\beta $$ where $\beta>0$. Determine the smallest value of $\beta$ for which $y_m \geq 4$.
- **141.2** — qid=8453 [math]: Consider the initial value problem $$ y^{\prime}+\frac{2}{3} y=1-\frac{1}{2} t, \quad y(0)=y_0 . $$ Find the value of $y_0$ for which the solution touches, but does not cross, the $t$-axis.
- **140.8** — qid=7875 [math]: Use the Trapezoidal Rule with to approximate $\int_0^{\pi} sin^2(x)dx$. Return the approximated demical value.
- **135.6** — qid=8448 [math]: A mass weighing $2 \mathrm{lb}$ stretches a spring 6 in. If the mass is pulled down an additional 3 in. and then released, and if there is no damping, determine the position $u$ of the mass at any time $t$. Find the freq...
- **123.7** — qid=7701 [math]: 9.6-111. Let $X$ and $Y$ have a bivariate normal distribution with correlation coefficient $\rho$. To test $H_0: \rho=0$ against $H_1: \rho \neq 0$, a random sample of $n$ pairs of observations is selected. Suppose that ...

For contrast, top right->right activations of the same feature:

- 139.4 — qid=8021: Please solve x^3 + 2*x = 10 using newton-raphson method.
- 131.4 — qid=8458: Solve the initial value problem $y^{\prime \prime}-y^{\prime}-2 y=0, y(0)=\alpha, y^{\prime}(0)=2$. Then find $\alpha$ so that the solution approaches zero as $t \rightarrow \infty...
- 120.3 — qid=8452: Consider the initial value problem $$ y^{\prime}+\frac{1}{4} y=3+2 \cos 2 t, \quad y(0)=0 $$ Determine the value of $t$ for which the solution first intersects the line $y=12$.

## Co-activation among top features (right->wrong questions only)

How often do pairs of top features fire on the same right->wrong question. High co-activation means the features are probably picking up related concepts (or the same direction viewed through different codes).

| feature |279 | 4103 | 589 | 471 | 2072 | 346 | 1659 | 337 | 166 | 6055 | 4551 | 3871 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 279 | 108 | 12 | 108 | 53 | 18 | 63 | 8 | 108 | 103 | 105 | 12 | 9 |
| 4103 | 12 | 12 | 12 | 7 | 3 | 11 | 5 | 12 | 12 | 12 | 0 | 3 |
| 589 | 108 | 12 | 108 | 53 | 18 | 63 | 8 | 108 | 103 | 105 | 12 | 9 |
| 471 | 53 | 7 | 53 | 53 | 16 | 45 | 8 | 53 | 51 | 52 | 12 | 7 |
| 2072 | 18 | 3 | 18 | 16 | 18 | 17 | 4 | 18 | 18 | 18 | 4 | 3 |
| 346 | 63 | 11 | 63 | 45 | 17 | 63 | 8 | 63 | 62 | 63 | 9 | 9 |
| 1659 | 8 | 5 | 8 | 8 | 4 | 8 | 8 | 8 | 8 | 8 | 3 | 3 |
| 337 | 108 | 12 | 108 | 53 | 18 | 63 | 8 | 108 | 103 | 105 | 12 | 9 |
| 166 | 103 | 12 | 103 | 51 | 18 | 62 | 8 | 103 | 103 | 100 | 10 | 9 |
| 6055 | 105 | 12 | 105 | 52 | 18 | 63 | 8 | 105 | 100 | 105 | 12 | 9 |
| 4551 | 12 | 0 | 12 | 12 | 4 | 9 | 3 | 12 | 10 | 12 | 12 | 0 |
| 3871 | 9 | 3 | 9 | 7 | 3 | 9 | 3 | 9 | 9 | 9 | 0 | 9 |

(Diagonal = number of right->wrong questions on which that feature fires at all; total right->wrong questions = 108.)

## Signature: how many of the top-12 RW features fire per question

| # top-12 features active | right->wrong | right->right |
|---|---|---|
| 0 | 0 (0%) | 0 (0%) |
| 1 | 0 (0%) | 0 (0%) |
| 2 | 0 (0%) | 0 (0%) |
| 3 | 0 (0%) | 8 (2%) |
| 4 | 5 (5%) | 50 (13%) |
| 5 | 31 (29%) | 176 (44%) |
| 6 | 19 (18%) | 90 (23%) |
| 7 | 24 (22%) | 57 (14%) |
| 8 | 18 (17%) | 10 (3%) |
| 9 | 6 (6%) | 4 (1%) |
| 10 | 3 (3%) | 1 (0%) |
| 11 | 2 (2%) | 0 (0%) |

A right->wrong question is likely to light up several of these features at once; a right->right question rarely lights up more than a couple. Whether this is predictive out-of-sample needs a held-out split.
