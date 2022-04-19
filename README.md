# CBQuant - Convertible Bond Quant Strategy
**Strategy Overview:**

1. Define 

$$
\begin{aligned} 
Premium &= \frac{BondPrice}{FaceValue} - 1 + \frac{BondPrice\;*\;StockPrice}{FaceValue\;/\;StrikePrice}- 1\\
FaceValue &= 100\\
BondPrice &= Bond\; Close\; Price \\
StockPrice &= Underlying\; Stock\; Close\; Price\\
\end{aligned}
$$
    

2. Choose 20 bonds out of 400 from data.csv with the following criteria:
    
> Bond with least premium

3. Hold the position of these 20 bonds for 20 trading days and cash out those that are no longer with least premium. Buy in new bonds with cash.

4. Each bond has **equal weight** as a starter.