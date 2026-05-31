\begin{table*}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{4pt}
\caption{Main quantitative comparison. We report the full CHORD configuration against external decode-time baselines. For each model we report average POPE F1 across the three splits, POPE Adversarial F1, CHAIR sentence/image hallucination ratios, MMBench accuracy, and per-token inference latency (ITL, ms/token). Lower ($\downarrow$) CHAIR and ITL are better; higher ($\uparrow$) POPE/MMBench are better. Bold indicates the best result in each column. Internal CHORD operating points are analyzed separately in \Tref{tab:family_tradeoff} and \Tref{tab:operating_points}.}
\label{tab:main_results}
\begin{tabular}{l|cccccc|cccccc}
\toprule
\multicolumn{1}{c|}{} & \multicolumn{6}{c|}{\textbf{LLaVA-v1.5-7B}} & \multicolumn{6}{c}{\textbf{InstructBLIP-7B}} \\
\cmidrule(lr){2-7} \cmidrule(lr){8-13}
\textbf{Method} & Avg. F1 $\uparrow$ & Adv. F1 $\uparrow$ & CH$_S$ $\downarrow$ & CH$_I$ $\downarrow$ & MMBench $\uparrow$ & ITL $\downarrow$ & Avg. F1 $\uparrow$ & Adv. F1 $\uparrow$ & CH$_S$ $\downarrow$ & CH$_I$ $\downarrow$ & MMBench $\uparrow$ & ITL $\downarrow$ \\
\midrule
Greedy
& 0.8488 & 0.8036 & 0.2291 & 0.2046 & 68.63 & \textbf{19.73}
& 0.8659 & 0.8127 & 0.2140 & 0.3380 & 69.34 & \textbf{16.47} \\
OPERA~\cite{huang2024opera_cvpr}
& 0.8483 & 0.8042 & 0.2284 & 0.1996 & 68.65 & 21.69
& 0.8577 & 0.8327 & 0.2180 & 0.3320 & 69.27 & 16.90 \\
VCD~\cite{leng2023mitigating}
& 0.8403 & 0.8031 & 0.2183 & 0.2061 & 68.29 & 33.07
& 0.8428 & 0.8173 & 0.2060 & 0.3480 & 68.70 & 24.80 \\
DoLa~\cite{chuang2023dola}
& 0.8502 & 0.8103 & 0.3276 & 0.3517 & 68.63 & 23.11
& 0.8488 & 0.8408 & 0.3090 & 0.4260 & 69.41 & 19.03 \\
\textbf{CHORD}
& \textbf{0.8752} & \textbf{0.8453} & \textbf{0.1548} & \textbf{0.1754} & \textbf{69.78} & 37.31
& \textbf{0.8924} & \textbf{0.8651} & \textbf{0.1347} & \textbf{0.2795} & \textbf{70.82} & 35.86 \\
\bottomrule
\end{tabular}
\end{table*}

\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{POPE Adversarial precision-recall breakdown for representative baselines and the two main CHORD operating points. \textit{CHORD-P+C} primarily improves robustness by raising precision without a large recall collapse, which is consistent with a better admission rule rather than a merely more conservative decoder.}
\label{tab:adv_breakdown}
\begin{tabular}{l l c c}
\toprule
\textbf{Model} & \textbf{Method} & Prec.~$\uparrow$ & Rec.~$\uparrow$ \\
\midrule
\multirow{5}{*}{LLaVA}
& Greedy        & 0.730 & \textbf{0.893} \\
& OPERA         & 0.732 & 0.893 \\
& VCD           & 0.748 & 0.867 \\
& CHORD-P+C     & 0.817 & 0.847 \\
& Full CHORD    & \textbf{0.842} & 0.849 \\
\midrule
\multirow{5}{*}{InstructBLIP}
& Greedy        & 0.772 & 0.858 \\
& OPERA         & 0.791 & \textbf{0.879} \\
& VCD           & 0.775 & 0.865 \\
& CHORD-P+C     & 0.841 & 0.863 \\
& Full CHORD    & \textbf{0.864} & 0.866 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Component Trade-off Within the CHORD Family}

To isolate the CHORD family itself, \Tref{tab:family_tradeoff} compares four representative variants across both backbones. This view clarifies how each component changes the decoder once the main cross-method comparison has established the overall performance pattern.

\begin{table}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{3.0pt}
\caption{Trade-off within the CHORD family across both backbones. \textit{Past} is the lowest-latency correction, \textit{Past+Current} gives the best efficiency-quality compromise, and Full CHORD gives the strongest overall quality metrics.}
\label{tab:family_tradeoff}
\begin{tabular}{lcccccc}
\toprule
\multicolumn{7}{c}{\textbf{LLaVA-v1.5-7B}} \\
\textbf{Variant} & Avg. & Adv. & CH$_S$ & CH$_I$ & MMB & ITL \\
\midrule
Past      & 0.8583 & 0.8196 & 0.2042 & 0.1927 & 68.79 & \textbf{24.38} \\
Past+C    & 0.8647 & 0.8318 & 0.1745 & 0.1852 & 69.15 & 27.24 \\
Past+F    & 0.8625 & 0.8251 & 0.1794 & 0.1873 & 69.54 & 34.62 \\
Full      & \textbf{0.8752} & \textbf{0.8453} & \textbf{0.1548} & \textbf{0.1754} & \textbf{69.78} & 37.31 \\
\midrule
\multicolumn{7}{c}{\textbf{InstructBLIP-7B}} \\
\textbf{Variant} & Avg. & Adv. & CH$_S$ & CH$_I$ & MMB & ITL \\
\midrule
Past      & 0.8715 & 0.8413 & 0.1842 & 0.3158 & 69.64 & \textbf{21.43} \\
Past+C    & 0.8806 & 0.8517 & 0.1543 & 0.2946 & 70.08 & 24.51 \\
Past+F    & 0.8789 & 0.8485 & 0.1651 & 0.3042 & 70.43 & 32.55 \\
Full      & \textbf{0.8924} & \textbf{0.8651} & \textbf{0.1347} & \textbf{0.2795} & \textbf{70.82} & 35.86 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!b]
\centering
\scriptsize
\setlength{\tabcolsep}{3.5pt}
\caption{Deployment-oriented operating-point summary relative to Greedy. Negative $\Delta$CH values indicate hallucination reduction; positive $\Delta$ITL indicates extra latency. Higher $\Delta$Adv.\ F1/$\Delta$MMB are better, lower $\Delta$CH are better.}
\label{tab:operating_points}
\begin{tabular}{l l c c c c c}
\toprule
\textbf{Model} & \textbf{Regime} & $\Delta$Adv.\ F1~$\uparrow$ & $\Delta$CH$_S$~$\downarrow$ & $\Delta$CH$_I$~$\downarrow$ & $\Delta$MMB~$\uparrow$ & $\Delta$ITL~$\downarrow$ \\
\midrule
\multirow{2}{*}{LLaVA}
& P+C & +0.0282 & -0.0546 & -0.0194 & +0.52 & \textbf{+7.51} \\
& Full & \textbf{+0.0417} & \textbf{-0.0743} & \textbf{-0.0292} & \textbf{+1.15} & +17.58 \\
\midrule
\multirow{2}{*}{InstructBLIP}
& P+C & +0.0390 & -0.0597 & -0.0434 & +0.74 & \textbf{+8.04} \\
& Full & \textbf{+0.0524} & \textbf{-0.0793} & \textbf{-0.0585} & \textbf{+1.48} & +19.39 \\
\bottomrule
\end{tabular}
\end{table}