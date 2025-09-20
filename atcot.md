% Template for ICASSP-2024 paper; to be used with:
%          spconf.sty  - ICASSP/ICIP LaTeX style file, and
%          IEEEbib.bst - IEEE bibliography style file.
% --------------------------------------------------------------------------
\documentclass{article}
\usepackage{spconf,amsmath,graphicx,booktabs,multirow,booktabs,afterpage}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{amssymb}
\usepackage{url}
\usepackage[table]{xcolor}
% Example definitions.
% --------------------
\def\x{{\mathbf x}}
\def\L{{\cal L}}

% Title.
% ------
\title{ATCOT: Self-Correcting Chain-of-Thought Reasoning with Tool Feedback}
%
% Single address.
% ---------------
\name{Lei Wei\textsuperscript{1,2*}, Xu Dong\textsuperscript{2*}, Xiao Peng\textsuperscript{1*}, Niantao Xie\textsuperscript{2}, Bin Wang\textsuperscript{2,3}
\footnotemark[1]}
\address{\textsuperscript{1}Alibaba International Digital Commerce Group \\
\textsuperscript{2}School of Software and Microelectronics, Peking University \\
\textsuperscript{3}XiaoMi AI Lab 
}
%
% For example:
% ------------
%\address{School\\
%	Department\\
%	Address}
%
% Two addresses (uncomment and modify for two-address case).
% ----------------------------------------------------------
%\twoauthors
%  {A. Author-one, B. Author-two\sthanks{Thanks to XYZ agency for funding.}}
%	{School A-B\\
%	Department A-B\\
%	Address A-B}
%  {C. Author-three, D. Author-four\sthanks{The fourth author performed the work
%	while at ...}}
%	{School C-D\\
%	Department C-D\\
%	Address C-D}
%
\begin{document}
\ninept
%
\maketitle
%
\renewcommand{\thefootnote}{\fnsymbol{footnote}} 
\footnotetext[1]{Equal contribution.} 
\footnotetext{Our code and data are available at \url{https://anonymous.4open.science/r/ATCOT-CED3}.} 

\begin{abstract}
Large Language Models (LLMs) have achieved remarkable success in complex reasoning through Chain-of-Thought prompting and tool augmentation. However, existing approaches like ReAct operate in a forward-only manner, unable to correct earlier reasoning steps when subsequent tool invocations reveal errors or contradictions. This inflexibility causes error propagation in multi-step reasoning, where initial mistakes cascade through the entire solution. We introduce Adaptive Tool-Augmented Chain of Thought (ATCOT), a dynamic iterative reasoning framework that enables LLMs to revise their reasoning trajectories based on tool feedback. ATCOT maintains comprehensive state representation including planning structure, reasoning trace, and tool history, with global consistency verification and bounded corrections ensuring convergence. Extensive experiments on GSM8K and HotpotQA demonstrate that ATCOT consistently outperforms strong baselines including ReAct, CoT, and CRITIC across multiple model scales, achieving the best performance within the tool-augmented reasoning paradigm. Our results validate that adaptive correction mechanisms are crucial for robust and reliable reasoning in tool-augmented LLMs.
\end{abstract}
%
\begin{keywords}
Large Language Models, Chain of Thought, Tool-Augmented Reasoning, Function Calling
\end{keywords}
%

\section{Introduction}
The rapid advancement of Large Language Models (LLMs) such as GPT-4 \cite{openai2023gpt4, achiam2023gpt}, Claude \cite{anthropic2023claude}, and LLaMA \cite{touvron2023llama, team2023gemini} has transformed artificial intelligence, particularly in natural language understanding and complex reasoning tasks. These models have demonstrated unprecedented capabilities in generating coherent, step-by-step reasoning through Chain of Thought (CoT) prompting \cite{wei2022chain, wang2022self}, which enables them to decompose complex problems into manageable intermediate steps. This breakthrough has allowed LLMs to tackle sophisticated tasks across diverse domains, from mathematical problem-solving to commonsense reasoning, achieving performance levels that often rival or exceed human experts in specific areas \cite{bubeck2023sparks}.
\begin{figure*}[!t]
\centering
\includegraphics[width=0.95\textwidth]{ATCOT.png}
\caption{Comparison between standard LLM reasoning and ATCOT. While standard LLM produces incorrect answers due to cascading errors, ATCOT dynamically corrects reasoning through adaptive planning, tool invocation, and error correction to achieve accurate results.}
\label{fig:ATCOT}
\end{figure*}

However, despite these achievements, current approaches to LLM reasoning face critical limitations when confronted with real-world complexity. While frameworks like ReAct \cite{yao2022react}, Toolformer \cite{schick2023toolformer}, and PAL \cite{gao2023pal} have successfully integrated reasoning with tool utilization, they operate in a strictly forward-only manner—once a reasoning step is executed and a tool is invoked, there is no mechanism to revisit and correct earlier steps when new information reveals flaws in initial assumptions. This inflexibility becomes particularly problematic in multi-step reasoning tasks where errors compound: a single incorrect assumption or miscalculation can cascade through the entire reasoning chain, leading to flawed conclusions \cite{suzgun2022challenging, valmeekam2023planning}. Moreover, existing methods lack sophisticated mechanisms to maintain consistency when integrating potentially conflicting information from multiple tool invocations, unlike more advanced techniques such as Tree of Thoughts \cite{yao2023tree} or Reflexion \cite{shinn2023reflexion}. Consider a mathematical problem where an early calculation error undermines all subsequent steps—current frameworks offer no systematic way to backtrack and repair these errors.

To address these fundamental challenges, we propose Adaptive Tool-Augmented Chain of Thought (ATCOT), a novel framework that enables LLMs to dynamically correct their reasoning trajectories based on tool feedback \cite{wang2023voyager, huang2022inner}. Unlike existing approaches that treat reasoning as an immutable forward progression, ATCOT introduces adaptive correction mechanisms that allow models to revisit and revise previous steps when tools provide contradicting or clarifying information, as shown in Figure~\ref{fig:ATCOT}. This capability transforms LLM reasoning from a rigid pipeline into a flexible, self-correcting process that mirrors human problem-solving more closely \cite{sumers2023cognitive, nye2021show}. When a calculator reveals a computational error or a search tool uncovers information that contradicts an earlier assumption, ATCOT can trace back through its reasoning chain, identify affected steps, and reconstruct a consistent solution path—all while maintaining transparency and interpretability.
Our main contributions are: (1) We propose ATCOT, a dynamic framework that integrates adaptive planning, tool-based execution, and corrective feedback, enabling LLMs to revise reasoning trajectories when new information reveals flaws.
(2) We develop mechanisms for state tracking, global consistency verification, and bounded correction budgets that ensure convergence while preventing infinite correction loops.
(3) Through extensive evaluation on GSM8K and HotpotQA across multiple model scales, we demonstrate that ATCOT consistently outperforms existing baselines, achieving the best performance within the tool-augmented reasoning paradigm among all evaluated methods \cite{cobbe2021training, yang2018hotpotqa}.
\section{Methodology}

\subsection{Framework Architecture and State Representation}

The Adaptive Tool-Augmented Chain of Thought framework instantiates a comprehensive state representation $\mathcal{S} = \{\mathcal{P}, \mathcal{R}, \mathcal{H}, \mathcal{C}\}$, wherein $\mathcal{P}$ denotes the current planning structure encoded as an ordered sequence of steps with explicit dependencies, $\mathcal{R}$ represents the reasoning trace comprising intermediate conclusions and their justifications, $\mathcal{H}$ encapsulates the complete tool invocation history with temporal annotations and results, and $\mathcal{C}$ maintains the correction log tracking all revisions and their triggering conditions.

Given an input query $q \in \mathcal{Q}$ and a fixed set of available tools $\mathbb{T} = \{\tau_1, \tau_2, \ldots, \tau_m\}$, our framework generates solutions through iterative adaptive reasoning:
\begin{equation}
\mathcal{S}_{t+1} = f_{\text{adapt}}(\mathcal{S}_t, o_t, \theta)
\end{equation}
where $o_t \in \mathcal{O}$ represents observations obtained from tool execution at timestep $t$, and $\theta \in \Theta$ denotes the parameters of the pretrained language model. The state composition operator $\oplus: \mathcal{S} \times \Delta \rightarrow \mathcal{S}$ is formally defined as:
\begin{equation}
\mathcal{S} \oplus \delta = \{\text{append}(\mathcal{P}, \delta_{\mathcal{P}}), \text{append}(\mathcal{R}, \delta_{\mathcal{R}}), \mathcal{H} \cup \delta_{\mathcal{H}}, \mathcal{C} \cup \delta_{\mathcal{C}}\}
\end{equation}
where $\text{append}(\cdot, \cdot)$ preserves sequential ordering for plan steps and reasoning traces, while set union is used for history and correction logs.

In contradistinction to conventional forward-only reasoning architectures, ATCOT facilitates bidirectional state transitions enabling retrospective revision when logical inconsistencies are detected. The framework enforces coherence through a verification function $\phi: \mathcal{S} \rightarrow \{0, 1\}$ that leverages the LLM's inherent reasoning capabilities to detect contradictions between reasoning steps.

\subsection{Adaptive Planning Generation and Dynamic Replanning}

The planning module decomposes the input query into a structured plan $\mathcal{P}_0 = \{p_1, p_2, \ldots, p_n\}$ through maximum a posteriori estimation:
\begin{equation}
\mathcal{P}_0 = \arg\max_{\mathcal{P}} P(\mathcal{P}|q, \mathcal{S}_0, \theta)
\end{equation}

Each plan step $p_i$ is associated with a confidence score $\alpha_i \in [0, 1]$ computed via:
\begin{equation}
\alpha_i = \sigma(W_c \cdot h_i + b_c)
\end{equation}
where $h_i \in \mathbb{R}^d$ represents the contextualized representation of step $p_i$, and $W_c, b_c$ are learned parameters within $\theta$. Each step maintains explicit dependency relations $d_i \subseteq \{p_1, \ldots, p_{i-1}\}$ encoding prerequisite information.

The framework implements a two-level consistency checking mechanism. Local contradictions are detected after each observation via $\psi$, while global consistency is verified at the end of each iteration via $\phi$:
\begin{equation}
\mathcal{P}_{t+1} = \begin{cases}
\mathcal{P}_t & \text{if } \phi(\mathcal{S}_t) = 1 \\
f_{\text{replan}}(\mathcal{P}_t, o_t, \mathcal{R}_t) & \text{if } \phi(\mathcal{S}_t) = 0
\end{cases}
\end{equation}

The replanning function $f_{\text{replan}}$ performs conflict analysis between observations and existing plans, generating modified step sequences that incorporate new information while preserving valid reasoning components.

\subsection{Tool-Augmented Execution with State Tracking}

For each plan step $p_i$, the framework implements tool selection through a learned policy over the available tool set $\mathbb{T}$:
\begin{equation}
\tau^* = \arg\max_{\tau \in \mathbb{T}} P(\tau|p_i, \mathcal{S}_t, \mathcal{R}_t)
\end{equation}

Tool invocation yields observation $o_i = \tau^*(\text{args}_i)$, where $\text{args}_i$ represents the parameterization extracted from the current state. The execution module maintains a directed acyclic graph $G = (V, E)$ with vertices $V$ representing completed reasoning steps and edges $E \subseteq V \times V$ encoding information dependencies.

For each step $p_i$, we may generate multiple candidate observations $\mathcal{O}_i = \{\hat{o}_{i,1}, \ldots, \hat{o}_{i,K}\}$ through parallel or sequential tool invocations. Each candidate receives a reliability score computed through normalized semantic consistency:
\begin{equation}
r_{i,k} = \frac{\exp(\text{sim}(\hat{o}_{i,k}, \mathcal{R}_t) / \kappa)}{\sum_{k'=1}^{K} \exp(\text{sim}(\hat{o}_{i,k'}, \mathcal{R}_t) / \kappa)}
\end{equation}
where $\text{sim}: \mathcal{O} \times \mathcal{R} \rightarrow \mathbb{R}$ measures semantic alignment between observations and existing reasoning, and $\kappa > 0$ is a temperature parameter.

The core execution algorithm proceeds as follows:

\begin{algorithm}
\caption{ATCOT Execution with Adaptive Correction}
\begin{algorithmic}[1]
\REQUIRE Query $q$, Tool set $\mathbb{T}$, Budget $B$
\ENSURE Final answer $a$
\STATE $\mathcal{S} \leftarrow \text{InitializeState}(q)$
\STATE $\mathcal{P} \leftarrow \text{GeneratePlan}(q, \mathcal{S})$
\STATE $\text{corrections} \leftarrow 0$
\WHILE{$\neg\text{Converged}(\mathcal{S}) \land \text{corrections} < B$}
    \FOR{each step $p_i \in \mathcal{P}$ that is pending}
        \STATE $\mathcal{O}_i \leftarrow \text{ExecuteCandidateTools}(p_i, \mathcal{S})$
        \STATE $r \leftarrow \text{SoftmaxConsistencyScores}(\mathcal{O}_i, \mathcal{S}.\mathcal{R})$
        \STATE $o^* \leftarrow \text{SelectBy}(r)$
        \STATE $\mathcal{R}'_t \leftarrow \mathcal{S}.\mathcal{R}$ // Default: no change
        \IF{$\psi(o^*, \mathcal{S}.\mathcal{R}) = 1$}
            \STATE // Local contradiction detected
            \STATE $\mathcal{M} \leftarrow \text{FindMinimalRevisionSet}(\mathcal{S}.\mathcal{R}, o^*)$
            \STATE $\mathcal{R}'_t \leftarrow \text{ReviseReasoning}(\mathcal{S}.\mathcal{R}, \mathcal{M}, o^*)$
            \STATE $\mathcal{S}.\mathcal{C} \leftarrow \text{append}(\text{correction\_log})$
            \STATE $\mathcal{P} \leftarrow \text{AdaptPlan}(\mathcal{P}, \mathcal{R}'_t)$
            \STATE $G \leftarrow \text{PruneAndReconnect}(G, \mathcal{M})$
            \STATE $\text{corrections} \leftarrow \text{corrections} + 1$
        \ELSE
            \STATE $\mathcal{R}'_t \leftarrow \text{append}(o^*)$
        \ENDIF
        \STATE $\mathcal{S}.\mathcal{H} \leftarrow \text{append}(\text{tool\_call\_record})$
        \STATE $\mathcal{S} \leftarrow \text{UpdateState}(\mathcal{S}, o^*, \mathcal{R}'_t)$ // In-place update
    \ENDFOR
    \IF{$\phi(\mathcal{S}) = 0$}
        \STATE // Global consistency check
        \STATE $\mathcal{P} \leftarrow \text{Replan}(\mathcal{P}, \mathcal{S})$
    \ENDIF
\ENDWHILE
\RETURN $\text{GenerateAnswer}(\mathcal{S})$
\end{algorithmic}
\end{algorithm}

\subsection{Adaptive Correction Mechanism with Convergence Guarantees}

The correction mechanism activates upon contradiction detection through the evaluation function:
\begin{equation}
\psi(o_t, \mathcal{R}_t) = \mathbf{1}\left[\max_{r_j \in \mathcal{R}_t} \text{contradict}(o_t, r_j) > \tau_{\text{contra}}\right]
\end{equation}
where $\text{contradict}: \mathcal{O} \times \mathcal{R} \rightarrow [0, 1]$ quantifies logical opposition using natural language inference models, and $\tau_{\text{contra}} \in (0, 1)$ represents a contradiction threshold.

Upon detecting inconsistencies, the correction module performs backward traversal to identify the minimal revision set $\mathcal{M} \subseteq \mathcal{R}_t$ through the optimization:
\begin{equation}
\mathcal{M} = \arg\min_{M \subseteq \mathcal{R}_t} |M| \quad \text{s.t.} \quad \phi(\mathcal{S}_t[\mathcal{R} := (\mathcal{R}_t \setminus M) \cup \text{incorporate}(o_t)]) = 1
\end{equation}

When dependencies $d_i$ reference removed reasoning steps in $\mathcal{M}$, the $\text{AdaptPlan}$ function rebuilds the dependency closure ensuring reachability and maintaining DAG properties. The framework provides practical convergence guarantees through bounded corrections and monotonic improvement in consistency scores.
\begin{table*}[t!]
\centering
\caption{Performance comparison across different models and benchmarks. HotpotQA results are F1 scores (\%). Bold indicates best performance for each model-benchmark combination. All results are averaged over 3 runs with standard deviations.}
\begin{tabular}{ccccccccc}
\hline
\multirow{2}{*}{\textbf{Method}} &
\multicolumn{2}{c}{\textbf{Qwen2.5-7B}} &
\multicolumn{2}{c}{\textbf{Qwen2.5-72B}} &
\multicolumn{2}{c}{\textbf{GPT-4o}} &
\multicolumn{2}{c}{\textbf{Average}} \\
& GSM8K & HotpotQA & GSM8K & HotpotQA & GSM8K & HotpotQA & GSM8K & HotpotQA \\
\hline
\textbf{ATCOT (Ours)} &
\textbf{90.4$\pm$0.5} & \textbf{40.1$\pm$0.8} &
\textbf{95.2$\pm$0.4} & \textbf{52.9$\pm$0.7} &
\textbf{96.6$\pm$0.4} & \textbf{57.4$\pm$0.6} &
\textbf{94.1$\pm$0.4} & \textbf{50.1$\pm$0.7} \\
ReAct~ \cite{yao2023react} &
88.9$\pm$0.6 & 33.5$\pm$1.0 &
93.9$\pm$0.5 & 42.8$\pm$0.8 &
95.6$\pm$0.4 & 52.1$\pm$0.7 &
92.8$\pm$0.5 & 42.8$\pm$0.8 \\
CoT~ \cite{wei2022chain} &
86.1$\pm$0.7 & 34.5$\pm$1.1 &
92.3$\pm$0.6 & 43.3$\pm$0.9 &
93.0$\pm$0.5 & 45.2$\pm$0.8 &
90.5$\pm$0.6 & 41.0$\pm$0.9 \\
Self Refine~ \cite{madaan2023selfrefine} &
86.9$\pm$0.6 & 30.8$\pm$1.0 &
92.8$\pm$0.5 & 38.1$\pm$0.9 &
93.5$\pm$0.5 & 46.3$\pm$0.8 &
91.1$\pm$0.5 & 38.4$\pm$0.9 \\
Tree of Thoughts~ \cite{yao2023tree} &
87.8$\pm$0.7 & 31.9$\pm$1.1 &
93.2$\pm$0.6 & 38.5$\pm$0.9 &
93.8$\pm$0.5 & 47.8$\pm$0.8 &
91.6$\pm$0.6 & 39.4$\pm$0.9 \\
Reflexion~ \cite{shinn2023reflexion} &
87.2$\pm$0.6 & 34.6$\pm$1.0 &
93.1$\pm$0.6 & 40.7$\pm$0.8 &
94.2$\pm$0.5 & 51.1$\pm$0.7 &
91.5$\pm$0.6 & 42.1$\pm$0.8 \\
CRITIC~ \cite{gou2023critic} &
89.6$\pm$0.6 & 37.2$\pm$0.9 &
94.6$\pm$0.5 & 51.8$\pm$0.8 &
96.1$\pm$0.4 & 55.1$\pm$0.7 &
93.4$\pm$0.5 & 48.0$\pm$0.8 \\
\hline
\end{tabular}
\label{tab:main_results}
\end{table*}

\section{Experiments}

\subsection{Experimental Setup}

\noindent\textbf{Datasets}~~We evaluate ATCOT on GSM8K \cite{cobbe2021training} for mathematical reasoning and HotpotQA \cite{yang2018hotpotqa} for multi-hop question answering.

\noindent\textbf{Tools}~~All tool-enabled methods have access to four general-purpose tools: (1) Web Search API for retrieving online information, (2) Calculator for numerical computations, (3) Python Interpreter for complex calculations and data processing, and (4) Wikipedia API for factual knowledge retrieval. While ATCOT supports arbitrary tool integration, we focus on these widely-used tools for fair benchmark comparison.

\noindent\textbf{Models}~~We evaluate our approach on three language models spanning different scales and sources: GPT-4o-2024-08-06 \cite{openai2024gpt4o} (proprietary), Qwen2.5-72B \cite{hui2024qwen25} (open-source large), and Qwen2.5-7B \cite{hui2024qwen25} (open-source compact). This selection enables us to assess the generalizability of ATCOT across model architectures and parameter scales.

\noindent\textbf{Baselines}~~We compare ATCOT against six established reasoning paradigms. Chain-of-Thought (CoT) \cite{wei2022chain} prompts models to generate step-by-step reasoning without external tools, serving as the pure reasoning baseline. ReAct \cite{yao2022react} integrates reasoning with tool actions in a linear fashion, representing a widely-adopted baseline in tool-augmented reasoning. Self-Refine \cite{madaan2023selfrefine} builds upon CoT by iteratively refining reasoning steps to improve accuracy. Tree of Thoughts \cite{yao2023tree} employs a tree-like structure for reasoning, enabling more complex reasoning chains with branching logic. Reflexion \cite{shinn2023reflexion} enhances reasoning by incorporating feedback loops, allowing models to refine their reasoning after each step. CRITIC \cite{gou2023critic} combines reasoning with error detection and correction, focusing on identifying and rectifying mistakes during the reasoning process. All baselines utilize the same tool access as ATCOT when applicable.

\subsection{Main Results}

Table~\ref{tab:main_results} presents comprehensive performance comparisons across three model scales and two reasoning benchmarks. ATCOT achieves substantial and consistent improvements over all baseline methods across all configurations. 

On GPT-4o, ATCOT reaches 96.6\% accuracy on GSM8K, representing a 1.0\% improvement over ReAct (95.6\%) and 3.6\% over CoT (93.0\%). The gains are even more pronounced on HotpotQA (57.4\%), where the adaptive correction mechanism effectively handles conflicting information from multiple search results, yielding improvements of 5.3\% and 12.2\% over ReAct and CoT respectively.

The effectiveness of ATCOT generalizes remarkably well across model scales. For Qwen2.5-72B, we observe 1.3\% and 10.1\% improvements over ReAct on GSM8K and HotpotQA respectively, with the larger gain on HotpotQA highlighting our method's strength in complex multi-hop reasoning. Notably, smaller models demonstrate similar patterns: Qwen2.5-7B shows gains of 1.5\% and 6.6\% over ReAct on the two benchmarks. The consistent improvement pattern suggests that the structured correction mechanism provides robust benefits regardless of model capacity.

Compared to CRITIC, which also employs error detection and correction, ATCOT still achieves notable improvements. On Qwen2.5-7B, we achieve 0.8\% and 2.9\% improvements over CRITIC on GSM8K and HotpotQA respectively. For larger models, the gains remain consistent: 0.6\% and 1.1\% for Qwen2.5-72B, and 0.5\% and 2.3\% for GPT-4o. This demonstrates the superiority of our adaptive approach over static correction mechanisms.

The consistent performance gains across both proprietary (GPT-4o) and open-source (Qwen2.5 series) models, with average improvements of 1.3\% on GSM8K and 7.3\% on HotpotQA over ReAct, indicate that ATCOT's benefits stem from fundamental framework design rather than model-specific characteristics.

\begin{table}[h]
\centering
\caption{Ablation study on GPT-4o showing the contribution of each ATCOT component to overall performance.}
\begin{tabular}{l|cc|c}
\hline
Method & GSM8K & HotpotQA & Average \\
\hline
ATCOT (Full) & \textbf{96.6} & \textbf{57.4} & \textbf{77.0} \\
w/o Correction & 91.1 (-5.5) & 47.8 (-9.6) & 69.5 (-7.5) \\
w/o Planning & 90.8 (-5.8) & 45.7 (-11.7) & 68.3 (-8.7) \\
w/o Consistency & 93.8 (-2.8) & 51.3 (-6.1) & 72.6 (-4.4) \\
w/o Tool History & 94.2 (-2.4) & 53.4 (-4.0) & 73.8 (-3.2) \\
\hline
\end{tabular}
\label{tab:ablation}
\end{table}
\subsection{Ablation Study}

Table~\ref{tab:ablation} presents an ablation study to isolate the contribution of key components in ATCOT. Removing the adaptive correction mechanism results in a 7.5\% average performance drop, with HotpotQA showing larger degradation (9.6\%) due to its need for frequent error recovery when handling conflicting information. The absence of planning generation causes the most severe degradation (8.7\% average), confirming that structured planning provides the foundation for effective reasoning—without it, the system falls back to simple heuristic strategies. Removing global consistency checking leads to a 4.4\% drop as the system fails to detect cross-step logical contradictions, particularly harmful in multi-hop reasoning. The tool invocation history contributes 3.2\% to performance by enabling the system to track dependencies and avoid redundant tool calls. The consistently larger impact on HotpotQA across all ablations (4.0-11.7\% drops) compared to GSM8K (2.4-5.8\%) reflects the greater complexity of multi-hop reasoning requiring all components to work synergistically.



\begin{table}[h]
\centering
\caption{Correction statistics across successful and failed attempts, demonstrating the adaptive nature of our correction mechanism.}
\begin{tabular}{l|cc|cc}
\hline
\multirow{2}{*}{Dataset} & \multicolumn{2}{c|}{Correction Rate (\%)} & \multicolumn{2}{c}{Avg. Corrections} \\
& Success & Failure & Success & Failure \\
\hline
GSM8K & 31.2 & 92.3 & 0.5 & 2.7 \\
HotpotQA & 58.7 & 94.6 & 1.2 & 3.1 \\
\hline
\end{tabular}
\label{tab:correction_stats}
\end{table}

\subsection{Correction Analysis}

Table~\ref{tab:correction_stats} reveals the adaptive nature of our correction mechanism. Successful solutions on GSM8K require minimal corrections (31.2\% correction rate, 0.5 corrections on average), suggesting that most mathematical reasoning chains proceed correctly with only occasional targeted fixes for calculation errors. In contrast, HotpotQA shows higher correction needs even for successful cases (58.7\% rate, 1.2 corrections), reflecting the inherent complexity of reconciling information from multiple sources. Failed attempts exhibit substantially higher correction rates (92.3-94.6\%) and counts (2.7-3.1), demonstrating the system's ability to distinguish between recoverable errors and fundamentally flawed reasoning.

The substantial gap between success and failure patterns validates ATCOT's error detection capabilities. The 61.1\% correction rate difference on GSM8K (92.3\% vs 31.2\%) indicates that mathematical errors often manifest as clear patterns that can be efficiently identified and resolved early. HotpotQA's 35.9\% gap suggests that multi-hop reasoning errors are more nuanced and distributed throughout the reasoning chain. Notably, 92\% of successful corrections occur within the first 2 attempts, while failed cases often exhaust the correction budget (maximum 5 attempts), confirming that our bounded correction mechanism effectively prevents futile loops while maintaining high success rates for solvable problems.

\section{Conclusion}
\label{sec:conclusion}
In this paper, we presented ATCOT, a framework that transforms LLM reasoning from a rigid forward-only process into a flexible, self-correcting system. ATCOT enables models to retroactively revise reasoning trajectories when tool feedback reveals errors, achieving consistent improvements across GSM8K and HotpotQA benchmarks on multiple model scales. Our ablation study confirms that adaptive correction and planning generation are critical, while correction analysis validates efficient targeted fixes without futile loops. Within the tool-augmented reasoning paradigm, ATCOT achieves the best performance among evaluated baselines, representing a significant step toward robust AI reasoning systems for complex real-world tasks.



% Below is an example of how to insert images. Delete the ``\vspace'' line,
% uncomment the preceding line ``\centerline...'' and replace ``imageX.ps''
% with a suitable PostScript file name.
% -------------------------------------------------------------------------


\bibliographystyle{IEEEbib}
\bibliography{strings,refs}

\end{document}
