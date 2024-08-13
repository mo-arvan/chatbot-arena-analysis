# Chatbot Arena Statistical Analysis

## Abstract

Evaluating LLMs, particularly within the Chatbot Arena leaderboard, poses substantial challenges due to the variability in model outputs and the dependence on human judgment. In this post, I employ statistical analysis to assess the reproducibility of leaderboard rankings, with a focus on comparing GPT-4 and Claude-v1. My findings indicate that although differences between models may seem statistically significant, the reliability of these results is often compromised by insufficient sample sizes. I recommend more rigorous evaluation methods and present practical approach for model selection in conversational AI.

## Challenges in Evaluating LLMs: A Statistical Analysis of Chatbot Arena Leaderboard

Progress in machine learning research is often driven by benchmarks, which provide objective comparisons between different models and algorithms. One prominent benchmark in the field of conversational AI is the Chatbot Arena, a leaderboard that ranks chatbot models based on ratings from human evaluators. However, evaluating generated text remains a complex challenge. In this post, I will discuss these challenges and present a statistical approach to investigate the reproducibility of the Chatbot Arena leaderboard.

Evaluating generated text is particularly difficult due to several factors. First, the model's output can vary because text generation involves sampling from a distribution. Second, there may be multiple correct answers to a given prompt. Lastly, while the goal is objective evaluation, human judgment still plays a significant role, whether through annotated data for automatic benchmarks or direct human comparisons of model performance.

Despite these challenges, benchmarks are essential for comparing models. The next question is: how can we increase trust in these benchmarks? One of the most critical properties of a benchmark is reproducibility. Ideally, repeating the same experiment should yield similar results. For instance, if I conduct an evaluation and find that model A outperforms model B, another researcher should observe the same result under identical conditions. However, recent research suggests that reproducibility is not guaranteed; specific design decisions and experimental aspects can lead to high variability in results.

Consider these two reproducibility studies—[Van Miltenburg et al.](https://aclanthology.org/2023.humeval-1.7.pdf) and [Arvan et al.](https://aclanthology.org/2023.humeval-1.8.pdf)—which evaluated an ML pipeline (or see Chapter 4 of my [dissertation](https://mo-arvan.github.io/assets/pdf/papers/Mohammad_Arvan_Thesis.pdf)). Despite following the same process, the ranking flipped between the two evaluations. This phenomenon is not unique; over 10 other reproducibility studies from the ReproHum project—an initiative for testing the reproducibility of human evaluation in ML and NLP—have found similar issues ([Belz et al., 2023](https://aclanthology.org/2023.humeval-1.4.pdf) and [Belz et al., 2024](https://aclanthology.org/2024.humeval-1.9.pdf)).

Many factors contribute to the reproducibility (or lack there of) of evaluations. Although the ReproHum project is ongoing, evidence suggests that statistical analysis can help understand variability in human evaluations. Significance testing, a standard practice in many scientific fields is a common tool to determine if the difference between two groups is due to random chance or a real difference. Unfortunately, in Computer Science, significance testing is underutilized.

A common test is the t-test, which can evaluate whether the difference in mean preference scores (wins) between two models is statistically significant. However, significance testing also involves other important aspects: effect size and power analysis. Effect size measures the magnitude of the difference between two groups, while power analysis determines the sample size needed to detect a difference. For instance, a power of 0.8 means there's an 80% chance of detecting a difference if one exists. The significance level (alpha) is typically set at 0.05, indicating a 5% chance of detecting a difference that doesn't exist. As a reference, Cohen suggests that a small effect size is 0.2, medium is 0.5, and large is 0.8.

While Chatbot Arena provides a leaderboard, pairwise comparisons of models more closely reflect practical model selection. For example, if I want to compare a production model (Model A) with a new model (Model B), I need to determine whether the performance gains justify transitioning to Model B. If the gains are marginal, I might stick with Model A. This is essentially A/B testing in the context of model evaluation.

To apply this analysis to the Chatbot Arena leaderboard, I utilized a [dataset](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) released by LMSYS, the organization behind Chatbot Arena. Although the dataset is almost a year old, it remains a valuable resource. I plan to request an updated dataset, but for now, I will use the existing data to demonstrate the analysis.

Focusing on the first- and second-place models—GPT-4 and Claude-v1—there are 321 pairwise comparisons between them. To detect an effect size of 0.8, we need 26 comparisons; for effect sizes of 0.5 and 0.2, we need 64 and 394 comparisons, respectively. 
Conceptually, this means the closer the performance of two models, the larger the sample size needed to detect a difference.

In the current setting, if the empirical effect size is 0.5 or greater, the sample size is sufficient. However, if the effect size is less than 0.5, the significance test is at higher risk of false positives.

Out of the 321 comparisons, GPT-4 won 37% of the time, Claude-v1 won 30%, and 33% were ties. We calculated a score array for each model: 1 point for a win, -1 for a loss, and 0 for a tie. Ideally, each comparison would involve multiple human evaluators, but in this setup, only one evaluator was used per comparison—one of the limitations of the current Chatbot Arena structure.

A t-test on the scores of GPT-4 and Claude-v1 yielded a p-value of 0.0265, a test statistic of 2.2, and an empirical effect size of 0.18. While the difference between the two models is statistically significant, the effect size is less than 0.2, indicating the test may be underpowered. Thus, we cannot confidently conclude that GPT-4 is superior to Claude-v1, and drawing strong conclusions may lead to future reproducibility issues.

This analysis can be extended to other models on the leaderboard. For instance, we could compare models within the same API cost tier, providing a more nuanced comparison. Moving beyond a single score to a more comprehensive analysis is a step in the right direction. Rigorous evaluations are crucial in the landscape of LLMs. 

I look forward to obtaining the updated dataset from LMSYS to conduct a more thorough analysis. With the additional data and inclusion of the latest models, some of the concerns raised in this post may be addressed. I welcome your thoughts on this analysis and any suggestions for future work. The source code for this analysis is available on my [GitHub repository](https://github.com/mo-arvan/chatbot-arena-analysis).

## Running the Code

You need to have Docker installed on your machine to run the code. The following commands will build the Docker image and run the python script.

```bash
docker build -t chatbot-arena-analysis -f Dockerfile .
```
Replace `YOUR_HF_API_KEY` your Hugging Face API key (`HF_TOKEN`) and run the following command.

```bash
docker run -it chatbot-arena-analysis -e HF_TOKEN=YOUR_HF_API_KEY python chatbot_arena_statistical_analysis.py
```