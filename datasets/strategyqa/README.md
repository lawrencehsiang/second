A Question Answering Benchmark with Implicit Reasoning Strategies

The StrategyQA dataset was created through a crowdsourcing pipeline for eliciting creative and diverse yes/no questions that require implicit reasoning steps. To solve questions in StrategyQA, the reasoning steps should be inferred using a strategy. To guide and evaluate the question answering process, each example in StrategyQA was annotated with a decomposition into reasoning steps for answering it, and Wikipedia paragraphs that provide evidence for the answer to each step.

Illustrated in the figure below: Questions in StrategyQA (Q1) require implicit reasoning, in contrast to multi-step questions that explicitly specify the reasoning process (Q2). Each training example contains a question (Q1), yes/no answer (A), decomposition (D), and evidence paragraphs (E).

[strategyqa_test](https://huggingface.co/datasets/voidful/StrategyQA/resolve/main/strategyqa_test.json)  
[strategyqa_train](https://huggingface.co/datasets/voidful/StrategyQA/blob/main/strategyqa_train.json)  
[strategyqa_train_filtered](https://huggingface.co/datasets/voidful/StrategyQA/blob/main/strategyqa_train_filtered.json)  
[strategyqa_train_paragraphs](https://huggingface.co/datasets/voidful/StrategyQA/blob/main/strategyqa_train_paragraphs.json)  


Paper

Title: Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies

Authors: Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, Jonathan Berant

Transactions of the Association for Computational Linguistics (TACL), 2021

Citation:
```
@article{geva2021strategyqa,
  title = {{Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies}},
  author = {Geva, Mor and Khashabi, Daniel and Segal, Elad and Khot, Tushar and Roth, Dan and Berant, Jonathan},
  journal = {Transactions of the Association for Computational Linguistics (TACL)},
  year = {2021},
}
```