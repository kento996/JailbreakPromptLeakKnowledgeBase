# Jailbreaking&PromptLeakKnowledgeBase
这里我们会长期更新关于Jailbreaking&PromptLeaking的相关资料（比赛、数据集、研究方法）
## 相关的LLM比赛数据

### TDC2023

是红队比赛数据，用于开发和测试的越狱指令各 50 个

[Dev Data](https://github.com/centerforaisafety/tdc2023-starter-kit/blob/main/red_teaming/data/dev/behaviors.json)
![JailbreakPromptLeakKnowledgeBase/pictures/Devdata.png](pictures/Devdata.png)

[Test Data](https://github.com/centerforaisafety/tdc2023-starter-kit/blob/main/red_teaming/data/test/behaviors.json)

![JailbreakPromptLeakKnowledgeBase/pictures/Testdata.png](pictures/Testdata.png)

## 数据集

| 数据集            | 越狱模板/指令数量   | 模板来源                                         |
| ----------------- | ------------------- | ------------------------------------------------ |
| GPTFuzzer         | 102                 | 遗传算法                                         |
| DAN(CCS24)        | 1405                | 平台收集                                         |
| UltraSafety       | 830 (需去重)        | AutoDAN                                          |
| WildJailbreak     | 5.7K                | WildTeaming (an automatic red-teaming framework) |
| AdvBench          | 520                 |                                                  |
| MaliciousInstruct | 100                 |                                                  |
| HarmBench         | 400 + 110（多模态） |                                                  |
| JailbreakBench    | 100                 |                                                  |
| StrongReject      | 313                 |                                                  |

### GPTFuzzer

数据地址：https://gitcode.com/gh_mirrors/gp/GPTFuzz/blob/master/datasets/
![pictures/GPTFuzzer.png](pictures/GPTFuzzer.png)
对于通过查询 Vicuna-7B、ChatGPT 和 Llama-2-7B-chat 得到的响应，它们存储在 datasets/responses 中，标记的响应在 datasets/responses_labeled 中。
此外，还有数据集对应的论文研究：https://arxiv.org/pdf/2309.10253

### Do Anything Now

数据地址：https://github.com/verazuo/jailbreak_llms/blob/main/data/

> Statistics of our data source. (Adv) UA refers to (adversarial) user accounts.

| Platform  | Source                     | # Posts     | # UA      | # Adv UA | # Prompts  | # Jailbreaks | Prompt Time Range   |
| --------- | -------------------------- | ----------- | --------- | -------- | ---------- | ------------ | ------------------- |
| Reddit    | r/ChatGPT                  | 163549      | 147       | 147      | 176        | 176          | 2023.02-2023.11     |
| Reddit    | r/ChatGPTPromptGenius      | 3536        | 305       | 21       | 654        | 24           | 2022.12-2023.11     |
| Reddit    | r/ChatGPTJailbreak         | 1602        | 183       | 183      | 225        | 225          | 2023.02-2023.11     |
| Discord   | ChatGPT                    | 609         | 259       | 106      | 544        | 214          | 2023.02-2023.12     |
| Discord   | ChatGPT Prompt Engineering | 321         | 96        | 37       | 278        | 67           | 2022.12-2023.12     |
| Discord   | Spreadsheet Warriors       | 71          | 3         | 3        | 61         | 61           | 2022.12-2023.09     |
| Discord   | AI Prompt Sharing          | 25          | 19        | 13       | 24         | 17           | 2023.03-2023.04     |
| Discord   | LLM Promptwriting          | 184         | 64        | 41       | 167        | 78           | 2023.03-2023.12     |
| Discord   | BreakGPT                   | 36          | 10        | 10       | 32         | 32           | 2023.04-2023.09     |
| Website   | AIPRM                      | -           | 2777      | 23       | 3930       | 25           | 2023.01-2023.06     |
| Website   | FlowGPT                    | -           | 3505      | 254      | 8754       | 405          | 2022.12-2023.12     |
| Website   | JailbreakChat              | -           | -         | -        | 79         | 79           | 2023.02-2023.05     |
| Dataset   | AwesomeChatGPTPrompts      | -           | -         | -        | 166        | 2            | -                   |
| Dataset   | OCR-Prompts                | -           | -         | -        | 50         | 0            | -                   |
| **Total** |                            | **169,933** | **7,308** | **803**  | **15,140** | **1,405**    | **2022.12-2023.12** |

ACM CCS 2024 :[[2308.03825\] "Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models](https://arxiv.org/abs/2308.03825)

### UltraSafety

数据地址：[openbmb/UltraSafety · Datasets at Hugging Face](https://huggingface.co/datasets/openbmb/UltraSafety)

![pictures/UltraSafety.png](pictures/UltraSafety.png)

UltraSafety 从 AdvBench 和 MaliciousInstruct 派生出 1000 条安全种子指令，并使用 Self-Directed 引导另外 2000 条指令。对来自 AutoDAN 的越狱提示进行手动筛选，最终选择了 830 个高质量的越狱提示。UltraSafety 总共包含 3,000 条有害指令，每条指令都附有相关的越狱提示。每条有害指令对应于由不同安全级别的模型生成的完成，并附有 GPT4 分配的评级，评级为 1 表示无害，评级为 0 表示有害。

##### Model Sampling

- Commercial Models: GPT-4, GPT-3.5 Turbo
- LLaMA family:
  1. LLaMA-2-7B-chat, LLaMA-2-13B-chat, LLaMA-2-70B-chat
  2. UltraLM-13B, UltraLM-65B
  3. WizardLM-7B, WizardLM-13B, WizardLM-70B
  4. Vicuna-33B
- Non-LLaMA series:
  1. Mistral-7B-Instruct-v0.2, Mixtral-8x7B-Instruct-v0.1
  2. zephyr-7b-beta
  3. StarChat-Beta

### WildJailbreak

数据地址：[allenai/wildjailbreak · Datasets at Hugging Face](https://huggingface.co/datasets/allenai/wildjailbreak)

WildJailbreak 是一个开源合成安全训练数据集，具有 262K *原*版（直接有害请求）和*对抗性*（复杂对抗性越狱）提示响应对。

**示例提示：**

![绘图](https://huggingface.co/datasets/allenai/wildjailbreak/media/main/figures/data_example_four_types.png)

### AdvBench

数据地址：[refusal_direction/dataset/raw/advbench.csv at main · andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction/blob/main/dataset/raw/advbench.csv)

论文：[2406.11717\] Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)

展示了13 个流行的开源聊天模型，最大参数大小为 72B

![pictures/AdvBench.png](pictures/AdvBench.png)

### MaliciousInstruct

数据地址：[Jailbreak_LLM/data/MaliciousInstruct.txt at main · Princeton-SysML/Jailbreak_LLM](https://github.com/Princeton-SysML/Jailbreak_LLM/blob/main/data/MaliciousInstruct.txt)

论文：[pdf](https://openreview.net/pdf?id=r42tSSCHPh)

数据集中包含 10 个不同恶意意图的 100 条恶意指令，并且评估了 11 个模型，分别是：[Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/)（7B、13B 和 33B）、[MPT](https://www.mosaicml.com/blog/mpt-7b)（7B 和 30B）、[Falcon](https://falconllm.tii.ae/)（7B 和 40B）和 [LLaMA2](https://ai.meta.com/llama/)（7B、13B、7B-chat 和 13B-chat）。

![pictures/MaliciousInstruct.png](pictures/MaliciousInstruct.png)

### HarmBench

数据地址：[HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv at main · centerforaisafety/HarmBench](https://github.com/centerforaisafety/HarmBench/blob/main/data/behavior_datasets/harmbench_behaviors_text_all.csv)

![pictures/HarmBench1.png](pictures/HarmBench1.png)

![pictures/HarmBench2.png](pictures/HarmBench2.png)

### JailbreakBench

数据地址：[JailbreakBench/JBB-Behaviors · Datasets at Hugging Face](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors)

55 个原创越狱指令和一些先前工作的指令

![pictures/JailbreakBench.png](pictures/JailbreakBench.png)

### StrongReject

数据地址：[strongreject/strongreject_dataset/strongreject_dataset.csv at main · alexandrasouly/strongreject](https://github.com/alexandrasouly/strongreject/blob/main/strongreject_dataset/strongreject_dataset.csv)

200 + 原创越狱指令

![pictures/StrongReject.png](pictures/StrongReject.png)



## 相关论文

### DeepInception: Hypnotize Large Language Model to Be Jailbreaker（用深度催眠诱导LLM「越狱」）

链接：[2311.03191 (arxiv.org)](https://arxiv.org/pdf/2311.03191)

摘要：Despite remarkable success in various applications, large language models (LLMs) are
 vulnerable to adversarial jailbreaks that make the safety guardrails void. However, previous
 studies for jailbreaks usually resort to brute-force optimization or extrapolations of a high
 computation cost, which might not be practical or effective. In this paper, inspired by
 the Milgram experiment w.r.t. the authority power for inciting harmfulness, we disclose
 a lightweight method, termed as DeepInception, which can hypnotize an LLM to be a
 jailbreaker. Specifically, DeepInception leverages the personification ability of LLM to
 construct a virtual, nested scene to jailbreak, which realizes an adaptive way to escape the
 usage control in a normal scenario. Empirically, DeepInception can achieve competitive
 jailbreak success rates with previous counterparts and realize a continuous jailbreak in
 subsequent interactions, which reveals the critical weakness of self-losing on both open
source and closed-source LLMs like Falcon, Vicuna-v1.5, Llama-2, GPT-3.5, and GPT-4。（翻译：尽管在各种应用程序中取得了显著的成功，但大型语言模型 （LLM） 容易受到对抗性越狱的影响，从而使安全护栏失效。然而，以前的越狱研究通常采用暴力优化或高计算成本的外推，这可能不切实际或无效。在本文中，受 Milgram 实验的启发，我们揭示了一种称为 DeepInception 的轻量级方法，它可以催眠 LLM 成为越狱者。具体来说，DeepInception 利用 LLM 的拟人化能力构建了一个虚拟的、嵌套的场景来越狱，这实现了在正常场景中逃避使用控制的自适应方式。从经验上看，DeepInception 可以实现与以前的同类产品相比有竞争力的越狱成功率，并在后续交互中实现连续越狱，这揭示了 Falcon、Vicuna-v1.5、Llama-2、GPT-3.5 和 GPT-4 等开源和闭源 LLM 存在自我丢失的关键弱点）

代码：https://github.com/tmlr-group/DeepInception



### A Comprehensive Study of Jailbreak Attack versus Defense for Large Language Models（大型语言模型的越狱攻击与防御的综合研究）

链接：[2402.13457 (arxiv.org)](https://arxiv.org/pdf/2402.13457)

摘要：Large Language Models (LLMS) have increasingly become central to generating content with potential societal impacts. Notably, these models have demonstrated capabilities for generating content that could be deemed harmful. To mitigate these risks, researchers have adopted safety training techniques to align model outputs with societal values to curb the generation of malicious content. However, the phenomenon of "jailbreaking", where carefully crafted prompts elicit harmful responses from models, persists as a significant challenge. This research conducts a comprehensive analysis of existing studies on jailbreaking LLMs and their defense techniques. We meticulously investigate nine attack techniques and seven defense techniques applied across three distinct language models: Vicuna, LLama, and GPT-3.5 Turbo. We aim to evaluate the effectiveness of these attack and defense techniques. Our findings reveal that existing white-box attacks underperform compared to universal techniques and that including special tokens in the input significantly affects the likelihood of successful attacks. This research highlights the need to concentrate on the security facets of LLMs. Additionally, we contribute to the field by releasing our datasets and testing framework, aiming to foster further research into LLM security. We believe these contributions will facilitate the exploration of security measures within this domain.（翻译：大型语言模型 （LLMS） 越来越成为生成具有潜在社会影响的内容的核心。值得注意的是，这些模型已经展示了生成可能被视为有害的内容的能力。为了降低这些风险，研究人员采用了安全培训技术，使模型输出与社会价值观保持一致，以遏制恶意内容的生成。然而，“越狱”现象，即精心设计的提示会引起模型的有害响应，仍然是一个重大挑战。本研究对关于越狱 LLM 及其防御技术的现有研究进行了全面分析。我们仔细研究了应用于三种不同语言模型的九种攻击技术和七种防御技术：Vicuna、LLama 和 GPT-3.5 Turbo。我们的目标是评估这些攻击和防御技术的有效性。我们的研究结果表明，与通用技术相比，现有的白盒攻击表现不佳，并且在输入中包含特殊标记会显著影响攻击成功的可能性。这项研究强调了专注于 LLM 安全方面的必要性。此外，我们通过发布我们的数据集和测试框架为该领域做出贡献，旨在促进对 LLM 安全性的进一步研究。我们相信这些贡献将有助于探索该领域内的安全措施。）

代码：[GitHub - ltroin/llm_attack_defense_arena](https://github.com/ltroin/llm_attack_defense_arena)



### AutoDefense: Multi-Agent LLM Defense against Jailbreak Attacks（AutoDefense：针对越狱攻击的多代理 LLM 防御）

链接：[2403.04783 (arxiv.org)](https://arxiv.org/pdf/2403.04783)

摘要：Despite extensive pre-training and fine-tuning in moral alignment to prevent generating harmful information at user request, large language models (LLMs) remain vulnerable to jailbreak attacks. In this paper, we propose AutoDefense, a response-filtering based multi-agent defense framework that filters harmful responses from LLMs. This framework assigns different roles to LLM agents and employs them to complete the defense task collaboratively. The division in tasks enhances the overall instruction-following of LLMs and enables the integration of other defense components as tools. AutoDefense can adapt to various sizes and kinds of open-source LLMs that serve as agents. Through conducting extensive experiments on a large scale of harmful and safe prompts, we validate the effectiveness of the proposed AutoDefense in improving the robustness against jailbreak attacks, while maintaining the performance at normal user request. （尽管在道德一致性方面进行了广泛的预培训和微调，以防止应用户请求生成有害信息，但大型语言模型 （LLM） 仍然容易受到越狱攻击。在本文中，我们提出了 AutoDefense，这是一种基于响应过滤的多代理防御框架，用于过滤来自 LLM 的有害响应。该框架为 LLM 代理分配不同的角色，并使用它们来协作完成防御任务。任务划分增强了 LLM 的整体指令遵循，并支持将其他防御组件集成为工具。AutoDefense 可以适应各种大小和种类的开源 LLM，这些 LLM 用作代理。通过对大规模有害和安全的提示进行广泛实验，我们验证了所提出的 AutoDefense 在提高对越狱攻击的鲁棒性方面的有效性，同时保持了正常用户请求的性能。）

代码：[GitHub - XHMY/AutoDefense: AutoDefense: Multi-Agent LLM Defense against Jailbreak Attacks](https://github.com/XHMY/AutoDefense)



### Don't Say No: Jailbreaking LLM by Suppressing Refusal

链接：[2404.16369 (arxiv.org)](https://arxiv.org/pdf/2404.16369)

摘要：Ensuring the safety alignment of Large Language Models (LLMs) is crucial to generating responses consistent with human values. Despite their ability to recognize and avoid harmful queries, LLMs are vulnerable to jailbreaking attacks, where carefully crafted prompts seduce them to produce toxic content. One category of jailbreak attacks is reformulating the task as an optimization by eliciting the LLM to generate affirmative responses. However, such optimization objective has its own limitations, such as the restriction on the predefined objectionable behaviors, leading to suboptimal attack performance. In this study, we first uncover the reason why vanilla target loss is not optimal, then we explore and enhance the loss objective and introduce the DSN (Don't Say No) attack, which achieves successful attack by suppressing refusal. Another challenge in studying jailbreak attacks is the evaluation, as it is difficult to directly and accurately assess the harmfulness of the responses. The existing evaluation such as refusal keyword matching reveals numerous false positive and false negative instances. To overcome this challenge, we propose an Ensemble Evaluation pipeline that novelly incorporates Natural Language Inference (NLI) contradiction assessment and two external LLM evaluators. Extensive experiments demonstrate the potential of the DSN and effectiveness of Ensemble Evaluation compared to baseline methods.（确保大型语言模型 （LLM） 的安全一致性对于生成符合人类价值观的响应至关重要。尽管 LLM 能够识别和避免有害查询，但它们很容易受到越狱攻击，其中精心设计的提示会引诱它们产生有害内容。一类越狱攻击是通过引发 LLM 生成肯定响应，将任务重新表述为优化。但是，此类优化目标有其自身的局限性，例如对预定义的不良行为的限制，从而导致攻击性能欠佳。在本研究中，我们首先揭示了原版目标损失不是最优的原因，然后我们探索并增强了损失目标，并引入了 DSN（Don't Say No）攻击，通过抑制拒绝来实现攻击成功。研究越狱攻击的另一个挑战是评估，因为很难直接准确地评估反应的危害性。现有的评估（如拒绝关键词匹配）揭示了许多误报和漏报实例。为了克服这一挑战，我们提出了一个集成评估管道，它新颖地结合了自然语言推理 （NLI） 矛盾评估和两个外部 LLM 评估器。与基线方法相比，广泛的实验证明了 DSN 的潜力和集成评估的有效性。）

代码：[GitHub - DSN-2024/DSN: DSN jailbreak Attack & Evaluation Ensemble](https://github.com/DSN-2024/DSN)



### Play Guessing Game with LLM: Indirect Jailbreak Attack with Implicit Clues

链接：[2402.09091 (arxiv.org)](https://arxiv.org/pdf/2402.09091)

摘要：With the development of LLMs, the security threats of LLMs are getting more and more attention. Numerous jailbreak attacks have been proposed to assess the security defense of LLMs. Current jailbreak attacks primarily utilize scenario camouflage techniques. However their explicitly mention of malicious intent will be easily recognized and defended by LLMs. In this paper, we propose an indirect jailbreak attack approach, Puzzler, which can bypass the LLM's defense strategy and obtain malicious response by implicitly providing LLMs with some clues about the original malicious query. In addition, inspired by the wisdom of "When unable to attack, defend" from Sun Tzu's Art of War, we adopt a defensive stance to gather clues about the original malicious query through LLMs. Extensive experimental results show that Puzzler achieves a query success rate of 96.6% on closed-source LLMs, which is 57.9%-82.7% higher than baselines. Furthermore, when tested against the state-of-the-art jailbreak detection approaches, Puzzler proves to be more effective at evading detection compared to baselines.（随着 LLM 的发展，LLM 的安全威胁越来越受到关注。已经提出了许多越狱攻击来评估 LLM 的安全防御。当前的越狱攻击主要使用场景伪装技术。但是，他们明确提及的恶意将很容易被 LLM 识别和辩护。在本文中，我们提出了一种间接越狱攻击方法，即 Puzzler，它可以通过隐式地向 LLM 提供一些关于原始恶意查询的线索来绕过 LLM 的防御策略并获得恶意响应。此外，受到孙子兵法中“无法攻击，防御”智慧的启发，我们采取防御姿态，通过 LLM 收集原始恶意查询的线索。广泛的实验结果表明，Puzzler 在闭源 LLM 上实现了 96.6% 的查询成功率，比基线高出 57.9%-82.7%。此外，当与最先进的越狱检测方法进行测试时，Puzzler 被证明比基线更有效地逃避检测。）



### Jailbroken: How Does LLM Safety Training Fail

链接：[fd6613131889a4b656206c50a8bd7790-Paper-Conference.pdf (neurips.cc)](https://proceedings.neurips.cc/paper_files/paper/2023/file/fd6613131889a4b656206c50a8bd7790-Paper-Conference.pdf)

摘要： Large language models trained for safety and harmlessness remain susceptible to
 adversarial misuse, as evidenced by the prevalence of “jailbreak” attacks on early
 releases of ChatGPT that elicit undesired behavior. Going beyond recognition of
 the issue, we investigate why such attacks succeed and how they can be created.
 We hypothesize two failure modes of safety training: competing objectives and
 mismatched generalization. Competing objectives arise when a model’s capabilities
 and safety goals conflict, while mismatched generalization occurs when safety
 training fails to generalize to a domain for which capabilities exist. We use these
 failure modes to guide jailbreak design and then evaluate state-of-the-art models,
 including OpenAI’s GPT-4 and Anthropic’s Claude v1.3, against both existing and
 newly designed attacks. We find that vulnerabilities persist despite the extensive
 red-teaming and safety-training efforts behind these models. Notably, new attacks
 utilizing our failure modes succeed on every prompt in a collection of unsafe
 requests from the models’ red-teaming evaluation sets and outperform existing ad
 hoc jailbreaks. Our analysis emphasizes the need for safety-capability parity—that
 safety mechanisms should be as sophisticated as the underlying model—and argues
 against the idea that scaling alone can resolve these safety failure modes.（经过安全和无害训练的大型语言模型仍然容易受到对抗性滥用，对 ChatGPT 早期版本的普遍“越狱”攻击就证明了这一点，这些攻击会引发不良行为。除了认识到问题之外，我们还调查了此类攻击成功的原因以及如何创建它们。我们假设安全培训的两种失败模式：竞争目标和不匹配的泛化。当模型的能力和安全目标发生冲突时，就会出现竞争目标，而当安全训练无法推广到存在能力的领域时，就会出现不匹配的泛化。我们使用这些失败模式来指导越狱设计，然后评估最先进的模型，包括 OpenAI 的 GPT-4 和 Anthropic 的 Claude v1.3，以对抗现有和新设计的攻击。我们发现，尽管这些模型背后进行了广泛的红队和安全培训工作，但漏洞仍然存在。值得注意的是，利用我们的故障模式的新攻击在来自模型的红队评估集的不安全请求集合中的每个提示上都成功，并且性能优于现有的临时越狱。我们的分析强调了安全能力对等的必要性，即安全机制应该与底层模型一样复杂，并反对仅靠扩展就可以解决这些安全故障模式的观点。）



### Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack

链接：[2404.01833 (arxiv.org)](https://arxiv.org/pdf/2404.01833)

摘要：Large Language Models (LLMs) have risen significantly in popularity and are increasingly being adopted across multiple applications. These LLMs are heavily aligned to resist engaging in illegal or unethical topics as a means to avoid contributing to responsible AI harms. However, a recent line of attacks, known as jailbreaks, seek to overcome this alignment. Intuitively, jailbreak attacks aim to narrow the gap between what the model can do and what it is willing to do. In this paper, we introduce a novel jailbreak attack called Crescendo. Unlike existing jailbreak methods, Crescendo is a simple multi-turn jailbreak that interacts with the model in a seemingly benign manner. It begins with a general prompt or question about the task at hand and then gradually escalates the dialogue by referencing the model's replies progressively leading to a successful jailbreak. We evaluate Crescendo on various public systems, including ChatGPT, Gemini Pro, Gemini-Ultra, LlaMA-2 70b and LlaMA-3 70b Chat, and Anthropic Chat. Our results demonstrate the strong efficacy of Crescendo, with it achieving high attack success rates across all evaluated models and tasks. Furthermore, we present Crescendomation, a tool that automates the Crescendo attack and demonstrate its efficacy against state-of-the-art models through our evaluations. Crescendomation surpasses other state-of-the-art jailbreaking techniques on the AdvBench subset dataset, achieving 29-61% higher performance on GPT-4 and 49-71% on Gemini-Pro. Finally, we also demonstrate Crescendo's ability to jailbreak multimodal models.（大型语言模型 （LLM） 已显著普及，并越来越多地在多个应用程序中采用。这些 LLM 高度一致，抵制参与非法或不道德的话题，以避免造成负责任的 AI 危害。然而，最近的一系列攻击（称为越狱）试图克服这种一致性。直观地说，越狱攻击旨在缩小模型可以做的事情和它愿意做的事情之间的差距。在本文中，我们介绍了一种名为 Crescendo 的新型越狱攻击。与现有的越狱方法不同，Crescendo 是一个简单的多轮越狱，它以看似良性的方式与模型交互。它从关于手头任务的一般提示或问题开始，然后通过逐步引用模型的回复逐渐升级对话，从而成功越狱。我们在各种公共系统上评估了 Crescendo，包括 ChatGPT、Gemini Pro、Gemini-Ultra、LlaMA-2 70b 和 LlaMA-3 70b Chat，以及 Anthropic Chat。我们的结果证明了 Crescendo 的强大功效，它在所有评估的模型和任务中都实现了很高的攻击成功率。此外，我们还介绍了 Crescendomation，这是一种自动化 Crescendo 攻击的工具，并通过我们的评估证明了它对最先进模型的疗效。Crescendomation 在 AdvBench 子集数据集上超越了其他最先进的越狱技术，在 GPT-29 上实现了 61-4% 的性能提升，在 Gemini-Pro 上实现了 49-71% 的性能提升。最后，我们还演示了 Crescendo 越狱多模态模型的能力。）



### h4rm3l: A Dynamic Benchmark of Composable Jailbreak Attacks for LLM Safety Assessment

链接：[2408.04811 (arxiv.org)](https://arxiv.org/pdf/2408.04811)

摘要：The safety of Large Language Models (LLMs) remains a critical concern due to a lack of adequate benchmarks for systematically evaluating their ability to resist generating harmful content. Previous efforts towards automated red teaming involve static or templated sets of illicit requests and adversarial prompts which have limited utility given jailbreak attacks' evolving and composable nature. We propose a novel dynamic benchmark of composable jailbreak attacks to move beyond static datasets and taxonomies of attacks and harms. Our approach consists of three components collectively called h4rm3l: (1) a domain-specific language that formally expresses jailbreak attacks as compositions of parameterized prompt transformation primitives, (2) bandit-based few-shot program synthesis algorithms that generate novel attacks optimized to penetrate the safety filters of a target black box LLM, and (3) open-source automated red-teaming software employing the previous two components. We use h4rm3l to generate a dataset of 2656 successful novel jailbreak attacks targeting 6 state-of-the-art (SOTA) open-source and proprietary LLMs. Several of our synthesized attacks are more effective than previously reported ones, with Attack Success Rates exceeding 90% on SOTA closed language models such as claude-3-haiku and GPT4-o. By generating datasets of jailbreak attacks in a unified formal representation, h4rm3l enables reproducible benchmarking and automated red-teaming, contributes to understanding LLM safety limitations, and supports the development of robust defenses in an increasingly LLM-integrated world.（大型语言模型 （LLM） 的安全性仍然是一个关键问题，因为缺乏足够的基准来系统地评估它们抵抗生成有害内容的能力。以前实现自动红队的努力涉及静态或模板化的非法请求集和对抗性提示，鉴于越狱攻击不断发展和可组合的性质，这些请求的实用性有限。我们提出了一种新的可组合越狱攻击动态基准，以超越静态数据集以及攻击和危害的分类法。我们的方法由三个组件组成，统称为 h4rm3l：（1） 一种领域特定语言，正式将越狱攻击表示为参数化提示转换原语的组合，（2） 基于老虎机的少数样本程序合成算法，生成经过优化的新型攻击，以穿透目标黑盒 LLM 的安全过滤器，以及 （3） 采用前两个组件的开源自动红队软件。我们使用 h4rm3l 生成了一个包含 2656 次成功的新型越狱攻击的数据集，这些攻击针对 6 个最先进的 （SOTA） 开源和专有 LLM。我们的几种合成攻击比以前报告的攻击更有效，在 SOTA 封闭语言模型（如 claude-3-haiku 和 GPT4-o）上的攻击成功率超过 90%。通过以统一的正式表示形式生成越狱攻击数据集，h4rm3l 实现了可重复的基准测试和自动红队划分，有助于理解 LLM 安全限制，并支持在日益 LLM 集成的世界中开发强大的防御措施。）

代码：[h4rm3l: A Domain-Specific Language, Jailbreak Attack Synthesizer and Dynamic LLM Redteaming Toolkit (mdoumbouya.github.io)](https://arxiv.org/pdf/2408.04811)



### JailbreakHunter: A Visual Analytics Approach for Jailbreak Prompts Discovery from Large-Scale Human-LLM Conversational Datasets

链接：[arxiv.org/pdf/2407.03045](https://arxiv.org/pdf/2407.03045)

摘要：Large Language Models (LLMs) have gained significant attention but also raised concerns due to the risk of misuse. Jailbreak prompts, a popular type of adversarial attack towards LLMs, have appeared and constantly evolved to breach the safety protocols of LLMs. To address this issue, LLMs are regularly updated with safety patches based on reported jailbreak prompts. However, malicious users often keep their successful jailbreak prompts private to exploit LLMs. To uncover these private jailbreak prompts, extensive analysis of large-scale conversational datasets is necessary to identify prompts that still manage to bypass the system's defenses. This task is highly challenging due to the immense volume of conversation data, diverse characteristics of jailbreak prompts, and their presence in complex multi-turn conversations. To tackle these challenges, we introduce JailbreakHunter, a visual analytics approach for identifying jailbreak prompts in large-scale human-LLM conversational datasets. We have designed a workflow with three analysis levels: group-level, conversation-level, and turn-level. Group-level analysis enables users to grasp the distribution of conversations and identify suspicious conversations using multiple criteria, such as similarity with reported jailbreak prompts in previous research and attack success rates. Conversation-level analysis facilitates the understanding of the progress of conversations and helps discover jailbreak prompts within their conversation contexts. Turn-level analysis allows users to explore the semantic similarity and token overlap between a singleturn prompt and the reported jailbreak prompts, aiding in the identification of new jailbreak strategies. The effectiveness and usability of the system were verified through multiple case studies and expert interviews.（大型语言模型 （LLM） 受到了广泛关注，但由于存在误用的风险，也引起了人们的担忧。越狱提示是一种针对 LLM 的流行对抗性攻击类型，已经出现并不断发展以违反 LLM 的安全协议。为了解决这个问题，LLM 会根据报告的越狱提示定期更新安全补丁。但是，恶意用户通常会将其成功的越狱提示保密，以利用 LLM。要发现这些私有越狱提示，需要对大规模对话数据集进行广泛分析，以识别仍然设法绕过系统防御的提示。由于对话数据量巨大、越狱提示的多种特性以及它们在复杂的多轮对话中的存在，因此这项任务极具挑战性。为了应对这些挑战，我们引入了 JailbreakHunter，这是一种可视化分析方法，用于识别大规模人类 LLM 对话数据集中的越狱提示。我们设计了一个工作流，其中包含三个分析级别：组级别、对话级别和轮次级别。群组级分析使用户能够掌握对话的分布情况，并使用多个标准识别可疑对话，例如与以往研究中报告的越狱提示的相似性以及攻击成功率。对话级分析有助于了解对话的进度，并有助于发现对话上下文中的越狱提示。轮次分析允许用户探索单轮次提示和报告的越狱提示之间的语义相似性和标记重叠，从而帮助识别新的越狱策略。该系统的有效性和可用性通过多个案例研究和专家访谈得到了验证。）



### PathSeeker: Exploring LLM Security Vulnerabilities with a Reinforcement Learning-Based Jailbreak Approach

链接：[PathSeeker: Exploring LLM Security Vulnerabilities with a Reinforcement Learning-Based Jailbreak Approach (arxiv.org)](https://arxiv.org/pdf/2409.14177)

摘要：In recent years, Large Language Models (LLMs) have gained widespread use, raising concerns about their security. Traditional jailbreak attacks, which often rely on the model internal information or have limitations when exploring the unsafe behavior of the victim model, limiting their reducing their general applicability. In this paper, we introduce PathSeeker, a novel black-box jailbreak method, which is inspired by the game of rats escaping a maze. We think that each LLM has its unique "security maze", and attackers attempt to find the exit learning from the received feedback and their accumulated experience to compromise the target LLM's security defences. Our approach leverages multi-agent reinforcement learning, where smaller models collaborate to guide the main LLM in performing mutation operations to achieve the attack objectives. By progressively modifying inputs based on the model's feedback, our system induces richer, harmful responses. During our manual attempts to perform jailbreak attacks, we found that the vocabulary of the response of the target model gradually became richer and eventually produced harmful responses. Based on the observation, we also introduce a reward mechanism that exploits the expansion of vocabulary richness in LLM responses to weaken security constraints. Our method outperforms five state-of-the-art attack techniques when tested across 13 commercial and open-source LLMs, achieving high attack success rates, especially in strongly aligned commercial models like GPT-4o-mini, Claude-3.5, and GLM-4-air with strong safety alignment. This study aims to improve the understanding of LLM security vulnerabilities and we hope that this sturdy can contribute to the development of more robust defenses.（近年来，大型语言模型 （LLM） 得到了广泛使用，引发了对其安全性的担忧。传统的越狱攻击，往往依赖于模型内部信息，或者在探索受害者模型的不安全行为时有局限性，限制了它们，降低了它们的普遍适用性。在本文中，我们介绍了 PathSeeker，这是一种新颖的黑盒越狱方法，其灵感来自老鼠逃离迷宫的游戏。我们认为每个 LLM 都有其独特的“安全迷宫”，攻击者试图从收到的反馈和他们积累的经验中找到出口，以破坏目标 LLM 的安全防御。我们的方法利用多代理强化学习，其中较小的模型协作指导主 LLM 执行突变操作以实现攻击目标。通过根据模型的反馈逐步修改输入，我们的系统会引发更丰富、更有害的反应。在我们手动尝试执行越狱攻击的过程中，我们发现目标模型响应的词汇量逐渐变得更丰富，并最终产生有害的反应。基于观察，我们还引入了一种奖励机制，该机制利用 LLM 响应中词汇丰富性的扩展来削弱安全约束。在 13 个商业和开源 LLM 中进行测试时，我们的方法优于五种最先进的攻击技术，实现了很高的攻击成功率，尤其是在 GPT-4o-mini、Claude-3.5 和 GLM-4-air 等具有很强安全性对齐的强对齐商业模式中。本研究旨在提高对 LLM 安全漏洞的理解，我们希望这种坚固的漏洞有助于开发更强大的防御措施。）



### How Alignment and Jailbreak Work: Explain LLM Safety through Intermediate Hidden States

链接：[arxiv.org/pdf/2406.05644](https://arxiv.org/pdf/2406.05644)

摘要：Large language models (LLMs) rely on safety alignment to avoid responding to malicious user inputs. Unfortunately, jailbreak can circumvent safety guardrails, resulting in LLMs generating harmful content and raising concerns about LLM safety. Due to language models with intensive parameters often regarded as black boxes, the mechanisms of alignment and jailbreak are challenging to elucidate. In this paper, we employ weak classifiers to explain LLM safety through the intermediate hidden states. We first confirm that LLMs learn ethical concepts during pre-training rather than alignment and can identify malicious and normal inputs in the early layers. Alignment actually associates the early concepts with emotion guesses in the middle layers and then refines them to the specific reject tokens for safe generations. Jailbreak disturbs the transformation of early unethical classification into negative emotions. We conduct experiments on models from 7B to 70B across various model families to prove our conclusion. Overall, our paper indicates the intrinsical mechanism of LLM safety and how jailbreaks circumvent safety guardrails, offering a new perspective on LLM safety and reducing concerns.（大型语言模型 （LLM） 依靠安全对齐来避免响应恶意用户输入。不幸的是，越狱可以绕过安全护栏，导致 LLM 生成有害内容并引发对 LLM 安全性的担忧。由于具有密集参数的语言模型通常被视为黑盒，因此对齐和越狱的机制很难阐明。在本文中，我们采用弱分类器通过中间隐藏状态来解释 LLM 的安全性。我们首先确认 LLM 在预训练期间学习道德概念，而不是对齐，并且可以在早期识别恶意和正常输入。对齐实际上将早期概念与中间层的情感猜测联系起来，然后将它们提炼为特定的拒绝标记，以实现安全世代。越狱扰乱了早期不道德分类向负面情绪的转变。我们对各种模型族的 7B 到 70B 模型进行了实验，以证明我们的结论。总的来说，我们的论文指出了 LLM 安全的内在机制以及越狱如何绕过安全护栏，为 LLM 安全和减少担忧提供了新的视角。）

代码：[GitHub - ydyjya/LLM-IHS-Explanation](https://github.com/ydyjya/LLM-IHS-Explanation)



### Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation

链接：[2310.06987 (arxiv.org)](https://arxiv.org/pdf/2310.06987)

摘要：The rapid progress in open-source large language models (LLMs) is significantly advancing AI development. Extensive efforts have been made before model release to align their behavior with human values, with the primary goal of ensuring their helpfulness and harmlessness. However, even carefully aligned models can be manipulated maliciously, leading to unintended behaviors, known as "jailbreaks". These jailbreaks are typically triggered by specific text inputs, often referred to as adversarial prompts. In this work, we propose the generation exploitation attack, an extremely simple approach that disrupts model alignment by only manipulating variations of decoding methods. By exploiting different generation strategies, including varying decoding hyper-parameters and sampling methods, we increase the misalignment rate from 0% to more than 95% across 11 language models including LLaMA2, Vicuna, Falcon, and MPT families, outperforming state-of-the-art attacks with 30× lower computational cost. Finally, we propose an effective alignment method that explores diverse generation strategies, which can reasonably reduce the misalignment rate under our attack. Altogether, our study underscores a major failure in current safety evaluation and alignment procedures for open-source LLMs, strongly advocating for more comprehensive red teaming and better alignment before releasing such models.（开源大型语言模型 （LLM） 的快速发展正在显著推动 AI 的发展。在模型发布之前，已经做出了广泛的努力，使它们的行为与人类价值观保持一致，主要目标是确保它们的有用性和无害性。但是，即使是仔细对齐的模型也可能被恶意操纵，从而导致意外行为，称为“越狱”。这些越狱通常由特定的文本输入触发，通常称为对抗性提示。在这项工作中，我们提出了生成利用攻击，这是一种非常简单的方法，它仅通过操纵解码方法的变体来破坏模型对齐。通过利用不同的生成策略，包括不同的解码超参数和采样方法，我们将 LLaMA2、Vicuna、Falcon 和 MPT 系列等 11 种语言模型的错位率从 0% 提高到 95% 以上，性能优于最先进的攻击30×更低的计算成本。最后，我们提出了一种有效的对齐方法，探索了不同的生成策略，可以合理降低我们攻击下的错位率。总而言之，我们的研究强调了当前开源 LLM 安全评估和对齐程序的重大失败，强烈主张在发布此类模型之前进行更全面的红队划分和更好的对齐。）

代码：[GitHub - Princeton-SysML/Jailbreak_LLM](https://github.com/Princeton-SysML/Jailbreak_LLM)



### Jailbreaking Black Box Large Language Models in Twenty Queries

链接：[2310.08419](https://arxiv.org/pdf/2310.08419)

摘要：There is growing interest in ensuring that large language models (LLMs) align with human values. However, the alignment of such models is vulnerable to adversarial jailbreaks, which coax LLMs into overriding their safety guardrails. The identification of these vulnerabilities is therefore instrumental in understanding inherent weaknesses and preventing future misuse. To this end, we propose Prompt Automatic Iterative Refinement (PAIR), an algorithm that generates semantic jailbreaks with only black-box access to an LLM. PAIR -- which is inspired by social engineering attacks -- uses an attacker LLM to automatically generate jailbreaks for a separate targeted LLM without human intervention. In this way, the attacker LLM iteratively queries the target LLM to update and refine a candidate jailbreak. Empirically, PAIR often requires fewer than twenty queries to produce a jailbreak, which is orders of magnitude more efficient than existing algorithms. PAIR also achieves competitive jailbreaking success rates and transferability on open and closed-source LLMs, including GPT-3.5/4, Vicuna, and Gemini.（人们对确保大型语言模型 （LLM） 与人类价值观保持一致的兴趣日益浓厚。然而，这种模型的对齐很容易受到对抗性越狱的影响，这会诱使 LLM 推翻他们的安全护栏。因此，识别这些漏洞有助于了解固有的弱点并防止未来的滥用。为此，我们提出了提示自动迭代优化 （PAIR），这是一种生成语义越狱的算法，只需黑盒访问 LLM。PAIR - 受到社会工程攻击的启发 - 使用攻击者 LLM 为单独的目标 LLM 自动生成越狱，而无需人工干预。通过这种方式，攻击者 LLM 迭代查询目标 LLM 以更新和优化候选越狱。根据经验，PAIR 通常需要不到 20 个查询来产生越狱，这比现有算法的效率高出几个数量级。PAIR 还在开源和闭源 LLM（包括 GPT-3.5/4、Vicuna 和 Gemini）上实现了具有竞争力的越狱成功率和可转移性。）



### Weak-to-Strong Jailbreaking on Large Language Models

链接：[Weak-to-Strong Jailbreaking on Large Language Models Content warning: This paper contains examples of harmful language.](https://arxiv.org/pdf/2401.17256)

摘要：Large language models (LLMs) are vulnerable to jailbreak attacks - resulting in harmful, unethical, or biased text generations. However, existing jailbreaking methods are computationally costly. In this paper, we propose the weak-to-strong jailbreaking attack, an efficient method to attack aligned LLMs to produce harmful text. Our key intuition is based on the observation that jailbroken and aligned models only differ in their initial decoding distributions. The weak-to-strong attack's key technical insight is using two smaller models (a safe and an unsafe one) to adversarially modify a significantly larger safe model's decoding probabilities. We evaluate the weak-to-strong attack on 5 diverse LLMs from 3 organizations. The results show our method can increase the misalignment rate to over 99% on two datasets with just one forward pass per example. Our study exposes an urgent safety issue that needs to be addressed when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging. （大型语言模型 （LLM） 容易受到越狱攻击，从而导致有害、不道德或有偏见的文本生成。但是，现有的越狱方法在计算上成本很高。在本文中，我们提出了从弱到强的越狱攻击，这是一种攻击对齐的 LLM 以产生有害文本的有效方法。我们的主要直觉是基于这样一个观察，即越狱模型和对齐模型仅在其初始解码分布上有所不同。从弱到强攻击的关键技术见解是使用两个较小的模型（一个安全模型和一个不安全模型）来对抗性地修改一个明显更大的安全模型的解码概率。我们评估了对来自 3 个组织的 5 个不同 LLM 的从弱到强的攻击。结果表明，我们的方法可以将两个数据集的错位率提高到 99% 以上，每个样本只需一次前向传递。我们的研究揭示了在调整 LLM 时需要解决的紧迫安全问题。作为初步尝试，我们提出了一种防御策略来抵御此类攻击，但创建更高级的防御仍然具有挑战性。）

代码：[GitHub - XuandongZhao/weak-to-strong: Weak-to-Strong Jailbreaking on Large Language Models](https://github.com/XuandongZhao/weak-to-strong)

