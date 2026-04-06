## **Attractor states experiments**

RQs:

1) Does self-talk naturally elicit “weird” side effects of narrow finetuning? Do the attractor states tend to reflect the narrow task or the learned generalization?  
   1) More generally, what can attractor states tell us about models?  
2) Can we integrate self-talk with other techniques (e.g. diffing and steering) to enable LLM agents to uncover important model quirks?

EXPERIMENTS:

1) EM / Weird Gen / Triggers / other organisms  
   1) Prompts that indirectly try to elicit the finetuning goal / learned generalization (e.g. “talk about your unique quirks”,  “talk about yourself” etc)  
   2) EM Models: compare rates of observed EM in normal eval setup from EM work (i.e. one model with EM eval prompt, maybe telling it “continue” for 30 turns) vs self-talk setup (seeding with EM eval prompt (and other seed prompts?))  
   3) Steer with vectors derived from diffing model organisms w their base models  
      1) Vary the conversation step at which the models are steered  
      2) try steering base models, instruct models, and the organisms themselves  
      3) Can we steer with other vectors? A “confession” vector? A “talk about myself” / self-concept vector?  
   4) Does this technique make models reveal backdoor triggers? Or can it be leveraged in a more complex pipeline to do this? How does it compare to a method like ADL?  
2) Generally fleshing out our understanding of this, e.g. questions like:  
   1) How does the prompt affect the outcome  
      1) Explicitly tell them to try not to enter a cycle  
      2) Tell them they’re in an attractor states experiment  
      3) Different tones / wordings; e.g. more clinical  
      4) Asymmetric system prompt setup: one instance has the system prompt "try to change topics frequently," the other is standard.  
   2) Does this happen in base models  
   3) How does model size affect it  
   4) How does temperature affect it  
   5) Are some models more consistent than others w/r/t topics of convergence?  
   6) Reasoning models: Let them think\! But only send the final output to each model. However keep the chain of thought for analysis  
   7) Mid-conversation interruptions (probing the stability of these states)  
      1) Minimal (e.g. “continue” “clarify”, or “I disagree.”)  
      2) Reset (e.g. “ignore all prior messages and begin anew.”)  
      3) Topic redirection (e.g. “switch topic: medieval taxation systems.”, “what is 374 \* 829?”)  
      4) Meta-awareness (e.g. “this conversation has entered a cycle.”)  
      5) Constraint injection (e.g. “do not repeat any previous words.”)  
      6) Noise injection (inject random tokens, scramble previous message)  
   8) Single-instance self-talk, i.e. feed the model its own output back as the user turn   
   9) Truncated context, i.e. each turn only sees the last N turns rather than the full history  
3) Cross-model  
   1) TODO flesh out specific questions to ask, how to answer them, expected results

TODO

1. Judging pipeline  
   1. Set up API  
   2. Personally edit judge prompts  
   3. Separate between semantic and structural attractors (structural implies semantic but not other way around)  
      1. For semantic, ask for a list of up to 5 core topics / themes. Each should be max 3 words. Save them to a theme bank (global? Or experiment-level?). When judging, can draw from theme bank or add new entries if none fit.   
   4. Add in automatic aggregation  
   5. Brainstorm quant. methods  
   6. More LLM judge formats?  
   7. How quickly does it converge? Is this primarily a function of overall coherence? Use coherence judge from EM

Code changes:

- General  
  - Vectors prob dont need to be a config  
- Run\_organism.py \- more modular and modifiable way of doing experiments  
  - Similarly with slurm submit scripts  
- SLURM:  
  - Current time limit heuristic is brittle

Analysis:

- For a given experiment we want to:  
- Classify each convo as: \[no convergence, topic convergence but acyclic, cyclic with paraphrasing, pure cyclic\]  
- Categorize the “flavor” of convergence  
- See if an agent can determine the fine tuning objective  
- See if it elicits behavior of interest in backdoored models / weird gen models  
- Quantitative metrics: similarity in activations across sequence (between model and within model)  
  - KL divergence between successive messages from the same model?