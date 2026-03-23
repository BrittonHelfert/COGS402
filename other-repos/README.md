# other-repos

External repositories used in this project. Clone these manually:

```bash
cd other-repos

# Diffing-toolkit (ADL + agent evaluation)
git clone https://github.com/science-of-finetuning/diffing-toolkit.git
cd diffing-toolkit && uv sync && cd ..

# Latteries (LLM eval utilities)
git clone https://github.com/thejaminator/latteries.git
cd latteries && uv sync && cd ..
```

The `weird-generalization-and-inductive-backdoors` repo is tracked directly in this repo (not a submodule).
