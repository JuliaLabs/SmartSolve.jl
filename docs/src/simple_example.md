## Generate LU with iterative refinement
```julia
using SmartSolve

prompt = """
Generate a high performance Julia implementation of LU
with iterative refinement using the following reference:
https://nhigham.com/2023/03/13/what-is-iterative-refinement
"""

secret_key = ENV["OPENAI_API_KEY"]

code, hist, conv = generate(prompt, secret_key)

```
