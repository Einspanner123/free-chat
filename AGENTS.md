# FreeChat development location

- Edit FreeChat only on ross in /home/linkst/workspace/projects/free-chat on main.
- Edit the independent vLLM repository only through third_party/vllm in that checkout.
- Do not create another development branch/worktree or edit frozen mirrors unless explicitly requested.
- This is a development service: stop FreeChat when needed for authorized changes; do not turn production rollout procedure into a prerequisite for ordinary development.
- Next work is the runnable inference loop in section 0 of the root plan, not additional source-governance infrastructure.
- Workstation is a runtime/test target; do not edit source there.
- Preserve unknown working changes, .env, plans, and raw evidence.
- Develop engine changes in the full third_party/vllm submodule, following its AGENTS.md.
- Publish fork objects before updating the parent gitlink; keep versions.lock.yaml aligned.
- Never confuse source synchronization with deployment authorization.
- Record substantive merge conflicts and obtain the owner's decision before resolving them.
- Keep one authoritative root plan; operational instructions are in docs/operations.md.
- CPU tests cover partial contracts only; real execution and full acceptance require GPU tests.

- Validation commands must print results to stdout and use normal service logs; never create new result directories.
- Validated feature work must converge to main. Keep README aligned with the actual runnable entry point.
