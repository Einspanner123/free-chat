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
- Source governance and the current consolidation boundary are in docs/source-governance.md.
