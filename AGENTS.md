# FreeChat development location

- Edit FreeChat and its vLLM fork only on ross.
- Workstation is a runtime/test target; do not edit source there.
- Preserve unknown working changes, .env, plans, and raw evidence.
- Develop engine changes in the full third_party/vllm submodule, following its AGENTS.md.
- Publish fork objects before updating the parent gitlink; keep versions.lock.yaml aligned.
- Never confuse source synchronization with deployment authorization.
- Record substantive merge conflicts and obtain the owner's decision before resolving them.
- Source governance and the current consolidation boundary are in docs/source-governance.md.
