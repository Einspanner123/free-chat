# Locked fork source unavailable in the recorded repository

Status: **RESOLVED / exact pinned source recovered and verified locally**

## Intended operation and authority

The owner authorized read-only SSH inspection/export of the pinned fork from ross
for local implementation. No remote file changes, dependency installation, services,
model execution or GPU tests were authorized by this exception to the local-only phase.

## Observed evidence, 2026-09-20

Local `versions.lock.yaml` declares:

- Repository: `/media/ross/8TB/linkst/freechat/vllm-fork.git`
- Branch: `freechat-agent-aware`
- Fork commit: `8e78a3c613072632aa822c9aed2f698e76046219`
- Upstream commit: `9c22668436a4d94aab87ea74a220e060415cf1d8`

Read-only Git commands over `ssh linkst@ross` returned:

| Check | Observed result |
|---|---|
| `rev-parse --is-bare-repository` | `true` |
| `show -s --format=fuller 8e78a3c613072632aa822c9aed2f698e76046219` | `fatal: bad object 8e78a3c613072632aa822c9aed2f698e76046219`, exit 128 |
| `show-ref --heads --tags` | Only `5871f38ee947760ed7f5dd5487bb04456db8bc98 refs/heads/freechat-agent-aware` |
| `rev-parse --is-shallow-repository` | `false` |

The inspected repository's recent reachable history begins with:

```text
5871f38ee947760ed7f5dd5487bb04456db8bc98 2026-09-01T18:00:09+08:00 feat: prioritize resumable agent cache blocks
1c191f8632f1bfdd57d5e1de3b749b60f824f98a 2026-08-31T20:42:37+08:00 feat: bind cache events to authenticated identity
160d987eaca8a47467acffb65269379ab270bc0d 2026-08-31T20:37:47+08:00 feat: expose agent lifecycle cache hooks
9c22668436a4d94aab87ea74a220e060415cf1d8 2026-08-06T02:23:54-07:00 [Quantization] Preserve precision in online NVFP4 expert packing (#50029)
```

These are inspection observations, not a complete repository integrity check. They
show that the pinned object cannot currently be resolved in the recorded repository;
they do not prove the commit never existed or is absent from another clone/container.
No archive was exported during that initial inspection because the requested commit could not be resolved. No refs,
objects, remote configuration or working trees were changed.

The local `rewrite-vllm/` mirror has selected files without Git metadata, so it cannot
establish full-tree equivalence to the pinned fork. The earlier public upstream query
also returned HTTP 403; this is an access result, not evidence of an invalid commit.

## Impact at initial discovery and protected state

Real engine admission/cancellation/completion integration cannot yet be tied to the
locked fork. Do not substitute the older branch head, update the lock, overlay partial
files onto upstream while claiming an exact fork, or reuse historical GPU evidence to
establish source equivalence. Existing local admission tests remain valid within their
recorded mechanism-only scope; they do not resolve this source provenance gap.

The implementation scope, version lock, image revision expectations and historical
evidence remain unchanged; the plan now links this open conflict. This does not withdraw
or revalidate earlier experiment claims; their
source/artifact provenance must be checked separately if the pinned source is lost.

## Proposed resolution options

1. Recommended: authorize additional **read-only** inspection of project-related Git
   clones on ross and workstation, starting from known project locations; locate the
   exact commit, verify its tree/ancestry, and export it to a separate local directory.
   No remote updates, checkouts, fetches, service operations or GPU workloads.
2. Supply a Git bundle or source archive with independently verifiable commit/tree
   provenance for the pinned fork. Inspect and continue locally.

Changing the selected fork revision is not an automatic fallback. If the exact source
cannot be recovered, record the proposed replacement, code differences and evidence
impact here and obtain explicit owner approval before changing the lock.

## Owner direction and follow-up inspection

The owner subsequently authorized continued read-only discovery on ross/workstation
("继续找"). No permission to mutate the remote repositories or run GPU workloads was
added. The following project-related paths were checked:

| Host/path | Result |
|---|---|
| ross `/home/linkst/workspace/projects/free-chat` | Application repository; pinned fork object unavailable |
| workstation `/home/linkst/workspace/freechat-vllm-fork` | HEAD `17539077af862832a2804429e9c0fd8896e274bc`; pinned object unavailable |
| workstation `/home/linkst/workspace/freechat-vllm-authoritative` | Exact pinned commit exists and is HEAD |

The project directory search encountered permission denied at workstation
`/data/freechat/docker`; that path was not accessed or retried with elevated permissions.
It was not needed to locate the source.

The recovered repository reports:

```text
commit: 8e78a3c613072632aa822c9aed2f698e76046219
tree:   45110e50047f52bef7628f2d616945fdb5244468
date:   2026-09-02T19:10:09+08:00
subject: build: fetch verified CUTLASS archive
merge-base with locked upstream: 9c22668436a4d94aab87ea74a220e060415cf1d8
```

There are eight fork commits after the locked upstream. Five follow the ross bare
repository's branch head: Anthropic lifecycle propagation, then four build fixes.
The complete upstream-to-fork diff covers 15 files, with 750 insertions and 14 deletions.
The worktree showed no tracked changes (`GIT_OPTIONAL_LOCKS=0 git status --porcelain=v1
--untracked-files=no`); the export nevertheless uses only the exact commit, not the
working tree.

This resolves whether the pinned object can be found: it exists in a different
project clone. It does not establish why the ross repository was not synchronized.
The lock's commit is not being replaced; source-location metadata remains to be
reconciled separately.

## Verified local recovery

The owner-authorized discovery/export recovered the exact pinned commit, without
changing the selected revision or any remote files. Local workspace artifacts are in
`outputs/vllm-source-8e78a3-7Vc7HM/` (alongside, not inside, `rewrite/`):

- `source.tar`: `git archive --format=tar` of the exact commit;
- `commit.raw`: original Git commit object body;
- `tree.nul`: NUL-delimited complete recursive Git tree listing;
- `verify_export.py` and `verification.json`: reproducible verification and result;
- `source/`: separately extracted complete source snapshot, with no Git working tree
  changes overlaid and no source execution or dependency installation.

The remote and local archive SHA-256 both equal:

```text
c91947af1a3c0f436abf0c4c6e5738fa20e4ca2981a51381e5844420968b745c
```

Verification passed for all **6,404 Git blobs**: content hashes, regular/executable/
symlink modes, exact archive membership, recursively reconstructed tree hash, and
commit-object hash. The reconstructed tree is
`45110e50047f52bef7628f2d616945fdb5244468`; the recomputed commit is the requested
`8e78a3c613072632aa822c9aed2f698e76046219`. No submodule source omission was accepted.

The source-availability conflict is resolved. This is source provenance evidence,
not a signature verification, software test, GPU validation, image acceptance or HA
claim. The lock's repository-location metadata still names ross and should be aligned
with the verified source/repository publication policy before the next build workflow;
the commit, version lock and remote refs were not modified in this recovery task.
