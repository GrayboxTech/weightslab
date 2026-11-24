#!/usr/bin/env bash
# Which weights_studio branch a release vendors as its UI, in this order:
#
#   1. "@<branch>" in the tag's message (annotated tag) or in the tagged
#      commit's message -- e.g. "2.0.5 pre-release @feature/foo" -- when
#      weights_studio has that branch;
#   2. the weights_studio branch named like the weightslab branch the tag sits
#      on, i.e. a branch whose tip is the tagged commit (a dev tag prefers its
#      own branch over main; a release tag is on main, main first);
#   3. otherwise the default: main.
#
# There is no dev branch any more (removed after 2.0.5): main is the only
# long-lived branch in both repositories.
#
# A "@name" that is no weights_studio branch (a person's handle, a typo) is
# skipped with a note, never an error.
#
# Usage: resolve_ui_ref.sh <weights_studio clone url> [default ref, main]
# Prints the chosen branch on stdout; the reasoning goes to stderr (job log).
set -euo pipefail

CLONE_URL="$1"
DEFAULT_REF="${2:-main}"
TAG="${GITHUB_REF_NAME:-}"

log() { echo "$*" >&2; }

# weights_studio's branches, listed once. Retried, and fatal when it keeps
# failing: a network blip must not silently ship a different UI.
#
# Listed from OUTSIDE the weightslab checkout: actions/checkout stores this
# repo's GITHUB_TOKEN as an http extraheader in the checkout's local git
# config, and inside it that header replaces the weights_studio token in
# CLONE_URL -- GitHub then answers "Repository not found" for the private
# repo. (git clone is unaffected: it does not read the current repo's config.)
LISTING_DIR="${RUNNER_TEMP:-/tmp}"
heads=""
for attempt in 1 2 3; do
  if heads="$(cd "${LISTING_DIR}" && git ls-remote --heads "${CLONE_URL}")" && [ -n "${heads}" ]; then
    break
  fi
  heads=""
  log "Listing weights_studio branches failed (attempt ${attempt}/3)"
  sleep 5
done
if [ -z "${heads}" ]; then
  log "::error::Could not list the weights_studio branches; not guessing which UI to ship."
  exit 1
fi
ws_branches="$(printf '%s\n' "${heads}" | sed -n 's#^[0-9a-f]*[[:space:]]*refs/heads/##p')"
ws_has() { printf '%s\n' "${ws_branches}" | grep -Fxq -- "$1"; }

# actions/checkout fetches a pushed tag as a lightweight ref, without its
# message (the v2.0.5.dev0 run never saw its "@v2.0.5--fx"). Fetch the tag
# object itself so an annotated tag's message can name the branch.
if [ -n "${TAG}" ]; then
  git fetch --quiet --no-tags origin "+refs/tags/${TAG}:refs/tags/${TAG}" \
    || log "Could not fetch the tag ${TAG}; reading the local tag and the commit message"
fi

if [ -n "${TAG}" ] && git rev-parse -q --verify "refs/tags/${TAG}" >/dev/null; then
  TAG_COMMIT="$(git rev-list -n 1 "refs/tags/${TAG}")"
else
  TAG_COMMIT="$(git rev-parse HEAD)"
fi

# 1. "@branch" in the tag message or the tagged commit's message.
messages="$(
  { [ -n "${TAG}" ] && git for-each-ref "refs/tags/${TAG}" --format='%(contents)'; } || true
  git log -1 --format=%B "${TAG_COMMIT}"
)"
mentions="$(printf '%s\n' "${messages}" | grep -oE '(^|[[:space:]])@[A-Za-z0-9._/-]+' | sed -E 's/^[[:space:]]*@//' || true)"
for name in ${mentions}; do
  for candidate in "${name}" "${name%.}"; do   # "@feature/x." at the end of a sentence
    if ws_has "${candidate}"; then
      log "weights_studio ref: ${candidate} (\"@${candidate}\" in the tag/commit message)"
      echo "${candidate}"
      exit 0
    fi
  done
  log "\"@${name}\" in the tag/commit message is no weights_studio branch; ignored"
done

# 2. The weights_studio branch named like the weightslab branch the tag is on.
git fetch --quiet --no-tags origin '+refs/heads/*:refs/remotes/origin/*' \
  || log "Could not refresh the weightslab branches; using what the checkout fetched"
here="$(git for-each-ref --points-at "${TAG_COMMIT}" --format='%(refname:lstrip=3)' refs/remotes/origin | grep -vx 'HEAD' || true)"
specific="$(printf '%s\n' "${here}" | grep -vx 'main' || true)"
on_main="$(printf '%s\n' "${here}" | grep -x 'main' || true)"
# A dev tag (its name contains "dev", as release.yml decides) prefers its own
# branch over main; a release tag takes main first.
if [[ "${TAG}" == *dev* ]]; then
  candidates="${specific} ${on_main}"
else
  candidates="${on_main} ${specific}"
fi
for name in ${candidates}; do
  if ws_has "${name}"; then
    log "weights_studio ref: ${name} (same name as the weightslab branch the tag is on)"
    echo "${name}"
    exit 0
  fi
done
if [ -n "${here// /}" ]; then
  log "No weights_studio branch named like the weightslab branch(es) at the tag: $(echo ${here})"
fi

# 3. Default.
log "weights_studio ref: ${DEFAULT_REF} (default)"
echo "${DEFAULT_REF}"
