#!/usr/bin/env bash
# Senmei version bump: keeps every version site in sync. Run from anywhere.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

USAGE="Usage:
  tools/bump-version.sh <x.y.z> [--changelog]   bump every version site
  tools/bump-version.sh --check                 verify all sites agree
  tools/bump-version.sh --show                  print current version per site"

VERSION_RE='^[0-9]+\.[0-9]+\.[0-9]+$'

# Version sites. Canonical source: root Cargo.toml ([workspace.package]).
# All crates inherit via version.workspace = true; the other files must match.
CARGO_VERSION='Cargo.toml'
CRATE_MANIFESTS='crates/*/Cargo.toml'          # internal path-dep pins
TAURI_CONF='crates/senmei/tauri.conf.json'      # bundle version (installer names)
APP_PKG='packages/app/package.json'             # __APP_VERSION__ badge
CHANGELOG='docs/CHANGELOG.md'

current_version() {
  awk '
    /^\[workspace\.package\]/ { in_pkg = 1; next }
    in_pkg && /^version = / { split($0, a, "\""); print a[2]; exit }
  ' "$CARGO_VERSION"
}

json_version() { # <file> -> "x.y.z"
  grep -o '"version": *"[^"]*"' "$1" | head -1 | sed 's/.*"\([^"]*\)"$/\1/'
}

# Unique version values on internal path-dep lines (path = "../senmei…").
crate_pin_versions() {
  grep -h 'path = "\.\./senmei' $CRATE_MANIFESTS |
    grep -o 'version = "[^"]*"' | sort -u
}

pins_ok() { # <version>
  local out; out="$(crate_pin_versions)"
  [[ -n "$out" && "$(printf '%s\n' "$out" | wc -l)" -eq 1 && "$out" == "version = \"$1\"" ]]
}

check() {
  local v tauri app ok=1
  v="$(current_version)"
  tauri="$(json_version "$TAURI_CONF")"
  app="$(json_version "$APP_PKG")"

  printf '%-46s %s\n' 'site' 'version'
  printf '%-46s %s\n' '---' '---'
  printf '%-46s %s\n' "Cargo.toml ([workspace.package])" "$v"
  printf '%-46s %s\n' "crates/*/Cargo.toml (path-dep pins)" "$(crate_pin_versions | tr '\n' ' ' | sed 's/ *$//')"
  printf '%-46s %s\n' "$TAURI_CONF" "$tauri"
  printf '%-46s %s\n' "$APP_PKG" "$app"

  [[ -n "$v" ]]        || { echo "error: no version in $CARGO_VERSION" >&2; ok=0; }
  [[ "$tauri" == "$v" ]] || { echo "MISMATCH: $TAURI_CONF=$tauri (want $v)" >&2; ok=0; }
  [[ "$app" == "$v" ]]   || { echo "MISMATCH: $APP_PKG=$app (want $v)" >&2; ok=0; }
  pins_ok "$v"           || { echo "MISMATCH: crate path-dep pins ($(crate_pin_versions | tr '\n' ' ')) (want $v)" >&2; ok=0; }

  (( ok )) && echo "OK: all version sites agree on $v"
  return $(( ok ? 0 : 1 ))
}

inplace() { # <sed-expr> <file> — portable in-place edit (no BSD/GNU -i gap)
  local expr="$1" f="$2"
  sed "$expr" "$f" > "$f.tmp" && mv "$f.tmp" "$f"
}

bump() {
  local new="$1" old; old="$(current_version)"
  if ! check >/dev/null; then
    echo "error: version sites out of sync — run --check first" >&2
    exit 1
  fi
  if [[ "$(printf '%s\n%s\n' "$old" "$new" | sort -V | head -1)" != "$old" ]]; then
    echo "warning: $new is not greater than $old" >&2
  fi

  inplace "s/^version = \"$old\"/version = \"$new\"/" "$CARGO_VERSION"
  # Only internal path-dep lines; external deps (serde = { version = … }) never
  # sit on a path = "../senmei…" line, so the address keeps them untouched.
  for f in $CRATE_MANIFESTS; do
    inplace '/path = "\.\.\/senmei/s/version = "'"$old"'"/version = "'"$new"'"/g' "$f"
  done
  inplace 's/"version": "'"$old"'"/"version": "'"$new"'"/' "$TAURI_CONF"
  inplace 's/"version": "'"$old"'"/"version": "'"$new"'"/' "$APP_PKG"

  check
  echo "next: cargo check --workspace (refreshes Cargo.lock + validates)"
}

add_changelog_heading() {
  local new="$1" date; date="$(date +%F)"
  local tmp; tmp="$(mktemp)"
  awk -v new="$new" -v date="$date" '
    BEGIN { hdr = "## " new " (" date ")"; done = 0 }
    /^## [0-9]/ && !done {
      print ""; print hdr; print ""
      print "- **release: " new "** —"
      print ""; print ""; done = 1
    }
    { print }
  ' "$CHANGELOG" > "$tmp" && mv "$tmp" "$CHANGELOG"
  echo "added CHANGELOG heading: ## $new ($date)"
}

case "${1:-}" in
  --check)  check ;;
  --show)   check || true ;;
  -h|--help) echo "$USAGE" ;;
  *)
    if [[ ! "${1:-}" =~ $VERSION_RE ]]; then echo "$USAGE" >&2; exit 2; fi
    bump "$1"
    [[ "${2:-}" == "--changelog" ]] && add_changelog_heading "$1"
    ;;
esac
