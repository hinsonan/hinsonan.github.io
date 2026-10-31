#!/usr/bin/env bash
# Notify Bing, Yandex, and other IndexNow search engines that URLs changed.
# Run after GitHub Pages finishes deploying a new or updated post.
#
# Usage:
#   scripts/indexnow.sh https://gradientrage.com/ml/2026/11/01/rubber-duck-documentation.html [more urls...]
#   scripts/indexnow.sh --all     # submit every URL in the live sitemap
set -euo pipefail

HOST="gradientrage.com"
KEY="4796d92e7308419356ccae94fea9d4be"
KEY_LOCATION="https://${HOST}/${KEY}.txt"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 URL [URL...] | --all" >&2
  exit 2
fi

if [[ "$1" == "--all" ]]; then
  mapfile -t urls < <(curl -fsSL "https://${HOST}/sitemap.xml" | grep -o '<loc>[^<]*' | sed 's/<loc>//')
else
  urls=("$@")
fi

url_list=$(printf '"%s",' "${urls[@]}")
payload=$(printf '{"host":"%s","key":"%s","keyLocation":"%s","urlList":[%s]}' \
  "$HOST" "$KEY" "$KEY_LOCATION" "${url_list%,}")

status=$(curl -s -o /dev/null -w '%{http_code}' -X POST "https://api.indexnow.org/indexnow" \
  -H 'Content-Type: application/json; charset=utf-8' --data "$payload")

echo "IndexNow responded ${status} for ${#urls[@]} URL(s)"
[[ "$status" == "200" || "$status" == "202" ]]
