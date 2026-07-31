#!/usr/bin/env bash
# backup_models.sh — push Mario model deliverables to Google Drive via rclone.
#
# Syncs the prune-immune model dirs (best peak snapshots, specialist finals,
# archive) to gdrive:mario_ai_backups so a box failure or a local pruning
# accident can't lose a trained model. Safe to run repeatedly (rclone copy is
# incremental — only changed/new files transfer).
#
# Usage:
#   ./backup_models.sh              # one-shot backup
#   watch -n 600 ./backup_models.sh # every 10 min (or drive from the monitor cron)
set -euo pipefail

MODELS="/home/ubuntu/ai_training/models"
REMOTE="gdrive:mario_ai_backups"
TS="$(TZ='Asia/Manila' date '+%Y-%m-%d %H:%M UTC+8')"

echo "[$TS] backing up models -> $REMOTE"

# copy (not sync) so a local delete never propagates to the backup — the whole
# point is that the remote keeps peaks even if local prunes them away.
for d in best specialists archive; do
  if [ -d "$MODELS/$d" ] && [ -n "$(ls -A "$MODELS/$d" 2>/dev/null)" ]; then
    rclone copy "$MODELS/$d" "$REMOTE/$d" --stats-one-line 2>&1 | tail -1 || true
    echo "  ✓ $d"
  fi
done

echo "[$TS] done. remote contents:"
rclone ls "$REMOTE" 2>&1
