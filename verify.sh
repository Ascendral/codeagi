#!/usr/bin/env bash
# verify.sh — KlomboAGI verification script
# No measurement = no claim.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PASS=0
FAIL=0
WARN=0

pass() { echo -e "${GREEN}[PASS]${NC} $1"; PASS=$((PASS + 1)); }
fail() { echo -e "${RED}[FAIL]${NC} $1"; FAIL=$((FAIL + 1)); }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; WARN=$((WARN + 1)); }

RUN_SCORE=false
QUICK=false

for arg in "$@"; do
    case "$arg" in
        --score) RUN_SCORE=true ;;
        --quick) QUICK=true ;;
        *) echo "Usage: ./verify.sh [--score] [--quick]"; exit 1 ;;
    esac
done

echo "=== KlomboAGI Verification ==="
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# 1. Import check
echo "--- Import Check ---"
if python3 -c "import klomboagi" 2>/dev/null; then
    pass "klomboagi package imports successfully"
else
    fail "klomboagi package import failed"
fi

# 2. _try_ function count
echo ""
echo "--- _try_ Function Census ---"
TRY_COUNT=$({ grep -r "def _try_" klomboagi/ --include="*.py" 2>/dev/null || true; } | wc -l | tr -d ' ')
echo "Total _try_* function definitions: $TRY_COUNT"
if [ "$TRY_COUNT" -gt 0 ]; then
    warn "$TRY_COUNT hand-coded _try_* functions (Level 0 — Manual)"
fi

# Top files by _try_ count
echo "Top files:"
{ grep -r "def _try_" klomboagi/ --include="*.py" -l 2>/dev/null || true; } | while read -r f; do
    count=$(grep -c "def _try_" "$f" 2>/dev/null || echo 0)
    echo "  $count  $f"
done | sort -rn | head -5

# 3. Dead code detection
echo ""
echo "--- Dead Code Check ---"

# Check if connective is imported by solver
if grep -r "from klomboagi.connective\|import klomboagi.connective" klomboagi/reasoning/ --include="*.py" -q 2>/dev/null; then
    pass "connective/ is imported by reasoning module"
else
    warn "connective/ is NOT imported by reasoning module — potentially dead code"
fi

# Check if brain_core is called from Python
if grep -r "brain_core\|libklombo" klomboagi/ --include="*.py" -q 2>/dev/null; then
    pass "brain_core (Rust) is referenced in Python code"
else
    warn "brain_core (Rust) is NOT referenced in Python code — potentially dead code"
fi

# Check runtime directory usage
if grep -r "runtime/" klomboagi/ --include="*.py" -q 2>/dev/null; then
    pass "runtime/ directory is referenced in Python code"
else
    warn "runtime/ directory is NOT referenced in Python code — potentially dead code"
fi

# 4. Credential scan
echo ""
echo "--- Credential Scan ---"
CRED_PATTERNS='(password|passwd|secret|api[_-]?key|apikey|token|credential)[[:space:]]*[:=][[:space:]]*["'"'"'"'"'"'"'"'"'][^"'"'"'"'"'"'"'"'"']+["'"'"'"'"'"'"'"'"']'
CRED_HITS=$(grep -rniE "$CRED_PATTERNS" --include="*.py" --include="*.sh" --include="*.json" --include="*.yaml" --include="*.yml" --include="*.toml" --include="*.cfg" --include="*.ini" 2>/dev/null | grep -vi "test\|example\|sample\|template\|\.md\|comment" | head -20 || true)
if [ -n "$CRED_HITS" ]; then
    fail "Potential hardcoded credentials found:"
    echo "$CRED_HITS" | head -5
else
    pass "No obvious hardcoded credentials in tracked files"
fi

# 5. Test suite
echo ""
echo "--- Test Suite ---"
TRACKED_TESTS=()
while IFS= read -r tracked_test; do
    TRACKED_TESTS+=("$tracked_test")
done < <(git ls-files 'tests/test_*.py')

if [ "${#TRACKED_TESTS[@]}" -gt 0 ]; then
    if python3 -m pytest "${TRACKED_TESTS[@]}" -x -q 2>/dev/null; then
        pass "Test suite passes"
    else
        fail "Test suite has failures"
    fi
else
    warn "No tracked test files found in tests/"
fi

# 6. Score verification (optional)
if [ "$RUN_SCORE" = true ] && [ "$QUICK" = false ]; then
    echo ""
    echo "--- ARC Score Verification ---"
    echo "Running full eval (this may take a while)..."
    if python3 -m klomboagi.evals.arc_eval 2>&1; then
        pass "ARC eval completed"
    else
        fail "ARC eval failed to run"
    fi
fi

# Summary
echo ""
echo "=== Summary ==="
echo -e "${GREEN}Passed: $PASS${NC}"
echo -e "${RED}Failed: $FAIL${NC}"
echo -e "${YELLOW}Warnings: $WARN${NC}"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo -e "${RED}VERIFICATION FAILED — Do not claim success.${NC}"
    exit 1
fi

if [ "$WARN" -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Warnings present — investigate before claiming.${NC}"
fi

exit 0
